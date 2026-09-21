//! Dump frozen-trunk pooled features for a labelled JSONL split.
//!
//! The linear-probe experiment: if a FROZEN backbone's pooled vectors already
//! separate our classes linearly, a specialist costs `[num_labels, 1024]`
//! weights (12 KiB) instead of a 1.4 GB full fine-tune — and, more decisively, one
//! trunk forward serves every head instead of one forward per specialist.
//!
//! This binary does only the expensive half (the trunk forward); fitting the
//! head is downstream and operates on the cached matrices.
//!
//! Usage:
//!   cargo run --release --example frozen_features -- \
//!       --model <ckpt-dir> --jsonl <split.jsonl> --out <features.safetensors>
//!
//! Emits a safetensors file with:
//!   - `cls`    `[n, hidden]` f32 — hidden state at position 0
//!   - `mean`   `[n, hidden]` f32 — mean over all positions
//!   - `labels` `[n]` u32 — class id under the FIXED order below
//!
//! Label ids match `runs/kube_ordinal_v6/config.json`'s `id2label` so a probe
//! trained here is directly comparable to that checkpoint's reported accuracy.
//! An unknown label string is a hard error: a silently dropped or misnumbered
//! row would move the accuracy we are trying to measure.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use lfm2_encoder::{Lfm2EncoderConfig, Lfm2Trunk};
use tokenizers::Tokenizer;

/// Fixed, matching v6's config. Order is the class id.
const LABELS: [&str; 3] = ["destructive", "informative", "mutating"];

fn label_id(s: &str) -> Result<u32> {
    LABELS
        .iter()
        .position(|l| *l == s)
        .map(|i| i as u32)
        .with_context(|| format!("unknown label {s:?}; expected one of {LABELS:?}"))
}

struct Args {
    model: PathBuf,
    jsonl: PathBuf,
    out: PathBuf,
    /// Prepended verbatim to every row. The Embedding checkpoint was trained
    /// with `"document: "` / `"query: "`; feeding it bare text measures a
    /// distribution it never saw.
    prefix: String,
}

fn parse_args() -> Result<Args> {
    let (mut model, mut jsonl, mut out) = (None, None, None);
    let mut prefix = String::new();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut take = |name: &str| -> Result<String> {
            it.next()
                .with_context(|| format!("{name} needs a value"))
        };
        match a.as_str() {
            "--model" => model = Some(PathBuf::from(take("--model")?)),
            "--jsonl" => jsonl = Some(PathBuf::from(take("--jsonl")?)),
            "--out" => out = Some(PathBuf::from(take("--out")?)),
            "--prefix" => prefix = take("--prefix")?,
            other => bail!("unknown argument {other:?}"),
        }
    }
    Ok(Args {
        model: model.context("--model is required")?,
        jsonl: jsonl.context("--jsonl is required")?,
        out: out.context("--out is required")?,
        prefix,
    })
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let device = Device::Cpu;

    let cfg_path = args.model.join("config.json");
    let cfg_bytes = std::fs::read(&cfg_path)
        .with_context(|| format!("reading {}", cfg_path.display()))?;
    let cfg = Lfm2EncoderConfig::from_json(&cfg_bytes)
        .with_context(|| format!("parsing {}", cfg_path.display()))?;
    cfg.validate()
        .map_err(|m| anyhow::anyhow!("{}: {m}", cfg_path.display()))?;

    let tokenizer = Tokenizer::from_file(args.model.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;

    let weights = args.model.join("model.safetensors");
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights], DType::F32, &device)? };
    // `load` probes the known prefixes itself (`lfm2.` for head-carrying
    // checkpoints, root for the bare encoder) and refuses an ambiguous one.
    let trunk = Lfm2Trunk::load(&cfg, vb.clone())?;

    let text = std::fs::read_to_string(&args.jsonl)
        .with_context(|| format!("reading {}", args.jsonl.display()))?;

    let mut cls_rows: Vec<Vec<f32>> = Vec::new();
    let mut mean_rows: Vec<Vec<f32>> = Vec::new();
    let mut labels: Vec<u32> = Vec::new();

    let started = std::time::Instant::now();
    for (lineno, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row: serde_json::Value = serde_json::from_str(line)
            .with_context(|| format!("{}:{}", args.jsonl.display(), lineno + 1))?;
        let s = row["text"]
            .as_str()
            .with_context(|| format!("{}:{}: no `text`", args.jsonl.display(), lineno + 1))?;
        let lab = row["label"]
            .as_str()
            .with_context(|| format!("{}:{}: no `label`", args.jsonl.display(), lineno + 1))?;
        labels.push(label_id(lab)?);

        let prefixed;
        let s = if args.prefix.is_empty() {
            s
        } else {
            prefixed = format!("{}{s}", args.prefix);
            &prefixed
        };
        let enc = tokenizer
            .encode(s, true)
            .map_err(|e| anyhow::anyhow!("{}:{}: encode: {e}", args.jsonl.display(), lineno + 1))?;
        let ids = enc.get_ids();
        if ids.is_empty() {
            bail!("{}:{}: tokenized to zero tokens", args.jsonl.display(), lineno + 1);
        }
        let input = Tensor::new(ids, &device)?.unsqueeze(0)?;
        // One sequence at a time, unpadded — no attention mask needed, and it
        // keeps the mean pool honest (no pad tokens diluting it).
        let hidden = trunk.forward(&input, None)?;
        let cls = Lfm2Trunk::pool_cls(&hidden)?.squeeze(0)?.to_vec1::<f32>()?;
        let mean = hidden.mean(1)?.squeeze(0)?.to_vec1::<f32>()?;
        cls_rows.push(cls);
        mean_rows.push(mean);

        if (lineno + 1) % 100 == 0 {
            let per = started.elapsed().as_secs_f32() / (lineno + 1) as f32;
            eprintln!("  {} rows, {:.0} ms/row", lineno + 1, per * 1000.0);
        }
    }

    let n = labels.len();
    if n == 0 {
        bail!("{}: no rows", args.jsonl.display());
    }
    let hidden_size = cls_rows[0].len();

    let flat_cls: Vec<f32> = cls_rows.into_iter().flatten().collect();
    let flat_mean: Vec<f32> = mean_rows.into_iter().flatten().collect();

    let mut out: HashMap<String, Tensor> = HashMap::new();
    out.insert(
        "cls".into(),
        Tensor::from_vec(flat_cls, (n, hidden_size), &device)?,
    );
    out.insert(
        "mean".into(),
        Tensor::from_vec(flat_mean, (n, hidden_size), &device)?,
    );
    out.insert("labels".into(), Tensor::from_vec(labels, n, &device)?);
    candle_core::safetensors::save(&out, &args.out)?;

    eprintln!(
        "wrote {} — {n} rows x {hidden_size} dims, {:.1}s total",
        args.out.display(),
        started.elapsed().as_secs_f32()
    );
    Ok(())
}
