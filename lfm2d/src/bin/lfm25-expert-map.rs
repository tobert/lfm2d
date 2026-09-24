//! Build, or load from cache, the causal expert map for one deployment.
//!
//! "Layer 16, expert 25" is an index into one set of trained weights, found under
//! one prompt, at one answer slot, on one backend. So the map is a DERIVED
//! artifact, and its cache key is everything it depends on: weights, tokenizer,
//! the rendered prompt spec, the backend, the probes and the followed tokens.
//! Change any of them and the key changes, so a stale map is never loaded; it is
//! simply not found, and built again.
//!
//! Probes are JSON lines of {"name", "input", "assistant_prefill"?, "group"?},
//! rendered exactly as the daemon renders a request, then continued by the
//! prefill so the last token stands at the answer slot.
//!
//! The map holds no prompt text, only probe names, so it may live anywhere; the
//! probes file is the caller's and stays where the caller keeps it.

use clap::Parser;
use lfm2d::adjudicator::{Checkpoint, PromptSpec, validate_text};
use lfm2d::expert_map::{ExpertMap, Probe, SCHEMA, expert_map};
use lfm2d::hash::{sha256_hex_bytes, sha256_hex_file};
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(about = "Which experts carry the answer: a knockout map, cached by what it depends on")]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    tokenizer: PathBuf,
    #[arg(long)]
    prompt: PathBuf,
    #[arg(long)]
    probes: PathBuf,
    /// Tokens to follow at the answer slot, as words: the FIRST token of each,
    /// encoded alone. Comma separated.
    #[arg(long, value_delimiter = ',', required = true)]
    follow: Vec<String>,
    /// Where maps are kept, one file per cache key.
    #[arg(long)]
    cache_dir: PathBuf,
    /// Build even if a map with this key exists; the old file is replaced only
    /// once the new one is complete.
    #[arg(long)]
    recompute: bool,
    #[arg(long, value_enum, default_value = "auto")]
    device: lfm2d::device::DeviceArg,
    #[arg(long, default_value_t = 0)]
    device_index: usize,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ProbeLine {
    name: String,
    input: String,
    #[serde(default)]
    assistant_prefill: String,
    #[serde(default)]
    group: Option<String>,
}

/// Everything the map depends on, in a fixed order. `weights` is the digest of
/// the whole GGUF, so a requantisation is a different key.
fn cache_key(parts: &[(&str, &str)]) -> String {
    // JSON, not joined lines: a value holding a separator cannot pass for a field.
    sha256_hex_bytes(&serde_json::to_vec(parts).expect("strings always serialise"))
}

fn run(args: Args) -> Result<(), String> {
    let io = |e: std::io::Error, p: &Path| format!("{}: {e}", p.display());
    let spec: PromptSpec = serde_json::from_slice(&std::fs::read(&args.prompt).map_err(|e| io(e, &args.prompt))?)
        .map_err(|e| format!("{}: {e}", args.prompt.display()))?;
    let prefix = spec.render_prefix()?;

    // The key needs no model on the device: digests and the rendered prefix.
    let weights = sha256_hex_file(&args.model).map_err(|e| io(e, &args.model))?;
    let tokenizer_hash = sha256_hex_file(&args.tokenizer).map_err(|e| io(e, &args.tokenizer))?;
    let probes_hash = sha256_hex_file(&args.probes).map_err(|e| io(e, &args.probes))?;
    let execution = lfm2d::device::ExecutionDevice::select(args.device, args.device_index)?;
    let (backend, device) = (execution.backend, execution.identity);
    let key = cache_key(&[
        ("schema", SCHEMA),
        ("weights", &weights),
        ("tokenizer", &tokenizer_hash),
        ("prefix", &sha256_hex_bytes(prefix.as_bytes())),
        // The reasoning opening lives in the USER turn, not the prefix, so a
        // spec that flips `reasoning` renders every probe differently while
        // `prefix` is byte-identical. Without this the cache would serve a map
        // built from the other mode.
        ("template", spec.template_version()),
        ("backend", backend.as_str()),
        // Kernels branch on the GPU target and change with the candle build,
        // so a map from another card or fork revision is a different map.
        ("device", &device),
        ("candle", lfm2d::adjudicator::CANDLE_REV),
        ("probes", &probes_hash),
        ("follow", &args.follow.join(",")),
    ]);
    std::fs::create_dir_all(&args.cache_dir).map_err(|e| io(e, &args.cache_dir))?;
    let target = args.cache_dir.join(format!("expert-map-{}.json", &key[..16]));
    if target.exists() && !args.recompute {
        let cached: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&target).map_err(|e| io(e, &target))?)
                .map_err(|e| format!("{}: {e}", target.display()))?;
        if cached["identity"]["key"] != key {
            return Err(format!("{} does not carry the key its name was cut from", target.display()));
        }
        println!("cached {}", target.display());
        return Ok(());
    }

    let checkpoint = Checkpoint::load(&args.model, &args.tokenizer, args.device, args.device_index)?;
    let tokenizer = &checkpoint.tokenizer;
    let encode = |text: &str| -> Result<Vec<u32>, String> {
        Ok(tokenizer.encode(text, false).map_err(|e| e.to_string())?.get_ids().to_vec())
    };
    let mut followed = Vec::new();
    for word in &args.follow {
        followed.push(*encode(word)?.first().ok_or_else(|| format!("{word:?} tokenizes to nothing"))?);
    }
    let mut probes = Vec::new();
    let lines = std::fs::read_to_string(&args.probes).map_err(|e| io(e, &args.probes))?;
    for (n, line) in lines.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
        let at = |e: String| format!("{} line {}: {e}", args.probes.display(), n + 1);
        let row: ProbeLine = serde_json::from_str(line).map_err(|e| at(e.to_string()))?;
        validate_text(&row.input).map_err(at)?;
        let turn = spec
            .render_user_turn_with_prefill(&row.input, &row.assistant_prefill)
            .map_err(at)?;
        let text = format!("{prefix}{turn}");
        probes.push(Probe { name: row.name, tokens: encode(&text).map_err(at)?, group: row.group });
    }

    let begin = std::time::Instant::now();
    let mut done = 0usize;
    let total = probes.len();
    let map: ExpertMap = expert_map(&checkpoint.model, &probes, &followed, |_| {
        done += 1;
        if done.is_multiple_of(10) || done == total {
            eprintln!("  {done}/{total} probes, {:.1}s", begin.elapsed().as_secs_f64());
        }
    })?;
    let seconds = begin.elapsed().as_secs_f64();
    let record = serde_json::json!({
        "identity": {
            "key": key,
            "model_id": checkpoint.model_id,
            "weight_hash": weights,
            "tokenizer_hash": tokenizer_hash,
            "prompt": args.prompt.file_name().and_then(|n| n.to_str()),
            "prefix_sha256": sha256_hex_bytes(prefix.as_bytes()),
            "template_version": spec.template_version(),
            "backend": backend.as_str(),
            "device": device,
            "candle_rev": lfm2d::adjudicator::CANDLE_REV,
            "probes_sha256": probes_hash,
        },
        "follow": args.follow.iter().zip(&followed).map(|(word, &id)| serde_json::json!({
            "word": word, "id": id, "piece": tokenizer.id_to_token(id),
        })).collect::<Vec<_>>(),
        "seconds": seconds,
        "map": map,
    });
    // Complete file or no file: a reader never sees half a map under a valid key.
    let partial = target.with_extension("json.partial");
    std::fs::write(&partial, serde_json::to_vec(&record).map_err(|e| e.to_string())?).map_err(|e| io(e, &partial))?;
    std::fs::rename(&partial, &target).map_err(|e| io(e, &target))?;
    println!(
        "built {} probes, {} knockouts, {seconds:.1}s ({:.2}s per probe) -> {}",
        total,
        map.probes.iter().map(|p| p.knockouts.len()).sum::<usize>(),
        seconds / total as f64,
        target.display()
    );
    Ok(())
}

fn main() {
    if let Err(e) = run(Args::parse()) {
        eprintln!("lfm25-expert-map: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_key_moves_when_anything_it_names_moves_and_only_then() {
        let base = [
            ("schema", "s"),
            ("weights", "w"),
            ("backend", "rocm"),
            ("device", "rocm:gfx1151:hip7.2"),
            ("candle", "rev"),
        ];
        assert_eq!(cache_key(&base), cache_key(&base));
        for i in 0..base.len() {
            let mut changed = base;
            changed[i].1 = "different";
            assert_ne!(cache_key(&base), cache_key(&changed), "{}", base[i].0);
        }
        // A value cannot slide into its neighbour's field.
        assert_ne!(cache_key(&[("a", "x\nb=y")]), cache_key(&[("a", "x"), ("b", "y")]) );
    }
}
