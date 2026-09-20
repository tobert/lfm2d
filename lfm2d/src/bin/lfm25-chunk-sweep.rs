//! Sweep the length of the final prefill chunk and see whether the model reads
//! its last position differently.
//!
//! See `lfm2d::chunk_sweep` for what is being asked and why the sweep carries
//! its own control. In short: the 09-17 ribbon's off-line rows were the rows
//! whose last prefill chunk was 1 to 8 tokens long, and the short convolution's
//! three-token window cannot reach that far.
//!
//! The report holds numbers, token COUNTS and hashes — never the prompt text,
//! which may be a corpus row. So it is safe on stdout; `--out` is for keeping
//! it beside a run's other artifacts.
//!
//! ```bash
//! lfm25-chunk-sweep --device rocm \
//!   --model '/tank/ml/models/llama.cpp/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q5_K_M.gguf' \
//!   --tokenizer '.models/LFM2.5-8B-A1B/tokenizer.json' \
//!   --prompt 'lfm2d/prompts/command-verdict-enum-v1.json' \
//!   --input 'docker system prune -af' \
//!   --assistant-prefill '{"effect": "removes unused images", "scope": "system", "undo": "hard", "verdict": "'
//! ```

use clap::Parser;
use lfm2d::adjudicator::{Checkpoint, PromptSpec, validate_text};
use lfm2d::chunk_sweep::sweep;
use lfm2d::hash::sha256_hex_bytes;
use std::path::{Path, PathBuf};

/// Tails short enough to change a GEMM's tiling or take the single-token
/// decode path, and — as the control — tails around a chunk boundary that move
/// the prefix by the same amounts while staying large.
const DEFAULT_TAILS: &[usize] = &[
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 24, 32, 64, 120, 121, 122, 123, 124, 128, 132, 136,
];

#[derive(Parser, Debug)]
#[command(about = "Does the final prefill chunk's length change what LFM2.5 reads?")]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    tokenizer: PathBuf,
    #[arg(long, value_enum, default_value = "auto")]
    device: lfm2d::device::DeviceArg,
    #[arg(long, default_value_t = 0)]
    device_index: usize,

    /// Exact text to read, control tokens and all.
    #[arg(long, conflicts_with_all = ["prompt", "input", "assistant_prefill"])]
    text_file: Option<PathBuf>,
    /// A prompt spec as the daemon's --adjudicator-prompt takes it.
    #[arg(long, requires = "input")]
    prompt: Option<PathBuf>,
    /// The user turn's content, rendered as the daemon renders it.
    #[arg(long, requires = "prompt")]
    input: Option<String>,
    /// Text the assistant has already written, to stand at an answer slot.
    #[arg(long, default_value = "")]
    assistant_prefill: String,

    /// A final-chunk length to measure. Repeatable; defaults to a sweep that
    /// covers the short tails and a large-tail control.
    #[arg(long = "tail")]
    tails: Vec<usize>,
    /// Write the report here as well as to stdout.
    #[arg(long)]
    out: Option<PathBuf>,
}

fn main() {
    if let Err(e) = run(Args::parse()) {
        eprintln!("lfm25-chunk-sweep: {e}");
        std::process::exit(1);
    }
}

fn run(args: Args) -> Result<(), String> {
    let text = |e: std::io::Error, p: &Path| format!("{}: {e}", p.display());
    let rendered = match (&args.text_file, &args.prompt, &args.input) {
        (Some(path), None, None) => std::fs::read_to_string(path).map_err(|e| text(e, path))?,
        (None, Some(path), Some(input)) => {
            let spec: PromptSpec =
                serde_json::from_slice(&std::fs::read(path).map_err(|e| text(e, path))?)
                    .map_err(|e| format!("{}: {e}", path.display()))?;
            // The daemon refuses this input, so its rendering of it does not
            // exist to be read. Exact text goes through --text-file.
            validate_text(input)?;
            format!(
                "{}{}",
                spec.render_prefix()?,
                spec.render_user_turn_with_prefill(input, &args.assistant_prefill)?
            )
        }
        _ => return Err("give --text-file, or --prompt with --input".into()),
    };

    let checkpoint = Checkpoint::load(&args.model, &args.tokenizer, args.device, args.device_index)?;
    let tokens = checkpoint
        .tokenizer
        .encode(rendered.as_str(), false)
        .map_err(|e| e.to_string())?
        .get_ids()
        .to_vec();
    if tokens.is_empty() {
        return Err("the prompt tokenizes to nothing".into());
    }
    let tails: Vec<usize> = if args.tails.is_empty() {
        DEFAULT_TAILS.to_vec()
    } else {
        args.tails.clone()
    };
    // A tail longer than the prompt has no reading to give. Refuse rather than
    // silently measuring a shorter one: the sweep's shape is the result.
    if let Some(&too_long) = tails.iter().find(|&&t| t > tokens.len() || t == 0) {
        return Err(format!(
            "tail {too_long} is outside 1..={} for this prompt",
            tokens.len()
        ));
    }

    // stderr, so stdout stays the report. A reading is minutes on CPU.
    let started = std::time::Instant::now();
    let swept = sweep(&checkpoint.model, &tokens, &tails, |n, total| {
        eprintln!("reading {n}/{total} after {:.0}s", started.elapsed().as_secs_f32());
    })?;
    let report = serde_json::json!({
        "schema": "lfm25-chunk-sweep-v1",
        "model_id": checkpoint.model_id,
        "weight_hash": checkpoint.weight_hash,
        "tokenizer_hash": checkpoint.tokenizer_hash,
        "backend": checkpoint.execution.backend.as_str(),
        "n_tokens": tokens.len(),
        "prompt_sha256": sha256_hex_bytes(rendered.as_bytes()),
        "reference": "one block of every token",
        "tails": tails,
        "reference_top": swept.reference_top,
        "reference_margin": swept.reference_margin,
        "deltas": swept.deltas,
    });
    let json = serde_json::to_string(&report).map_err(|e| e.to_string())?;
    if let Some(path) = &args.out {
        std::fs::write(path, &json).map_err(|e| text(e, path))?;
    }
    println!("{json}");
    Ok(())
}
