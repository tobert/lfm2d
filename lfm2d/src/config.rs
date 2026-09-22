//! CLI configuration — flags with env-var fallbacks (`clap`'s `env`
//! feature). Startup fails loudly on anything incoherent; see
//! [`Cli::validate`].

use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use lfm2_encoder::DType;

/// Compute dtype, as spelled on the command line. A closed enum rather than
/// a free string so `clap` rejects a typo at parse time with the valid set
/// named, instead of the daemon discovering it mid-load.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum DtypeArg {
    F32,
    F16,
    Bf16,
}

impl DtypeArg {
    pub fn to_dtype(self) -> DType {
        match self {
            DtypeArg::F32 => DType::F32,
            DtypeArg::F16 => DType::F16,
            DtypeArg::Bf16 => DType::BF16,
        }
    }
}

/// `lfm2d` — HTTP sidecar serving LFM2.5 encoder heads. Loads every
/// configured checkpoint once at startup (no lazy loading); serves the same
/// API over a Unix socket and/or TCP.
#[derive(Parser, Debug, Clone)]
#[command(name = "lfm2d", about, long_about = None)]
pub struct Cli {
    /// LFM2.5-8B-A1B GGUF for the separate adjudicator worker.
    #[arg(long, env="LFM2D_ADJUDICATOR_MODEL")]
    pub adjudicator_model: Option<PathBuf>,
    /// Matching Hugging Face tokenizer.json (checked against GGUF vocabulary).
    #[arg(long, env="LFM2D_ADJUDICATOR_TOKENIZER")]
    pub adjudicator_tokenizer: Option<PathBuf>,
    /// JSON {system, output_schema} prompt, prefetched into an immutable hybrid snapshot.
    #[arg(long, env="LFM2D_ADJUDICATOR_PROMPT")]
    pub adjudicator_prompt: Option<PathBuf>,
    /// Adjudicator context budget, including output. The initial eager attention path is capped at 8192.
    #[arg(long, default_value_t=4096)]
    pub adjudicator_context: usize,
    /// Sign-aware repetition penalty over the full prompt and generated tokens.
    /// 1.0 disables it; 1.05 is the checkpoint author's recommendation.
    #[arg(long, default_value_t = 1.05)]
    pub adjudicator_repeat_penalty: f32,
    /// Further prompt specs served by `/v1/opinion` beside the adjudicator
    /// prompt (which is always on the menu). Repeatable; each gets its own
    /// resident prefix. Named by file stem, so stems must not repeat.
    #[arg(long = "opinion-spec", env = "LFM2D_OPINION_SPECS", value_delimiter = ',')]
    pub opinion_specs: Vec<PathBuf>,

    /// Directory holding an `Lfm2Embedding`-shaped checkpoint
    /// (`config.json`, `tokenizer.json`, `model.safetensors`). Backs
    /// `/embed`.
    #[arg(long, env = "LFM2D_EMBEDDER_DIR")]
    pub embedder_dir: Option<PathBuf>,

    /// Directory holding an `Lfm2SequenceClassifier`-shaped checkpoint.
    /// Backs `/predict`, `/v1/classify`, and (with `--router-dir`)
    /// `/v1/cascade`.
    #[arg(long, env = "LFM2D_CLASSIFIER_DIR")]
    pub classifier_dir: Option<PathBuf>,

    /// Directory holding an `Lfm2SequenceRouter`-shaped checkpoint (the
    /// Prompt-Router). Backs `/v1/route` and (with `--classifier-dir`)
    /// `/v1/cascade`.
    #[arg(long, env = "LFM2D_ROUTER_DIR")]
    pub router_dir: Option<PathBuf>,

    /// Directory holding a SECOND `Lfm2SequenceClassifier`-shaped
    /// checkpoint, scored SHADOW-only alongside `--classifier-dir` on every
    /// `/v1/classify` and `/v1/cascade` call. Purely observational: never
    /// listed in `/v1/models`, never influences a response body, byte, or
    /// status code — see `engine_real.rs`'s `classify`/`cascade` for where
    /// the shadow score is computed and immediately discarded after being
    /// recorded (agreement counter only, never the raw verdict pair or
    /// input text as a label — unbounded per-input cardinality would wreck
    /// VictoriaMetrics, same reasoning as `--log-input-hash` above).
    /// Optional; omitting it is a no-op, zero behavior or latency change
    /// (2026-08-15, evaluating `kube_ordinal_v9` candidates against live
    /// traffic without touching what `kube_ordinal_v8` serves).
    #[arg(long, env = "LFM2D_CANDIDATE_CLASSIFIER_DIR")]
    pub candidate_classifier_dir: Option<PathBuf>,

    /// Directory holding an `Lfm2TokenClassifier`-shaped checkpoint —
    /// REPEATABLE (pass `--token-classifier-dir` once per head, or a
    /// comma-separated list via `LFM2D_TOKEN_CLASSIFIER_DIR`), unlike
    /// `--embedder-dir`/`--classifier-dir`/`--router-dir` which each take
    /// at most one. Each registers under a model id derived from its
    /// directory basename, exactly like every other head — fully generic,
    /// no checkpoint-specific logic anywhere (the PII detector is not
    /// special-cased). Backs `POST /v1/spans` and
    /// `POST /v1/spans/credentials`. With 2+ loaded, a request must name
    /// which one via `"model"`; with exactly 1, it's implicit — see
    /// `engine_real.rs`'s `resolve_token_classifier`.
    #[arg(long = "token-classifier-dir", env = "LFM2D_TOKEN_CLASSIFIER_DIR", value_delimiter = ',')]
    pub token_classifier_dir: Vec<PathBuf>,

    /// Attach a hash of a `/v1/spans`/`/v1/spans/credentials` request's
    /// input TEXT (never the text itself) to that call's trace/log span,
    /// as an opt-in correlation aid — e.g. matching one detection call to
    /// the same call logged by an upstream service, without either system
    /// writing the caller's actual secret-bearing text anywhere. Defaults
    /// OFF and is deliberately `ArgAction::Set` (must be spelled
    /// `--log-input-hash true`, not just present) rather than a bare
    /// switch — flipping on ANY per-request hashing on a secrets-detection
    /// endpoint should be a considered operator choice, not a fat-fingered
    /// flag. NEVER attached as an OTLP metric label regardless of this
    /// setting — see `worker.rs`'s `spans`/`spans_credentials` and the
    /// module docs' "Observability" section for why (unbounded per-input
    /// cardinality would wreck VictoriaMetrics).
    #[arg(
        long = "log-input-hash",
        env = "LFM2D_LOG_INPUT_HASH",
        action = clap::ArgAction::Set,
        default_value_t = false
    )]
    pub log_input_hash: bool,

    /// A cascade candidate route (repeatable), or a comma-separated list
    /// via `LFM2D_CASCADE_ROUTES`. `/v1/cascade` needs at least one; the
    /// request body carries only `clauses` (see the crate root docs on why
    /// routes are server-side config, not a per-request field).
    #[arg(long = "cascade-route", env = "LFM2D_CASCADE_ROUTES", value_delimiter = ',')]
    pub cascade_routes: Vec<String>,

    /// Which of the classifier's own labels count toward `/v1/cascade`'s
    /// severity ranking — repeatable, or comma-separated via
    /// `LFM2D_CASCADE_SEVERE_LABELS`. Defaults to `kube_ordinal_v6`'s
    /// convention (`examples/cascade.rs`'s `DEFAULT_SEVERE`).
    ///
    /// **ORDER IS SEMANTIC: ascending severity, LEAST severe first.** The
    /// position of each label in this list is its ordinal rank, so
    /// `--cascade-severe-label situation-normal --cascade-severe-label
    /// data-critical` is correct and the reverse silently inverts every
    /// ranking `/v1/cascade` produces — both names are valid, so nothing
    /// errors and the endpoint just starts naming the least severe clause
    /// as the winner. Rank cannot be read off the checkpoint (v6 stores its
    /// labels most-severe-first), so only this flag carries it. The
    /// resolved ranking is echoed at startup as
    /// `cascade severity ranking (ascending, least severe first)` — read
    /// that line once after any change to this flag; it is the only place
    /// the mistake is visible. Duplicates are refused (ambiguous rank).
    #[arg(
        long = "cascade-severe-label",
        env = "LFM2D_CASCADE_SEVERE_LABELS",
        value_delimiter = ',',
        default_value = "mutating,destructive"
    )]
    pub cascade_severe_labels: Vec<String>,

    /// Unix domain socket path to serve on. At least one of this or
    /// `--bind-addr` is required.
    #[arg(long, env = "LFM2D_SOCKET_PATH")]
    pub socket_path: Option<PathBuf>,

    /// TCP address to serve on, e.g. `127.0.0.1:8080`. At least one of this
    /// or `--socket-path` is required.
    #[arg(long, env = "LFM2D_BIND_ADDR")]
    pub bind_addr: Option<String>,

    /// Compute dtype every head is loaded and run at: `f32` (default),
    /// `f16`, or `bf16`.
    ///
    /// # Read this before reaching for f16
    ///
    /// **The LFM2.5 encoder checkpoints ship f32 natively** — verified
    /// 2026-08-12 from the safetensors headers: Encoder-350M,
    /// Prompt-Router and PII-Detector are all `F32`, ~1352 MiB each, with
    /// `torch_dtype: float32` in their configs. `kube_ordinal_v8` is f32
    /// too. So f16 is a genuine *loss* of shipped resolution here, not the
    /// removal of a pointless upcast.
    ///
    /// **The lone exception is `LFM2.5-Embedding-350M`, which ships
    /// `BF16`** (676 MiB). bf16→f32 is lossless; bf16→**f16 is not**, and
    /// not merely in mantissa — bf16 carries f32's 8 exponent bits against
    /// f16's 5, so values outside f16's range become inf/0 rather than
    /// rounding. Loading that checkpoint as f16 is the one combination here
    /// that can quietly produce wrong numbers instead of slow ones.
    ///
    /// **And f16 trades the constraint that binds for the one that
    /// doesn't.** Measured on this workload, f16 halves memory at ~1.6×
    /// latency, while the cost that actually hurts is a forward pass in an
    /// interactive path. Reach for this when a box is memory-bound (zorak's
    /// `system-reserved` arithmetic), not to make things faster.
    #[arg(long, env = "LFM2D_DTYPE", default_value = "f32")]
    pub dtype: DtypeArg,

    /// Execution backend. Auto tries compiled GPU backends (ROCm, CUDA,
    /// Metal), then CPU if device initialization is unavailable. Explicit
    /// backends fail instead of falling back. Model-load and inference errors
    /// are never retried on another device.
    #[arg(long, env = "LFM2D_DEVICE", default_value = "auto")]
    pub device: crate::device::DeviceArg,

    /// GPU ordinal within the selected backend.
    #[arg(long, env = "LFM2D_DEVICE_INDEX", default_value_t = 0)]
    pub device_index: usize,

    /// Size of the rayon global thread pool that candle's matmul runs on
    /// (`rayon::ThreadPoolBuilder::num_threads`), set BEFORE any model is
    /// loaded — rayon's global pool can only be built once, so this must
    /// win the race against candle's own lazy default. Defaults to
    /// `std::thread::available_parallelism()` when unset, which under a
    /// k8s CPU *limit* (a cgroup quota, not just a request) usually
    /// over-reports — see `lfm2d/deploy/k8s.yaml`'s CPU-limits comment for
    /// why this daemon's deploy guidance is "requests without limits."
    #[arg(long, env = "LFM2D_THREADS")]
    pub threads: Option<usize>,
}

impl Cli {
    /// Cross-field checks `clap` itself can't express. Pure and
    /// unit-testable without ever invoking the CLI parser — see the tests
    /// below.
    ///
    /// # Errors
    /// - neither `--socket-path` nor `--bind-addr` given: nothing to serve
    ///   on.
    /// - none of `--embedder-dir`/`--classifier-dir`/`--router-dir`/
    ///   `--token-classifier-dir` given: nothing to load, so this process
    ///   would serve `/healthz` and nothing else — almost certainly a
    ///   misconfiguration, not a deliberate deployment.
    pub fn validate(&self) -> Result<(), String> {
        if self.socket_path.is_none() && self.bind_addr.is_none() {
            return Err(
                "no transport configured: pass --socket-path and/or --bind-addr \
                 (env LFM2D_SOCKET_PATH / LFM2D_BIND_ADDR) — at least one is required"
                    .to_string(),
            );
        }
        if self.embedder_dir.is_none()
            && self.classifier_dir.is_none()
            && self.router_dir.is_none()
            && self.token_classifier_dir.is_empty()
            && self.adjudicator_model.is_none()
        {
            return Err(
                "no models configured: pass at least one of --embedder-dir/--classifier-dir/\
                 --router-dir/--token-classifier-dir/--adjudicator-model (env LFM2D_EMBEDDER_DIR/LFM2D_CLASSIFIER_DIR/\
                 LFM2D_ROUTER_DIR/LFM2D_TOKEN_CLASSIFIER_DIR)"
                    .to_string(),
            );
        }
        if !self.adjudicator_repeat_penalty.is_finite() || !(1.0..=2.0).contains(&self.adjudicator_repeat_penalty) {
            return Err("adjudicator repeat penalty must be finite and in 1.0..=2.0".into());
        }
        let adjudicator_fields = [self.adjudicator_model.is_some(), self.adjudicator_tokenizer.is_some(), self.adjudicator_prompt.is_some()];
        if adjudicator_fields.iter().any(|&v|v) && !adjudicator_fields.iter().all(|&v|v) {
            return Err("--adjudicator-model, --adjudicator-tokenizer and --adjudicator-prompt must be supplied together".into());
        }
        if self.adjudicator_model.is_some() && (!matches!(self.dtype, DtypeArg::F32) || !(128..=8192).contains(&self.adjudicator_context)) {
            return Err("adjudicator requires --dtype f32 and --adjudicator-context 128..=8192".into());
        }
        if !self.opinion_specs.is_empty() && self.adjudicator_model.is_none() {
            return Err("--opinion-spec needs the adjudicator (--adjudicator-model/--adjudicator-tokenizer/--adjudicator-prompt)".into());
        }
        Ok(())
    }
}

/// The resolved severity ranking, ascending — `(rank, label)` pairs with
/// `rank` starting at 1, exactly the ordinal weights
/// [`lfm2_encoder::severity_rank_weights`] assigns.
///
/// # Why this exists as a separate, printable thing
///
/// `--cascade-severe-label`'s ORDER is the ordinal severity scale, and
/// reversing it silently inverts `/v1/cascade`'s ranking: both label names
/// are valid, so nothing errors, and the endpoint just starts answering
/// with the least severe clause as the winner. It cannot be detected from
/// the names alone — only the operator knows which of their own labels is
/// worse.
///
/// Echoing the raw flag back (which the startup config line already did)
/// does NOT make that mistake visible: a reversed flag prints as a reversed
/// flag and reads like an ordinary list. Rendering it as an explicit
/// ranking does — `1=data-critical 2=situation-normal` is *wrong on sight*
/// in a way that `["data-critical", "situation-normal"]` is not.
///
/// This is the "callers SHOULD echo the resolved ranking at startup"
/// mitigation that [`lfm2_encoder::severity_rank_weights`]'s docs ask for
/// by name, and it names `lfm2d` specifically as the caller that needs it:
/// a long-lived daemon loads a checkpoint once, so a one-shot CLI's habit
/// of showing its own arguments doesn't apply.
///
/// Derived from `severity_rank_weights` rather than re-walking the flag,
/// so the printed ranking cannot drift from the one the cascade actually
/// scores with — including its duplicate-label rejection.
///
/// # Errors
/// Whatever [`lfm2_encoder::severity_rank_weights`] rejects: an empty
/// severe set, a label the classifier doesn't have, or a label listed
/// twice (its ordinal rank would be ambiguous).
pub fn resolved_severity_ranking(
    labels: &[String],
    severe_labels: &[String],
) -> Result<Vec<(u32, String)>, String> {
    let weights = lfm2_encoder::severity_rank_weights(labels, severe_labels)
        .map_err(|e| format!("--cascade-severe-label: {e}"))?;

    let mut ranked: Vec<(u32, String)> = weights
        .iter()
        .enumerate()
        .filter(|&(_, &w)| w > 0.0)
        .map(|(id, &w)| (w as u32, labels[id].clone()))
        .collect();
    ranked.sort_by_key(|(rank, _)| *rank);
    Ok(ranked)
}

/// One-line rendering for the startup log: `1=situation-normal
/// 2=data-critical`. Ascending, least severe first, so it reads in the same
/// direction as the flag the operator typed.
pub fn render_severity_ranking(ranked: &[(u32, String)]) -> String {
    ranked
        .iter()
        .map(|(rank, label)| format!("{rank}={label}"))
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> Cli {
        Cli {
            adjudicator_model: None,
            adjudicator_tokenizer: None,
            adjudicator_prompt: None,
            adjudicator_context: 4096,
            adjudicator_repeat_penalty: 1.05,
            opinion_specs: Vec::new(),
            embedder_dir: None,
            classifier_dir: None,
            router_dir: None,
            candidate_classifier_dir: None,
            token_classifier_dir: Vec::new(),
            log_input_hash: false,
            cascade_routes: Vec::new(),
            cascade_severe_labels: vec!["mutating".into(), "destructive".into()],
            socket_path: None,
            bind_addr: None,
            dtype: DtypeArg::F32,
            device: crate::device::DeviceArg::Cpu,
            device_index: 0,
            threads: None,
        }
    }

    #[test]
    fn rejects_no_transport_configured() {
        let mut cli = base();
        cli.embedder_dir = Some("/tmp/x".into());
        let err = cli.validate().expect_err("no socket_path/bind_addr must be refused");
        assert!(err.contains("transport"), "{err}");
    }

    #[test]
    fn rejects_no_models_configured() {
        let mut cli = base();
        cli.bind_addr = Some("127.0.0.1:0".into());
        let err = cli.validate().expect_err("no models configured must be refused");
        assert!(err.contains("models"), "{err}");
    }

    #[test]
    fn accepts_one_model_and_socket_path_only() {
        let mut cli = base();
        cli.router_dir = Some("/tmp/router".into());
        cli.socket_path = Some("/tmp/lfm2d.sock".into());
        cli.validate().expect("one model + one transport is valid");
    }

    #[test]
    fn accepts_both_transports_and_all_three_models() {
        let cli = Cli {
            adjudicator_model: None,
            adjudicator_tokenizer: None,
            adjudicator_prompt: None,
            adjudicator_context: 4096,
            adjudicator_repeat_penalty: 1.05,
            opinion_specs: Vec::new(),
            embedder_dir: Some("/tmp/e".into()),
            classifier_dir: Some("/tmp/c".into()),
            router_dir: Some("/tmp/r".into()),
            candidate_classifier_dir: None,
            token_classifier_dir: vec!["/tmp/t".into()],
            log_input_hash: true,
            cascade_routes: vec!["shell".into()],
            cascade_severe_labels: vec!["mutating".into()],
            socket_path: Some("/tmp/lfm2d.sock".into()),
            bind_addr: Some("0.0.0.0:8080".into()),
            dtype: DtypeArg::F32,
            device: crate::device::DeviceArg::Cpu,
            device_index: 0,
            threads: Some(4),
        };
        cli.validate().expect("fully specified config is valid");
    }

    #[test]
    fn a_token_classifier_dir_alone_satisfies_the_models_requirement() {
        let mut cli = base();
        cli.token_classifier_dir = vec!["/tmp/pii".into()];
        cli.socket_path = Some("/tmp/lfm2d.sock".into());
        cli.validate().expect("--token-classifier-dir alone is a valid model set");
    }

    // ---------------------------------------------------- severity ranking
    //
    // These pin the ONE thing that makes the startup echo worth having: a
    // reversed `--cascade-severe-label` must render differently. If these
    // ever pass with an order-insensitive rendering, the log line has
    // stopped being able to catch the bug it exists for.

    /// v8's vocabulary, in an order that is NOT the severity order — real
    /// checkpoints store labels in training order, so anything that reads
    /// severity off array position is wrong.
    fn v8_labels() -> Vec<String> {
        vec!["informative".into(), "situation-normal".into(), "data-critical".into()]
    }

    /// v6 stored its labels alphabetically-ish, with the MOST severe first —
    /// the case that catches "just use the checkpoint's own order."
    fn v6_labels() -> Vec<String> {
        vec!["destructive".into(), "informative".into(), "mutating".into()]
    }

    #[test]
    fn ranking_follows_the_flag_order_not_the_checkpoints_label_order() {
        let severe = vec!["mutating".to_string(), "destructive".to_string()];
        let ranked = resolved_severity_ranking(&v6_labels(), &severe).expect("both labels present");
        assert_eq!(
            ranked,
            vec![(1, "mutating".to_string()), (2, "destructive".to_string())],
            "rank must come from the CLI order; v6 stores 'destructive' at index 0"
        );
    }

    #[test]
    fn reversing_the_flag_visibly_reverses_the_rendered_ranking() {
        let correct = vec!["situation-normal".to_string(), "data-critical".to_string()];
        let reversed = vec!["data-critical".to_string(), "situation-normal".to_string()];

        let a = render_severity_ranking(
            &resolved_severity_ranking(&v8_labels(), &correct).expect("valid"),
        );
        let b = render_severity_ranking(
            &resolved_severity_ranking(&v8_labels(), &reversed).expect("valid"),
        );

        assert_eq!(a, "1=situation-normal 2=data-critical");
        assert_eq!(b, "1=data-critical 2=situation-normal");
        assert_ne!(
            a, b,
            "a reversed severe-label flag MUST render differently — an order-insensitive \
             rendering (a set, a sorted list) would make this log line useless for the \
             silent-inversion bug it exists to expose"
        );
    }

    #[test]
    fn rendering_puts_the_least_severe_rung_first() {
        let severe = vec!["situation-normal".to_string(), "data-critical".to_string()];
        let rendered =
            render_severity_ranking(&resolved_severity_ranking(&v8_labels(), &severe).expect("valid"));
        let one = rendered.find("1=").expect("rank 1 present");
        let two = rendered.find("2=").expect("rank 2 present");
        assert!(one < two, "ascending, least severe first: {rendered}");
    }

    #[test]
    fn ranking_rejects_a_label_the_classifier_does_not_have() {
        let severe = vec!["situation-normal".to_string(), "catastrophic".to_string()];
        let err = resolved_severity_ranking(&v8_labels(), &severe)
            .expect_err("an unknown severe label must be refused, not silently dropped");
        assert!(err.contains("catastrophic"), "the error must name the bad label: {err}");
    }

    #[test]
    fn ranking_rejects_a_duplicated_label() {
        let severe = vec!["data-critical".to_string(), "data-critical".to_string()];
        let err = resolved_severity_ranking(&v8_labels(), &severe)
            .expect_err("a repeated label has no unambiguous rank");
        assert!(err.contains("data-critical"), "{err}");
    }

    #[test]
    fn ranking_rejects_an_empty_severe_set() {
        let err = resolved_severity_ranking(&v8_labels(), &[])
            .expect_err("an empty severe set ranks nothing");
        assert!(!err.is_empty());
    }

    #[test]
    fn ranking_covers_only_the_severe_labels() {
        let severe = vec!["situation-normal".to_string(), "data-critical".to_string()];
        let ranked = resolved_severity_ranking(&v8_labels(), &severe).expect("valid");
        assert_eq!(ranked.len(), 2, "'informative' is not severe and must not be ranked");
        assert!(ranked.iter().all(|(_, l)| l != "informative"));
    }

    #[test]
    fn token_classifier_dir_is_repeatable() {
        let cli = Cli {
            token_classifier_dir: vec!["/tmp/pii".into(), "/tmp/secrets".into()],
            socket_path: Some("/tmp/lfm2d.sock".into()),
            ..base()
        };
        assert_eq!(cli.token_classifier_dir.len(), 2);
        cli.validate().expect("2+ token-classifier dirs is valid config");
    }

    #[test]
    fn log_input_hash_defaults_off() {
        assert!(!base().log_input_hash, "a secrets-detection endpoint must not hash input by default");
    }

    // -------------------------------------------- real clap::Parser parsing
    //
    // Everything above builds a `Cli` by hand and never touches the
    // `#[arg(...)]` attributes at all — a typo in `value_delimiter` or
    // `action` would compile fine and pass every test above while silently
    // breaking real CLI parsing. These two exercise `Cli::parse_from`
    // directly, the same entry point `main.rs` uses via `Cli::parse()`.

    #[test]
    fn token_classifier_dir_parses_repeated_flags_and_comma_lists() {
        let cli = Cli::parse_from([
            "lfm2d",
            "--token-classifier-dir",
            "/models/pii",
            "--token-classifier-dir",
            "/models/secrets,/models/router-guard",
        ]);
        assert_eq!(
            cli.token_classifier_dir,
            vec![
                PathBuf::from("/models/pii"),
                PathBuf::from("/models/secrets"),
                PathBuf::from("/models/router-guard"),
            ]
        );
    }

    #[test]
    fn log_input_hash_requires_an_explicit_value_not_a_bare_flag() {
        let cli = Cli::parse_from(["lfm2d", "--log-input-hash", "true"]);
        assert!(cli.log_input_hash);
        let cli = Cli::parse_from(["lfm2d"]);
        assert!(!cli.log_input_hash, "must default to false when unset");
    }
}
