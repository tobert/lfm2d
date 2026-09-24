//! LFM2.5 bidirectional encoders on [candle] — pure Rust, CPU-first.
//!
//! LiquidAI's LFM2.5 encoder branch is a family of small (230M/350M)
//! hybrid conv+attention models with task heads: text embedding, masked-LM
//! fine-tune bases, token classification (PII/secrets detection, policy
//! linting), and prompt routing. Upstream candle-transformers implements
//! the *causal* LFM2 (`lfm2.rs`); nothing implements the bidirectional
//! encoder branch. This crate is that gap.
//!
//! ## Milestones
//!
//! 1. **Config** (done): parse every family checkpoint's `config.json`,
//!    verified against real fixtures; head detection via [`config::EncoderArch`].
//! 2. **Trunk** (done): the bidirectional hybrid stack ([`Lfm2Trunk`]) —
//!    blocks adapted from candle-transformers' `lfm2.rs` (MIT/Apache-2.0,
//!    attribution retained), minus the causal mask and KV cache, plus the
//!    encoder branch's *centered* short conv. Verified to max|Δ| ≈ 5e-5
//!    against the checkpoint's own Python modeling code.
//! 3. **Heads**, in order of use: pooled embedding (`Embedding-350M`),
//!    token classification (`PII-Detector`, `Policy-Linter`), sequence
//!    classification, sequence routing (`Prompt-Router` —
//!    [`routing::Lfm2SequenceRouter`], rule-projection scoring against
//!    caller-supplied routes; not a fixed-label softmax).
//! 4. **Quantized loads** (GGUF) if CPU f16/f32 latency disappoints.
//!
//! ## Padding is not inert
//!
//! The short conv is deliberately unmasked, matching how the checkpoints
//! were trained, so pad states bleed one position per conv layer into
//! their real neighbours. **A sequence's embedding depends on what it was
//! batched with.** Embed one sequence at a time, unpadded, whenever you
//! need reproducible vectors (cache keys, stored embeddings, cross-process
//! comparisons); batch when throughput matters more. See
//! [`Lfm2Trunk::forward`].
//!
//! Design intents: CPU-first (consumers embed this in long-lived server
//! processes — kaijutsu's kernel, kaibo — where a 350M encoder pass is
//! tens of milliseconds); no C++ in the dependency tree; weights loaded
//! from a local dir by default (`hub` feature adds Hub download).
//!
//! [candle]: https://github.com/huggingface/candle

pub mod config;
pub mod colbert;
pub mod embedding;
pub mod error;
mod labels;
pub mod routing;
pub mod sequence_classification;
pub mod token_classification;
pub mod trunk;

// Every `from_dir_with` in this crate takes a `DType` and a `&Device`, so a
// caller that wants anything other than the f32/CPU default cannot use the
// public API without naming candle's types. Re-exported here so they don't
// have to add candle-core as a direct dependency and keep its version
// pinned in lockstep with ours — the version coupling is the real hazard,
// since a mismatched candle would produce two incompatible `DType` types
// with identical names and a baffling error message.
pub use candle_core::{DType, Device};

pub use config::{EncoderArch, LayerType, Lfm2EncoderConfig};
pub use colbert::{ColbertModel, MultiVector};
pub use embedding::{cosine_similarity, Lfm2Embedding, TextKind};
pub use error::{Error, Result};
pub use routing::{ClauseRouting, Lfm2SequenceRouter, RouteComputation, RouteMatch};
pub use sequence_classification::Lfm2SequenceClassifier;
pub use token_classification::{
    is_secret_label, Lfm2TokenClassifier, Span, SECRET_ENTITY_TYPES,
};
pub use trunk::Lfm2Trunk;
