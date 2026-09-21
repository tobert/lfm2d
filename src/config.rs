//! Checkpoint configuration — the `config.json` shapes LiquidAI ships for
//! the LFM2.5 bidirectional-encoder family.
//!
//! One struct parses every encoder checkpoint we care about; the
//! `architectures` field tells us which head the weights carry. Verified
//! against real fixtures in `tests/fixtures/` (fetched from the Hub
//! 2026-08-03) rather than transcribed from docs — the checkpoints
//! disagree with each other in instructive ways (the PII detector is a
//! `Lfm2BidirP2ForTokenClassification` with `full_attn_idxs`; the Router
//! is a `Lfm2BidirForSequenceRouting` with `rule_proj_dim`, not a plain
//! classifier).
//!
//! Parsing is deliberately permissive (no deny_unknown_fields): these
//! configs are upstream-owned and grow keys; what we DO read is pinned by
//! the fixture tests.

use std::collections::HashMap;

use serde::Deserialize;

/// Per-layer kind in the hybrid stack, from `layer_types`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    FullAttention,
    Conv,
}

/// `rope_parameters` sub-object (some checkpoints also carry a flat
/// `rope_theta`; [`Lfm2EncoderConfig::rope_theta`] resolves the precedence).
#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    pub rope_theta: Option<f64>,
}

/// Which head this checkpoint's weights carry, derived from
/// `architectures[0]`.
///
/// Matched by substring, not exact name — LiquidAI's naming already
/// drifts within the family (`Lfm2Bidirectional…` vs `Lfm2Bidir…` vs the
/// `P2` revision infix), and a new revision infix should not knock a
/// checkpoint into `Unknown`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncoderArch {
    /// Bare bidirectional trunk (the embedding checkpoint).
    BidirectionalModel,
    /// Masked-LM head (the fine-tune base checkpoints).
    MaskedLm,
    /// Token-classification head (PII detector, policy linter).
    TokenClassification,
    /// Sequence-routing head (the Prompt-Router: scores a prompt against
    /// rule projections — `rule_proj_dim` — not a fixed label head).
    SequenceRouting,
    /// Token-level rule-matching head (the Policy-Linter): scores every
    /// token against free-text rule projections. Same `rule_proj_dim`
    /// machinery as [`EncoderArch::SequenceRouting`], but per-token rather
    /// than per-sequence.
    RuleMatching,
    /// Plain sequence-classification head (none shipped yet; anticipated).
    SequenceClassification,
    /// Anything we don't recognize — carried verbatim so the caller can
    /// report it instead of us guessing.
    Unknown(String),
}

/// Parsed `config.json` for an LFM2.5 encoder checkpoint.
#[derive(Debug, Clone, Deserialize)]
pub struct Lfm2EncoderConfig {
    #[serde(default)]
    pub architectures: Vec<String>,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f64,
    pub layer_types: Vec<LayerType>,
    pub max_position_embeddings: usize,
    /// The config's *nominal* FFN width. Do not build an MLP from this —
    /// it disagrees with the shipped weights on three of four checkpoints.
    /// Use [`Lfm2EncoderConfig::ffn_dim`].
    pub intermediate_size: usize,
    /// Present only on the Embedding checkpoint; when present it, not
    /// `intermediate_size`, is the FFN base width.
    block_ff_dim: Option<usize>,
    /// When set, the base width is Llama-style two-thirds-adjusted before
    /// rounding. Absent on the 230M base, which uses its width verbatim.
    #[serde(default)]
    block_auto_adjust_ff_dim: bool,
    #[serde(default = "default_ffn_dim_multiplier")]
    block_ffn_dim_multiplier: f64,
    #[serde(default = "default_block_multiple_of")]
    block_multiple_of: usize,
    /// Short-conv kernel/cache length (`conv_L_cache` upstream).
    #[serde(rename = "conv_L_cache")]
    pub conv_l_cache: usize,
    #[serde(default)]
    pub conv_bias: bool,
    pub conv_dim: Option<usize>,
    pub conv_dim_out: Option<usize>,
    /// Flat form; older/embedding checkpoints carry it.
    rope_theta: Option<f64>,
    /// Structured form; newer checkpoints carry it.
    rope_parameters: Option<RopeParameters>,
    #[serde(default)]
    pub use_pos_enc: bool,
    /// Label map for classification heads (absent elsewhere).
    pub id2label: Option<HashMap<String, String>>,
    /// Router-only: dimension of the rule projection space.
    pub rule_proj_dim: Option<usize>,
    /// PII/P2-only: explicit indices of the full-attention layers. When
    /// present it must agree with `layer_types`; we trust `layer_types`
    /// and keep this only for cross-checking.
    pub full_attn_idxs: Option<Vec<usize>>,
    pub pad_token_id: Option<u32>,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
}

fn default_norm_eps() -> f64 {
    1e-5
}

fn default_ffn_dim_multiplier() -> f64 {
    1.0
}

fn default_block_multiple_of() -> usize {
    256
}

/// LFM2's documented RoPE default when neither config form carries one.
const DEFAULT_ROPE_THETA: f64 = 1_000_000.0;

impl Lfm2EncoderConfig {
    /// Parse a checkpoint's `config.json` bytes.
    pub fn from_json(bytes: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(bytes)
    }

    /// The head this checkpoint carries, from `architectures[0]`.
    pub fn arch(&self) -> EncoderArch {
        let Some(name) = self.architectures.first() else {
            return EncoderArch::Unknown(String::new());
        };
        // Order matters: every `For…` head must be tested BEFORE the bare
        // `Bidir` fallback, or a head-carrying checkpoint gets misread as a
        // headless trunk and quietly loses its head. That is exactly how
        // `Lfm2BidirForRuleMatching` (Policy-Linter) slipped through on day
        // 0 — hence the belt-and-braces guard below.
        if name.contains("ForTokenClassification") {
            EncoderArch::TokenClassification
        } else if name.contains("ForSequenceRouting") {
            EncoderArch::SequenceRouting
        } else if name.contains("ForRuleMatching") {
            EncoderArch::RuleMatching
        } else if name.contains("ForSequenceClassification") {
            EncoderArch::SequenceClassification
        } else if name.contains("ForMaskedLM") {
            EncoderArch::MaskedLm
        } else if name.contains("For") {
            // A head we've never seen. Refuse to guess: report it verbatim
            // so the caller fails with a name to go read on the Hub,
            // instead of loading a trunk and dropping the head.
            EncoderArch::Unknown(name.clone())
        } else if name.contains("Bidirectional") || name.contains("Bidir") {
            EncoderArch::BidirectionalModel
        } else {
            EncoderArch::Unknown(name.clone())
        }
    }

    /// Effective RoPE theta: flat key wins, then the structured form, then
    /// the LFM2 default.
    pub fn rope_theta(&self) -> f64 {
        self.rope_theta
            .or(self.rope_parameters.as_ref().and_then(|r| r.rope_theta))
            .unwrap_or(DEFAULT_ROPE_THETA)
    }

    /// Per-head width. LFM2 derives it rather than shipping it.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// The FFN width the *weights* actually have.
    ///
    /// `intermediate_size` in config.json is the pre-adjustment nominal
    /// width and is wrong on three of the four checkpoints (Embedding, PII,
    /// Router all say 6656 while shipping 4608). LiquidAI applies the
    /// Llama-style two-thirds adjustment and rounds up to
    /// `block_multiple_of`:
    ///
    /// ```text
    /// base = block_ff_dim ?? intermediate_size
    /// if block_auto_adjust_ff_dim { base = 2*base/3 }
    /// round_up(base * block_ffn_dim_multiplier, block_multiple_of)
    /// ```
    ///
    /// Verified against every checkpoint's `feed_forward.w1.weight`
    /// out-features (safetensors headers, 2026-08-04) — see the fixture
    /// test. Note this differs from candle-transformers' causal
    /// `compute_intermediate_size`, which starts from `hidden_size * 4` and
    /// would yield 4096 here.
    pub fn ffn_dim(&self) -> usize {
        let base = self.block_ff_dim.unwrap_or(self.intermediate_size);
        let base = if self.block_auto_adjust_ff_dim {
            2 * base / 3
        } else {
            base
        };
        let scaled = (base as f64 * self.block_ffn_dim_multiplier) as usize;
        scaled.div_ceil(self.block_multiple_of) * self.block_multiple_of
    }

    /// Dimension of the rule-projection space the routing/rule-matching
    /// heads score in. `None` when the key is absent from `config.json`;
    /// the Python head reads it as `getattr(config, "rule_proj_dim", 256)`,
    /// so 256 is the default here too, not an error — every shipped
    /// checkpoint carries the key explicitly (256), but a future one might
    /// rely on the default the way the Python code allows.
    pub fn rule_proj_dim(&self) -> usize {
        self.rule_proj_dim.unwrap_or(256)
    }

    /// Number of classification labels, when this checkpoint has a label
    /// head. `None` (not zero) for label-less checkpoints — a caller
    /// building a classifier over an embedding checkpoint should fail
    /// loudly, not allocate a 0-wide head.
    pub fn num_labels(&self) -> Option<usize> {
        self.id2label.as_ref().map(|m| m.len())
    }

    /// Sanity checks that catch a mismatched or truncated config before a
    /// weight load turns it into a shape error three layers deep.
    pub fn validate(&self) -> Result<(), String> {
        if self.layer_types.len() != self.num_hidden_layers {
            return Err(format!(
                "layer_types has {} entries but num_hidden_layers is {}",
                self.layer_types.len(),
                self.num_hidden_layers
            ));
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            return Err(format!(
                "num_attention_heads {} not divisible by num_key_value_heads {}",
                self.num_attention_heads, self.num_key_value_heads
            ));
        }
        if let Some(idxs) = &self.full_attn_idxs {
            let from_types: Vec<usize> = self
                .layer_types
                .iter()
                .enumerate()
                .filter(|(_, t)| **t == LayerType::FullAttention)
                .map(|(i, _)| i)
                .collect();
            if *idxs != from_types {
                return Err(format!(
                    "full_attn_idxs {idxs:?} disagrees with layer_types {from_types:?}"
                ));
            }
        }
        Ok(())
    }
}
