//! Text → vector for `LFM2.5-Embedding-350M`: tokenize, run the trunk,
//! CLS-pool.
//!
//! Three things about this checkpoint are easy to get wrong in ways that
//! produce perfectly reasonable-looking vectors, so the API makes all three
//! explicit rather than defaulted:
//!
//! 1. **It is asymmetric (E5-style).** Queries and documents get different
//!    prefixes — `"query: "` and `"document: "`, from
//!    `config_sentence_transformers.json`. This is not cosmetic: the same
//!    text under the two prefixes embeds at cosine ≈ 0.70. Using one
//!    prefix for both sides quietly degrades retrieval and nothing errors.
//!    Hence [`TextKind`] is a required argument, not an option with a
//!    default.
//! 2. **Pooling is CLS**, not mean — `1_Pooling/config.json` says
//!    `pooling_mode_cls_token: true`. With `add_bos_token: true`, position
//!    0 is `<|startoftext|>`.
//! 3. **Vectors come out un-normalized.** The sentence-transformers module
//!    list has no Normalize step; `similarity_fn_name` is cosine, which
//!    normalizes at comparison time. [`Lfm2Embedding::embed`] returns the
//!    raw vector; use [`Lfm2Embedding::embed_normalized`] if you want to
//!    compare with a dot product.

use std::path::Path;
use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;

use crate::config::Lfm2EncoderConfig;
use crate::error::{Error, Result};
use crate::trunk::Lfm2Trunk;

/// Which side of the asymmetric pair this text is.
///
/// There is deliberately no `Default`: picking one for the caller is the
/// mistake this type exists to prevent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextKind {
    /// The search query — prefixed `"query: "`.
    Query,
    /// The corpus text being searched — prefixed `"document: "`.
    Document,
}

impl TextKind {
    /// The prefix LiquidAI trained with, verbatim (trailing space included).
    pub fn prefix(self) -> &'static str {
        match self {
            TextKind::Query => "query: ",
            TextKind::Document => "document: ",
        }
    }
}

/// `sentence_bert_config.json` caps sequences here. Longer inputs are
/// truncated; the model's `max_position_embeddings` claims 128000, but 512
/// is what it was trained and evaluated at.
pub const MAX_SEQ_LEN: usize = 512;

/// A loaded embedding model: tokenizer + trunk.
#[derive(Debug)]
pub struct Lfm2Embedding {
    trunk: Arc<Lfm2Trunk>,
    tokenizer: Tokenizer,
    hidden_size: usize,
}

impl Lfm2Embedding {
    /// Load from a checkpoint directory holding `config.json`,
    /// `tokenizer.json` and `model.safetensors`.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        Self::from_dir_with(dir, DType::F32, &Device::Cpu)
    }

    /// As [`Self::from_dir`], choosing the compute dtype and device. F32 on
    /// CPU is the verified path; the checkpoint itself ships bf16.
    pub fn from_dir_with(dir: impl AsRef<Path>, dtype: DType, device: &Device) -> Result<Self> {
        let dir = dir.as_ref();

        let cfg_path = dir.join("config.json");
        let bytes = std::fs::read(&cfg_path).map_err(|e| Error::io(&cfg_path, e))?;
        let cfg =
            Lfm2EncoderConfig::from_json(&bytes).map_err(|source| Error::ConfigParse {
                path: cfg_path.clone(),
                source,
            })?;
        cfg.validate().map_err(|message| Error::ConfigInvalid {
            path: cfg_path.clone(),
            message,
        })?;

        let tok_path = dir.join("tokenizer.json");
        let mut tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| Error::Tokenizer {
            path: tok_path.clone(),
            message: e.to_string(),
        })?;
        // Truncate rather than let a long input silently blow past what the
        // model was trained at.
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: MAX_SEQ_LEN,
                ..Default::default()
            }))
            .map_err(|e| Error::Tokenizer {
                path: tok_path.clone(),
                message: e.to_string(),
            })?;

        let weights = dir.join("model.safetensors");
        if !weights.is_file() {
            return Err(Error::io(
                &weights,
                std::io::Error::new(std::io::ErrorKind::NotFound, "no model.safetensors"),
            ));
        }
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], dtype, device)? };
        let trunk = Arc::new(Lfm2Trunk::load(&cfg, vb)?);

        Ok(Self {
            trunk,
            tokenizer,
            hidden_size: cfg.hidden_size,
        })
    }

    /// Build over an already-loaded, possibly-shared trunk. This head has no
    /// weights of its own beyond the trunk, so this loads ONLY `dir`'s
    /// `tokenizer.json` — `dir`'s `model.safetensors` is never opened at
    /// all. See
    /// [`crate::sequence_classification::Lfm2SequenceClassifier::from_trunk`]
    /// for the fuller rationale.
    ///
    /// Errors loudly, naming every mismatched field, if `dir`'s config
    /// disagrees with the trunk's shape — see [`Lfm2Trunk::check_compatible`].
    pub fn from_trunk(trunk: Arc<Lfm2Trunk>, dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();

        let cfg_path = dir.join("config.json");
        let bytes = std::fs::read(&cfg_path).map_err(|e| Error::io(&cfg_path, e))?;
        let cfg =
            Lfm2EncoderConfig::from_json(&bytes).map_err(|source| Error::ConfigParse {
                path: cfg_path.clone(),
                source,
            })?;
        cfg.validate().map_err(|message| Error::ConfigInvalid {
            path: cfg_path.clone(),
            message,
        })?;
        trunk
            .check_compatible(&cfg)
            .map_err(|message| Error::ConfigInvalid { path: cfg_path.clone(), message })?;

        let tok_path = dir.join("tokenizer.json");
        let mut tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| Error::Tokenizer {
            path: tok_path.clone(),
            message: e.to_string(),
        })?;
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: MAX_SEQ_LEN,
                ..Default::default()
            }))
            .map_err(|e| Error::Tokenizer {
                path: tok_path.clone(),
                message: e.to_string(),
            })?;

        Ok(Self {
            trunk,
            tokenizer,
            hidden_size: cfg.hidden_size,
        })
    }

    /// Dimension of the vectors this model produces (1024).
    pub fn dim(&self) -> usize {
        self.hidden_size
    }

    /// Token ids for `text` under `kind`'s prefix, exactly as the reference
    /// pipeline produces them. Exposed because a tokenizer mismatch is the
    /// failure most worth being able to inspect directly.
    pub fn token_ids(&self, text: &str, kind: TextKind) -> Result<Vec<u32>> {
        let prefixed = format!("{}{text}", kind.prefix());
        let encoding = self
            .tokenizer
            .encode(prefixed, true)
            .map_err(|e| Error::Encode(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Embed one text. Returns the raw (un-normalized) CLS vector.
    ///
    /// One sequence at a time, unpadded — the reproducible path. See the
    /// crate docs on why padding is not inert here.
    pub fn embed(&self, text: &str, kind: TextKind) -> Result<Vec<f32>> {
        let ids = self.token_ids(text, kind)?;
        let device = self.trunk.device();
        let input = Tensor::new(ids.as_slice(), device)?.unsqueeze(0)?;
        let hidden = self.trunk.forward(&input, None)?;
        let pooled = Lfm2Trunk::pool_cls(&hidden)?.to_dtype(DType::F32)?;
        Ok(pooled.squeeze(0)?.to_vec1::<f32>()?)
    }

    /// Embed and L2-normalize, so cosine similarity is a plain dot product.
    pub fn embed_normalized(&self, text: &str, kind: TextKind) -> Result<Vec<f32>> {
        let mut v = self.embed(text, kind)?;
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        // A zero vector can't be normalized; returning it unchanged would
        // hand the caller something that silently breaks cosine similarity.
        if !norm.is_normal() {
            return Err(Error::Encode(format!(
                "embedding has degenerate norm {norm} — cannot normalize"
            )));
        }
        for x in &mut v {
            *x /= norm;
        }
        Ok(v)
    }

    /// The underlying trunk, for callers that want hidden states rather
    /// than a pooled vector.
    pub fn trunk(&self) -> &Lfm2Trunk {
        &self.trunk
    }

    /// This checkpoint's own tokenizer, for a caller that needs to inspect
    /// tokenization directly (e.g. `lfm2d`'s `POST /v1/tokenize`) rather
    /// than through [`Self::token_ids`]'s fixed `TextKind` prefix.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
}

/// Cosine similarity between two vectors of equal length.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "cosine similarity needs equal lengths");
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}
