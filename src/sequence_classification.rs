//! Sequence classification — a whole-sequence head over the trunk, for OUR
//! OWN fine-tunes (e.g. a commandline-safety advisory classifier). There is
//! no LiquidAI checkpoint shipping this head yet; the loading contract below
//! is what a companion trainer targets, so it is fixed rather than inferred.
//!
//! # The contract
//!
//! - `config.json`: a normal LFM2 encoder config, `architectures: ["Lfm2BidirForSequenceClassification"]`,
//!   plus `id2label` (string keys `"0"`, `"1"`, … to label names — same
//!   shape and the same gap problem as [`crate::token_classification`]'s,
//!   handled by the same [`crate::labels::order_labels`]).
//! - `model.safetensors`: the trunk under `lfm2.` (see [`Lfm2Trunk::load`]),
//!   plus `classifier.weight` `[num_labels, hidden_size]` and
//!   `classifier.bias` `[num_labels]` — a plain `Linear` WITH bias, same as
//!   the token-classification head's.
//! - Pooling is **CLS** (hidden state at position 0), matching the Embedding
//!   checkpoint's `pooling_mode_cls_token: true` — deliberate, not a mean
//!   pooling default, via [`Lfm2Trunk::pool_cls`].
//!
//! # These heads are advisory
//!
//! [`Lfm2SequenceClassifier::logits`] is exposed alongside the softmax
//! [`Lfm2SequenceClassifier::predict`] specifically so a caller can do their
//! own calibration/thresholding — an advisory safety classifier is not
//! something you want hard-wired to argmax at a fixed 0.5 boundary.

use std::path::Path;
use std::sync::Arc;

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Linear, VarBuilder};
use tokenizers::Tokenizer;

use crate::config::Lfm2EncoderConfig;
use crate::error::{Error, Result};
use crate::labels::order_labels;
use crate::trunk::Lfm2Trunk;

/// A loaded sequence-classification checkpoint.
#[derive(Debug)]
pub struct Lfm2SequenceClassifier {
    trunk: Arc<Lfm2Trunk>,
    classifier: Linear,
    tokenizer: Tokenizer,
    /// Label string per class id, ordered by id.
    id2label: Vec<String>,
}

/// Parse+validate `config.json`, pull out `id2label`, and load the
/// tokenizer — the prefix shared by [`Lfm2SequenceClassifier::from_dir_with`]
/// (which goes on to load its own trunk) and
/// [`Lfm2SequenceClassifier::from_trunk`] (which reuses one instead).
/// Factored out so the two loading paths cannot drift apart on what counts
/// as a valid checkpoint.
fn load_head_metadata(dir: &Path) -> Result<(Lfm2EncoderConfig, Vec<String>, Tokenizer)> {
    let cfg_path = dir.join("config.json");
    let bytes = std::fs::read(&cfg_path).map_err(|e| Error::io(&cfg_path, e))?;
    let cfg = Lfm2EncoderConfig::from_json(&bytes).map_err(|source| Error::ConfigParse {
        path: cfg_path.clone(),
        source,
    })?;
    cfg.validate().map_err(|message| Error::ConfigInvalid {
        path: cfg_path.clone(),
        message,
    })?;

    let labels = cfg.id2label.as_ref().ok_or_else(|| Error::ConfigInvalid {
        path: cfg_path.clone(),
        message: "no id2label: this checkpoint carries no sequence-classification labels"
            .to_string(),
    })?;
    let id2label = order_labels(labels, &cfg_path)?;

    let tok_path = dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| Error::Tokenizer {
        path: tok_path.clone(),
        message: e.to_string(),
    })?;

    Ok((cfg, id2label, tokenizer))
}

impl Lfm2SequenceClassifier {
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        Self::from_dir_with(dir, DType::F32, &Device::Cpu)
    }

    pub fn from_dir_with(dir: impl AsRef<Path>, dtype: DType, device: &Device) -> Result<Self> {
        let dir = dir.as_ref();
        let (cfg, id2label, tokenizer) = load_head_metadata(dir)?;

        let weights = dir.join("model.safetensors");
        if !weights.is_file() {
            return Err(Error::io(
                &weights,
                std::io::Error::new(std::io::ErrorKind::NotFound, "no model.safetensors"),
            ));
        }
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], dtype, device)? };
        let trunk = Arc::new(Lfm2Trunk::load(&cfg, vb.clone())?);

        // A plain Linear over the CLS-pooled hidden state, WITH a bias
        // (unlike every projection inside the trunk) — same convention as
        // the token-classification head's `classifier`.
        let classifier = candle_nn::linear(cfg.hidden_size, id2label.len(), vb.pp("classifier"))?;

        Ok(Self {
            trunk,
            classifier,
            tokenizer,
            id2label,
        })
    }

    /// Build a head over an already-loaded, possibly-shared trunk — loads
    /// ONLY the tokenizer and the classifier weights from `dir`, at the
    /// trunk's own dtype/device (so a dtype/device mismatch is structurally
    /// impossible, not merely checked). This is the frozen-trunk specialist
    /// path: `dir`'s own trunk weights (under `lfm2.` in the same
    /// `model.safetensors`) are never read — the already-loaded, shared
    /// trunk is reused as-is, so N heads cost one trunk's memory instead of
    /// N.
    ///
    /// Errors loudly, naming every mismatched field, if `dir`'s config
    /// disagrees with the trunk's shape — see [`Lfm2Trunk::check_compatible`].
    pub fn from_trunk(trunk: Arc<Lfm2Trunk>, dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let (cfg, id2label, tokenizer) = load_head_metadata(dir)?;

        let cfg_path = dir.join("config.json");
        trunk
            .check_compatible(&cfg)
            .map_err(|message| Error::ConfigInvalid { path: cfg_path, message })?;

        let weights = dir.join("model.safetensors");
        if !weights.is_file() {
            return Err(Error::io(
                &weights,
                std::io::Error::new(std::io::ErrorKind::NotFound, "no model.safetensors"),
            ));
        }
        // Only `classifier.*` is pulled from this mmap; the trunk tensors
        // it also carries under `lfm2.` are never touched.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights], trunk.dtype(), trunk.device())?
        };
        let classifier = candle_nn::linear(cfg.hidden_size, id2label.len(), vb.pp("classifier"))?;

        Ok(Self {
            trunk,
            classifier,
            tokenizer,
            id2label,
        })
    }

    /// The underlying trunk, for callers that want raw hidden states or
    /// want to confirm two heads share the same loaded trunk (e.g.
    /// `std::ptr::eq`).
    pub fn trunk(&self) -> &Lfm2Trunk {
        &self.trunk
    }

    /// Number of classes.
    pub fn num_labels(&self) -> usize {
        self.id2label.len()
    }

    /// Label strings, ordered by class id.
    pub fn labels(&self) -> &[String] {
        &self.id2label
    }

    /// This checkpoint's own tokenizer, for a caller that needs to inspect
    /// tokenization directly (e.g. `lfm2d`'s `POST /v1/tokenize`).
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Tokenize, run the trunk, CLS-pool, and project to logits — the shared
    /// path under [`Self::logits`] and [`Self::predict`].
    fn compute_logits(&self, text: &str) -> Result<Tensor> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| Error::Encode(e.to_string()))?;
        let ids = encoding.get_ids();

        let device = self.trunk.device();
        let input = Tensor::new(ids, device)?.unsqueeze(0)?;
        // One sequence at a time, unpadded, so no attention_mask is needed —
        // the same reproducible-path convention as `Lfm2Embedding::embed`.
        let hidden = self.trunk.forward(&input, None)?;
        let pooled = Lfm2Trunk::pool_cls(&hidden)?;
        let logits = self.classifier.forward(&pooled)?.to_dtype(DType::F32)?;
        Ok(logits)
    }

    /// Pre-softmax class scores. Exposed for calibration/thresholding: these
    /// are advisory models, and a caller may want a threshold other than
    /// "argmax at 0.5" over the softmax output.
    pub fn logits(&self, text: &str) -> Result<Vec<f32>> {
        Ok(self.compute_logits(text)?.squeeze(0)?.to_vec1::<f32>()?)
    }

    /// Class probabilities: softmax over [`Self::logits`].
    pub fn predict(&self, text: &str) -> Result<Vec<f32>> {
        let logits = self.compute_logits(text)?;
        let probs = candle_nn::ops::softmax_last_dim(&logits)?;
        Ok(probs.squeeze(0)?.to_vec1::<f32>()?)
    }

    /// The argmax label and its probability.
    pub fn predict_label(&self, text: &str) -> Result<(&str, f32)> {
        let probs = self.predict(text)?;
        let (idx, prob) = probs
            .iter()
            .copied()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .ok_or_else(|| Error::Encode("empty probability vector".to_string()))?;
        Ok((self.id2label[idx].as_str(), prob))
    }
}
