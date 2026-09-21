//! Token classification — `LFM2.5-Encoder-350M-PII-Detector` and friends:
//! a per-token BIOES head over the checkpoint's label set.
//!
//! # The label set comes from `config.json`, never `label_schema.json`
//!
//! The PII checkpoint ships BOTH, and they disagree: `label_schema.json`
//! declares **109 labels over 27 types**, while `config.json` and the
//! actual `classifier.weight` are **161 labels over 40 types**. The schema
//! file is stale, and what it omits is not incidental — it is missing
//! `credential.connection_string`, `credential.jwt`, `credential.password`
//! and `credential.private_key`, i.e. every credential type a guard tool
//! would be reading this model for. Decoding 161-wide logits through that
//! file's `id2label` would mislabel silently.
//!
//! # This head is not the whole product
//!
//! The checkpoint also ships `pii_hybrid_decode.py`, which wraps this head
//! in a regex + validator tier and is what LiquidAI calls the intended
//! decode. That tier is deliberately NOT implemented here: it is product
//! policy rather than the model, and measurement does not flatter it —
//! on our eval set it fires on 30% of clean text against this head's 4%,
//! and its ordered regexes can mislabel a connection string as an email
//! (the email pattern claims `user:password@host` first, suppressing the
//! `credential.connection_string` this head gets right). Consumers who
//! want it should port it deliberately, with that trade in view.

use std::path::Path;
use std::sync::Arc;

use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::{Linear, VarBuilder};
use tokenizers::Tokenizer;

use crate::config::Lfm2EncoderConfig;
use crate::error::{Error, Result};
use crate::labels::order_labels;
use crate::trunk::Lfm2Trunk;

/// Per-token predicted label ids, each token's byte span into the input,
/// and the softmax probability of its predicted class — the shape
/// [`Lfm2TokenClassifier::token_labels`] returns and its three public
/// per-axis accessors slice apart.
type TokenLabels = (Vec<usize>, Vec<(usize, usize)>, Vec<f32>);

/// A detected entity, with byte offsets into the input string.
#[derive(Debug, Clone, PartialEq)]
pub struct Span {
    /// Byte offset of the first byte, into the text passed to `predict`.
    pub start: usize,
    /// Byte offset one past the last byte.
    pub end: usize,
    /// Entity type with the BIOES prefix stripped, e.g.
    /// `credential.api_key`.
    pub label: String,
    /// Confidence in this span: the **minimum** over its tokens of the
    /// softmax probability of that token's predicted class.
    ///
    /// Minimum, not mean, because a span is a CONJUNCTION of per-token
    /// decisions — it is wrong if any one of them is wrong, so its
    /// trustworthiness is that of its weakest token. Averaging would hide
    /// exactly the token that says the boundary may be misplaced, which for
    /// a credential is the interesting failure. Amy's ruling, 2026-08-11.
    ///
    /// Consequence worth knowing: these read systematically LOWER than
    /// services that average (Hugging Face's grouped-entity pipeline) or
    /// report a recognizer's own confidence (Presidio). Do not compare the
    /// numbers across tools. As everywhere else in this crate, it is a
    /// ranking signal with no calibrated absolute meaning — nothing here
    /// thresholds it.
    pub score: f32,
}

impl Span {
    /// The matched text. Panics if the offsets are not char boundaries,
    /// which would mean the tokenizer and the input disagree.
    pub fn text<'a>(&self, source: &'a str) -> &'a str {
        &source[self.start..self.end]
    }
}

/// Entity types that are SECRETS — the set [`Lfm2TokenClassifier::credentials`]
/// filters to, and the one a boundary guard keys on.
///
/// The whole `credential.*` family, **plus `developer.login_credentials`**.
/// That second one is not tidiness: the PII checkpoint's taxonomy carries a
/// `credential.*` family *and* a `developer.*` one, and the model routes real
/// secrets to the latter. Measured against the live checkpoint 2026-08-11:
///
/// ```text
/// postgres://admin:hunter2@10.0.0.5:5432/prod -> credential.connection_string
/// Authorization: Bearer eyJhbGciOiJIUzI1Ni...  -> developer.login_credentials 0.972
/// AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE       -> developer.login_credentials 0.596
/// ```
///
/// A `credential.`-prefix filter therefore returned NOTHING for a JWT the
/// model was 97% sure about, and nothing for an AWS access key — a guard
/// trusting it would fail open, silently, on two of the most common secret
/// shapes there are. Note `credential.jwt` exists in the taxonomy; the model
/// simply doesn't use it for bearer tokens. Prefix-matching a label family
/// assumed the taxonomy's naming was semantic, and it isn't.
///
/// `developer.device_id` is deliberately EXCLUDED — an identifier, not a
/// secret.
///
/// This is a starting set, not a settled one (Amy, 2026-08-11: *"do the
/// simple thing now, we'll tune the classifiers over time"*). Widen it from
/// measurement — check what the model actually emits for a secret shape
/// before assuming a label name covers it.
pub const SECRET_ENTITY_TYPES: &[&str] = &["developer.login_credentials"];

/// Whether an entity type counts as a secret: any `credential.*`, or a
/// member of [`SECRET_ENTITY_TYPES`].
pub fn is_secret_label(label: &str) -> bool {
    label.starts_with("credential.") || SECRET_ENTITY_TYPES.contains(&label)
}

/// A loaded token-classification checkpoint.
#[derive(Debug)]
pub struct Lfm2TokenClassifier {
    trunk: Arc<Lfm2Trunk>,
    classifier: Linear,
    tokenizer: Tokenizer,
    /// Label string per class id, ordered by id.
    id2label: Vec<String>,
}

/// Parse+validate `config.json`, pull out `id2label`, and load the
/// tokenizer — the prefix shared by [`Lfm2TokenClassifier::from_dir_with`]
/// (which goes on to load its own trunk) and
/// [`Lfm2TokenClassifier::from_trunk`] (which reuses one instead).
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
        message: "no id2label: this checkpoint carries no token-classification labels"
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

impl Lfm2TokenClassifier {
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

        // A plain Linear over the trunk's per-token hidden states, WITH a
        // bias (unlike every projection inside the trunk). The checkpoint
        // also ships `class_weights` — a training-time class-balancing
        // artifact, not an inference parameter; loading it into anything
        // would be wrong.
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
    /// trunk's own dtype/device. `dir`'s own trunk weights (under `lfm2.`
    /// in the same `model.safetensors`) are never read; see
    /// [`crate::sequence_classification::Lfm2SequenceClassifier::from_trunk`]
    /// for the fuller rationale — this head mirrors it exactly.
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
    /// want to confirm two heads share the same loaded trunk.
    pub fn trunk(&self) -> &Lfm2Trunk {
        &self.trunk
    }

    /// Number of classes (161 for the PII detector).
    pub fn num_labels(&self) -> usize {
        self.id2label.len()
    }

    /// The distinct entity types, BIOES prefixes stripped.
    pub fn entity_types(&self) -> Vec<&str> {
        let mut types: Vec<&str> = self
            .id2label
            .iter()
            .filter_map(|l| l.split_once('-').map(|(_, t)| t))
            .collect();
        types.sort_unstable();
        types.dedup();
        types
    }

    /// Per-token predicted label ids, alongside each token's byte span and
    /// the softmax probability of the predicted class.
    fn token_labels(&self, text: &str) -> Result<TokenLabels> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| Error::Encode(e.to_string()))?;
        let ids = encoding.get_ids();
        let offsets: Vec<(usize, usize)> = encoding.get_offsets().to_vec();

        let device = self.trunk.device();
        let input = Tensor::new(ids, device)?.unsqueeze(0)?;
        let hidden = self.trunk.forward(&input, None)?;
        let logits = self.classifier.forward(&hidden)?.to_dtype(DType::F32)?;

        let logits = logits.i(0)?;
        let best = logits.argmax(1)?.to_vec1::<u32>()?;

        // Softmax over the label axis, then take the probability of the
        // class that argmax already chose. Computed here rather than in a
        // caller because reproducing the label ordering outside this file is
        // how a 161-wide output gets silently mislabelled.
        let probs = candle_nn::ops::softmax_last_dim(&logits)?.to_vec2::<f32>()?;
        let confidences: Vec<f32> = best
            .iter()
            .zip(&probs)
            .map(|(&class, row)| row[class as usize])
            .collect();

        Ok((
            best.into_iter().map(|i| i as usize).collect(),
            offsets,
            confidences,
        ))
    }

    /// Per-token predicted class ids, one per token including specials.
    ///
    /// Exposed so a divergence in the MODEL can be told apart from a
    /// divergence in the span DECODE — they have completely different
    /// causes and the distinction is worth being able to make directly.
    pub fn token_label_ids(&self, text: &str) -> Result<Vec<usize>> {
        Ok(self.token_labels(text)?.0)
    }

    /// Per-token softmax probability of each token's own predicted class,
    /// one per token including specials — the raw material behind
    /// [`Span::score`], exposed for the same reason as
    /// [`Self::token_label_ids`]: so a confidence question can be asked of
    /// the MODEL without going through the span decode.
    pub fn token_confidences(&self, text: &str) -> Result<Vec<f32>> {
        Ok(self.token_labels(text)?.2)
    }

    /// Each token's byte span into `text`, one per token including specials
    /// (which carry an empty `(n, n)` range). Same ordering as
    /// [`Self::token_label_ids`] and [`Self::token_confidences`], so the
    /// three zip together — which is what lets a test check the span decode
    /// against the per-token evidence it was built from.
    pub fn token_offsets(&self, text: &str) -> Result<Vec<(usize, usize)>> {
        Ok(self.token_labels(text)?.1)
    }

    /// Detect entities in `text`.
    ///
    /// Decoding follows the checkpoint's own `model_spans`, which is
    /// deliberately forgiving about malformed BIOES: a span starts on `B-`
    /// or `S-` or a change of type, and otherwise extends. Tokens with an
    /// empty offset span (the specials) close the current span, and
    /// leading/trailing whitespace is trimmed off the result.
    pub fn predict(&self, text: &str) -> Result<Vec<Span>> {
        let (label_ids, offsets, confidences) = self.token_labels(text)?;

        let mut spans: Vec<Span> = Vec::new();
        let mut current: Option<Span> = None;

        for ((&label_id, &(start, end)), &confidence) in
            label_ids.iter().zip(&offsets).zip(&confidences)
        {
            let label = self
                .id2label
                .get(label_id)
                .map(String::as_str)
                .unwrap_or("O");

            if end <= start || label == "O" {
                spans.extend(current.take());
                continue;
            }
            let (prefix, entity) = match label.split_once('-') {
                Some((p, t)) => (p, t),
                None => ("S", label),
            };

            let starts_new = matches!(prefix, "B" | "S")
                || current.as_ref().is_none_or(|c| c.label != entity);
            if starts_new {
                spans.extend(current.take());
                current = Some(Span {
                    start,
                    end,
                    label: entity.to_string(),
                    score: confidence,
                });
            } else if let Some(c) = current.as_mut() {
                c.end = end;
                // Weakest link: see Span::score.
                c.score = c.score.min(confidence);
            }
        }
        spans.extend(current);

        // Trim whitespace, then drop anything that trimmed to nothing.
        let bytes = text.as_bytes();
        for span in &mut spans {
            while span.start < span.end && bytes[span.start].is_ascii_whitespace() {
                span.start += 1;
            }
            while span.end > span.start && bytes[span.end - 1].is_ascii_whitespace() {
                span.end -= 1;
            }
        }
        spans.retain(|s| s.end > s.start);
        Ok(spans)
    }

    /// Spans whose type is a `credential.*` — the boundary-guard question.
    pub fn credentials(&self, text: &str) -> Result<Vec<Span>> {
        Ok(self
            .predict(text)?
            .into_iter()
            .filter(|s| is_secret_label(&s.label))
            .collect())
    }
}
