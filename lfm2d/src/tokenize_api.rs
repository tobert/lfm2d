//! `POST /v1/tokenize`: per-model tokenization, answered on the request
//! handler over a CLONED tokenizer — it never touches a worker's job queue,
//! so it answers even while the adjudicator or an encoder head is mid-job.
//! Ruled 2026-09-23, `docs/system1-split-plan.md` "Tokenize and probe
//! endpoints".
//!
//! `model` names any loaded model with its own tokenizer — the adjudicator
//! and every encoder head each have one — as listed by `GET /v1/models`.
//! `main.rs` builds one [`TokenizerRegistry`] from every loaded model's
//! tokenizer BEFORE handing that model's owning engine to its worker
//! ([`crate::worker::WorkerHandle::spawn_crash_on_panic`] for the encoder
//! heads, [`crate::adjudicator::Handle::spawn`] for the adjudicator) — see
//! each's doc comment on why the clone has to happen at that exact point.
use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::post;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use crate::types::ApiError;

/// One loaded model's tokenizer plus its own audit hash (sha256 of its
/// `tokenizer.json`, matching every other hash this crate reports — see
/// `hash.rs`).
pub struct TokenizerEntry {
    pub tokenizer: tokenizers::Tokenizer,
    pub tokenizer_hash: String,
}

/// Every loaded model's tokenizer, keyed by the SAME id `GET /v1/models`
/// reports (`ModelInfo::id` / `PrefixInfo::model_id`) — `POST /v1/tokenize`
/// never has its own separate id space to keep in sync with that one.
#[derive(Default)]
pub struct TokenizerRegistry {
    entries: std::collections::BTreeMap<String, TokenizerEntry>,
}

impl TokenizerRegistry {
    pub fn new() -> Self {
        Self::default()
    }
    /// `tokenizer` is cleared of truncation AND padding before it's stored
    /// — never the caller's own responsibility. Some heads' tokenizers
    /// (`Lfm2Embedding`'s, `MAX_SEQ_LEN` = 512) carry truncation set for
    /// THEIR OWN inference path, where a silently shortened embedding is
    /// the intended behavior; a clone of that same tokenizer handed to
    /// `POST /v1/tokenize` must not inherit it, or a >512-token request
    /// would silently report only the first 512 tokens as if that were
    /// the whole input — the same class of hazard `Checkpoint::load`
    /// already guards against for the adjudicator's own tokenizer.
    pub fn insert(
        &mut self,
        model_id: impl Into<String>,
        mut tokenizer: tokenizers::Tokenizer,
        tokenizer_hash: impl Into<String>,
    ) {
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(None)
            .expect("clearing truncation (None) never fails validation");
        self.entries
            .insert(model_id.into(), TokenizerEntry { tokenizer, tokenizer_hash: tokenizer_hash.into() });
    }
    pub fn get(&self, model_id: &str) -> Option<&TokenizerEntry> {
        self.entries.get(model_id)
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

const MAX_TEXT_BYTES: usize = 65536;
const MAX_CONTEXT_BYTES: usize = 65536;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenizeRequest {
    /// A model id as `GET /v1/models` lists it, including the adjudicator's
    /// own id (`GET /v1/adjudicator`'s `model_id`) — a separate tokenizer
    /// from every encoder head's.
    pub model: String,
    pub text: String,
    /// When given, also report the tokens `text` occupies immediately AFTER
    /// `context` and whether that reading is stable — see
    /// [`ContextTokenization`].
    #[serde(default)]
    pub context: Option<String>,
}

impl TokenizeRequest {
    pub fn validate(&self) -> Result<(), String> {
        if self.model.is_empty() {
            return Err("model must not be empty".into());
        }
        if self.text.is_empty() {
            return Err("text must not be empty".into());
        }
        if self.text.len() > MAX_TEXT_BYTES {
            return Err(format!("text exceeds {MAX_TEXT_BYTES} bytes"));
        }
        if let Some(context) = &self.context
            && context.len() > MAX_CONTEXT_BYTES
        {
            return Err(format!("context exceeds {MAX_CONTEXT_BYTES} bytes"));
        }
        Ok(())
    }
}

/// One token: its vocabulary id, the tokenizer's own piece spelling (byte-
/// level BPE, so e.g. a leading space reads as `"Ġ..."`, NOT a pretty
/// decoded string — this is the tokenizer's own vocabulary entry, verbatim),
/// and its UTF-8 BYTE span (not codepoints, not UTF-16 units — same
/// convention as `POST /v1/spans`, see `lfm2d/README.md`).
#[derive(Clone, Debug, Serialize)]
pub struct TokenSpan {
    pub id: u32,
    pub token: String,
    pub start: usize,
    pub end: usize,
}

/// The tokens `text` occupies once preceded by `context`, and whether that
/// reading is trustworthy. This is exactly the readability check
/// `LoadedSpec::load` runs on every opinion option before trusting it can
/// be taught-forced at a slot ([`crate::adjudicator::suffix_is_stable`]) —
/// reused here to REPORT the fact instead of refusing on it, since this
/// endpoint is an inspection tool, not a gate.
#[derive(Clone, Debug, Serialize)]
pub struct ContextTokenization {
    /// `text`'s tokens as they actually appear inside `context + text`'s
    /// own tokenization — found by stripping the longest common prefix
    /// with `context` tokenized alone, so a BPE merge right at the
    /// boundary is reflected here even if it moves the split point.
    pub ids: Vec<u32>,
    /// Same tokens, with pieces and BYTE spans into `context + text`
    /// (not into `text` alone — a different offset space than the
    /// top-level `tokens` field, since these tokens only exist once
    /// `context` precedes them).
    pub tokens: Vec<TokenSpan>,
    /// `true` when `text`, tokenized ALONE, is exactly the tail of
    /// `context + text`'s tokenization
    /// ([`crate::adjudicator::suffix_is_stable`]). In the common case (no
    /// merge disturbs `context`'s own tokens) this also means `ids` above
    /// equals `text`'s own standalone encoding, but the two are computed
    /// independently — `ids` by where `context`'s and `context+text`'s
    /// token sequences first diverge, `suffix_stable` by whether `text`'s
    /// tokens land at the very end — so a caller that needs the guarantee
    /// checks `suffix_stable`, not `ids == text`'s own encoding. `false`
    /// means a boundary merge changed `text`'s tokens once `context`
    /// precedes it, which is exactly the hazard that makes teacher-forcing
    /// text at an arbitrary slot unsafe
    /// (`docs/lfm25-adjudicator.md` "The opinion API").
    pub suffix_stable: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct TokenizeResponse {
    pub model: String,
    pub tokenizer_hash: String,
    pub ids: Vec<u32>,
    pub tokens: Vec<TokenSpan>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<ContextTokenization>,
}

/// Encode `text` alone into ids, pieces and byte spans.
pub fn tokenize_text(
    tokenizer: &tokenizers::Tokenizer,
    text: &str,
) -> Result<(Vec<u32>, Vec<TokenSpan>), String> {
    let encoding = tokenizer.encode(text, false).map_err(|e| e.to_string())?;
    let ids = encoding.get_ids().to_vec();
    let tokens = ids
        .iter()
        .zip(encoding.get_tokens())
        .zip(encoding.get_offsets())
        .map(|((&id, piece), &(start, end))| TokenSpan { id, token: piece.clone(), start, end })
        .collect();
    Ok((ids, tokens))
}

/// [`ContextTokenization`] for `text` after `context`.
pub fn tokenize_with_context(
    tokenizer: &tokenizers::Tokenizer,
    context: &str,
    text: &str,
) -> Result<ContextTokenization, String> {
    let context_encoding = tokenizer.encode(context, false).map_err(|e| e.to_string())?;
    let whole_encoding = tokenizer
        .encode(format!("{context}{text}"), false)
        .map_err(|e| e.to_string())?;
    let split =
        crate::opinion::common_prefix_len(&[context_encoding.get_ids().to_vec(), whole_encoding.get_ids().to_vec()]);
    let ids = whole_encoding.get_ids()[split..].to_vec();
    let tokens = whole_encoding.get_ids()[split..]
        .iter()
        .zip(&whole_encoding.get_tokens()[split..])
        .zip(&whole_encoding.get_offsets()[split..])
        .map(|((&id, piece), &(start, end))| TokenSpan { id, token: piece.clone(), start, end })
        .collect();
    let (_, suffix_stable) = crate::adjudicator::suffix_is_stable(tokenizer, context, text)?;
    Ok(ContextTokenization { ids, tokens, suffix_stable })
}

fn bad_request(message: String) -> (StatusCode, Json<ApiError>) {
    (StatusCode::BAD_REQUEST, Json(ApiError::bad_request(message)))
}
fn not_found(message: String) -> (StatusCode, Json<ApiError>) {
    (StatusCode::NOT_FOUND, Json(ApiError::not_found(message)))
}
fn internal(message: String) -> (StatusCode, Json<ApiError>) {
    (StatusCode::INTERNAL_SERVER_ERROR, Json(ApiError::internal(message)))
}

async fn tokenize(
    State(registry): State<Arc<TokenizerRegistry>>,
    crate::server::ValidJson(req): crate::server::ValidJson<TokenizeRequest>,
) -> Result<Json<TokenizeResponse>, (StatusCode, Json<ApiError>)> {
    req.validate().map_err(bad_request)?;
    let entry = registry.get(&req.model).ok_or_else(|| {
        not_found(format!(
            "no loaded model {:?}; GET /v1/models lists every id this instance can tokenize for",
            req.model
        ))
    })?;
    let (ids, tokens) = tokenize_text(&entry.tokenizer, &req.text).map_err(internal)?;
    let context = req
        .context
        .as_ref()
        .map(|context| tokenize_with_context(&entry.tokenizer, context, &req.text))
        .transpose()
        .map_err(internal)?;
    Ok(Json(TokenizeResponse {
        model: req.model,
        tokenizer_hash: entry.tokenizer_hash.clone(),
        ids,
        tokens,
        context,
    }))
}

/// `POST /v1/tokenize`'s own tiny router, merged into the main one by
/// `main.rs`. Deliberately independent of [`crate::server::AppState`] and
/// every `WorkerHandle`/adjudicator `Handle`: the whole point of this route
/// is that it shares no queue with either worker.
pub fn router(registry: Arc<TokenizerRegistry>) -> Router {
    Router::new()
        .route("/v1/tokenize", post(tokenize))
        .with_state(registry)
        .layer(axum::middleware::from_fn(crate::server::telemetry_middleware))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn models_dir() -> PathBuf {
        std::env::var_os("LFM2_MODELS_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().join(".models")
            })
    }

    fn real_tokenizer() -> tokenizers::Tokenizer {
        let path = std::env::var_os("LFM2D_ADJUDICATOR_TOKENIZER")
            .map(PathBuf::from)
            .unwrap_or_else(|| models_dir().join("LFM2.5-8B-A1B/tokenizer.json"));
        assert!(
            path.is_file(),
            "missing tokenizer at {}; set LFM2_MODELS_DIR/LFM2D_ADJUDICATOR_TOKENIZER",
            path.display()
        );
        tokenizers::Tokenizer::from_file(&path).unwrap()
    }

    fn req(model: &str, text: &str) -> TokenizeRequest {
        TokenizeRequest { model: model.into(), text: text.into(), context: None }
    }

    #[test]
    fn empty_model_or_text_is_refused() {
        assert!(req("", "hi").validate().is_err());
        assert!(req("m", "").validate().is_err());
        assert!(req("m", "hi").validate().is_ok());
    }

    #[test]
    fn oversized_text_or_context_is_refused() {
        let mut r = req("m", "x".repeat(MAX_TEXT_BYTES + 1).as_str());
        assert!(r.validate().is_err());
        r.text = "hi".into();
        r.context = Some("x".repeat(MAX_CONTEXT_BYTES + 1));
        assert!(r.validate().is_err());
    }

    #[test]
    fn registry_looks_up_by_the_same_id_v1_models_would_use() {
        let mut reg = TokenizerRegistry::new();
        assert!(reg.is_empty());
        reg.insert("LFM2.5-8B-A1B", real_tokenizer(), "deadbeef");
        assert_eq!(reg.len(), 1);
        assert!(reg.get("LFM2.5-8B-A1B").is_some());
        assert!(reg.get("nope").is_none());
    }

    /// F1 (kaibo review, 2026-09-23): `Lfm2Embedding` loads its tokenizer
    /// with 512-token truncation set FOR ITS OWN inference path
    /// (`MAX_SEQ_LEN`, `src/embedding.rs`) — a clone handed to this
    /// registry must not inherit it, or a >512-token `/v1/tokenize`
    /// request would silently report only the first 512 tokens as if that
    /// were the whole input. Reproduces the exact tokenizer setup
    /// `Lfm2Embedding::from_dir_with` runs (real embedder checkpoint,
    /// same `TruncationParams`), so this is the actual hazard, not a
    /// stand-in for it — only `Lfm2Embedding`'s own model loading is
    /// skipped, since nothing here depends on its weights.
    #[test]
    fn tokenize_never_inherits_an_embedder_style_truncation() {
        let tok_path = models_dir().join("LFM2.5-Embedding-350M/tokenizer.json");
        assert!(
            tok_path.is_file(),
            "missing embedder tokenizer at {}; set LFM2_MODELS_DIR",
            tok_path.display()
        );
        let mut tokenizer = tokenizers::Tokenizer::from_file(&tok_path).unwrap();
        // The exact call `Lfm2Embedding::from_dir_with` makes (`src/embedding.rs`).
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams { max_length: 512, ..Default::default() }))
            .unwrap();

        let mut reg = TokenizerRegistry::new();
        reg.insert("LFM2.5-Embedding-350M", tokenizer, "deadbeef");
        let entry = reg.get("LFM2.5-Embedding-350M").unwrap();

        // A word-repeated text that is comfortably over 512 tokens under
        // this tokenizer, so a silent 512-token truncation would be
        // visible as a wrong `ids.len()`, not just a suspiciously round one.
        let text = "hello world ".repeat(600);
        let (ids, tokens) = tokenize_text(&entry.tokenizer, &text).unwrap();
        assert!(ids.len() > 512, "got only {} tokens; truncation is still active", ids.len());
        assert_eq!(ids.len(), tokens.len());

        // Spans are contiguous and cover the text end to end: token i's
        // end offset is token i+1's start, the first starts at 0, and the
        // last ends at text.len() — proving nothing was silently dropped
        // at either end, not just that the count looks right.
        assert_eq!(tokens.first().unwrap().start, 0);
        for pair in tokens.windows(2) {
            assert_eq!(pair[0].end, pair[1].start, "a gap or overlap between adjacent tokens");
        }
        assert_eq!(tokens.last().unwrap().end, text.len(), "the last token must reach the end of the text");
    }

    #[test]
    fn tokenize_text_reports_ids_pieces_and_byte_offsets_into_text() {
        let tok = real_tokenizer();
        let text = "héllo wörld";
        let (ids, tokens) = tokenize_text(&tok, text).unwrap();
        assert_eq!(ids.len(), tokens.len());
        assert!(!ids.is_empty());
        for t in &tokens {
            // Panics if `start`/`end` don't land on a char boundary -- the
            // property that separates a BYTE offset from a char index on
            // multibyte input, which this text (é, ö) exercises.
            let _ = &text[t.start..t.end];
        }
        // The reconstructed byte spans must cover `text` contiguously start
        // to end, proving these are real byte offsets, not char indices
        // (this text has multibyte characters, so a char-index bug would
        // desync the spans from actual byte content well before the end).
        assert_eq!(tokens.first().unwrap().start, 0);
        assert_eq!(tokens.last().unwrap().end, text.len());
    }

    #[test]
    fn context_tokenization_reports_stable_when_the_prefill_ends_in_a_quote() {
        let tok = real_tokenizer();
        let ctx = tokenize_with_context(&tok, "{\"verdict\": \"", "allow").unwrap();
        assert!(ctx.suffix_stable);
        assert_eq!(ctx.ids, tokenize_text(&tok, "allow").unwrap().0);
    }

    #[test]
    fn context_tokenization_reports_unstable_across_a_space_merge() {
        let tok = real_tokenizer();
        let ctx = tokenize_with_context(&tok, "the verdict is ", "allow").unwrap();
        assert!(!ctx.suffix_stable, "a leading-space BPE merge must be reported, not hidden");
    }
}
