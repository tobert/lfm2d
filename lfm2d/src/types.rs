//! Wire types for the `lfm2d` HTTP API — v1 (see the crate root docs for
//! the full endpoint list and the TEI-compat/audit-trail tension each type
//! resolves).
//!
//! Every map that gets compared as a JSON string in tests
//! ([`ClassifyResult::scores`], [`CascadeWinner::severity_scores`],
//! [`CascadeClause::severity_scores`]) is a [`BTreeMap`], not a
//! `std::collections::HashMap` — `HashMap`'s iteration order is randomized
//! per process (a DoS-hardening default), which would make an
//! exact-JSON-string assertion flaky by construction. A `BTreeMap`
//! serializes its keys in a fixed (alphabetical) order, so the wire output
//! is deterministic run to run.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

// --------------------------------------------------------------- shared

/// `{"inputs": "one string"}` or `{"inputs": ["many", "strings"]}` — the
/// TEI convention both `/embed` and `/predict` accept. Always normalizes to
/// a `Vec<String>` via [`Self::into_vec`]; the response is always an array
/// (batch of 1 for a single-string request), matching the endpoint docs'
/// `→ [[f32,...]]` / `→ [[{label,score},...]]` shape regardless of which
/// input form was used.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum Inputs {
    One(String),
    Many(Vec<String>),
}

impl Inputs {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            Inputs::One(s) => vec![s],
            Inputs::Many(v) => v,
        }
    }
}

/// Which kind of head a loaded model is — `GET /v1/models`'s `kind` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    Adjudicator,
    Embedder,
    Classifier,
    Router,
    /// A per-token BIOES span-detection head (`POST /v1/spans`,
    /// `POST /v1/spans/credentials`) — the PII detector's shape, but not
    /// special-cased to it: any checkpoint `Lfm2TokenClassifier::from_dir`
    /// accepts registers under this kind.
    TokenClassifier,
}

/// One entry in `GET /v1/models`'s response array.
#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub kind: ModelKind,
    pub weight_hash: String,
    /// Only present for a classifier or a token classifier — an
    /// embedder/router has no fixed label set (the router's "labels" are
    /// caller-supplied routes at call time, not a trained output width; see
    /// `src/routing.rs`'s module docs). For a token classifier this is
    /// `entity_types()` (BIOES prefixes stripped, deduped), NOT the raw
    /// `id2label` — the full BIOES-prefixed label set is 161 entries wide
    /// on the PII checkpoint alone and duplicates every entity 5× (B-/I-/O-/
    /// E-/S-); the distinct entity TYPE is the thing a caller of
    /// `/v1/spans` actually cares about naming.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub labels: Option<Vec<String>>,
    pub hidden_size: usize,
}

// --------------------------------------------------------------- /embed

/// Which side of `LFM2.5-Embedding-350M`'s asymmetric (E5-style) pair to
/// embed as — see [`lfm2_encoder::TextKind`]'s docs: the same text
/// under `"query: "` vs `"document: "` embeds at cosine ≈ 0.70, so this is
/// not cosmetic. TEI's own `/embed` has no such parameter (single-purpose
/// embedders don't need one), so this is an ADDITIVE optional field a
/// strict TEI client simply never sets — defaulting to `Document`, the more
/// common "index this text" case, keeps wire compatibility for clients that
/// don't know it exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum EmbedKind {
    #[default]
    Document,
    Query,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EmbedRequest {
    pub inputs: Inputs,
    #[serde(default)]
    pub kind: EmbedKind,
}

// -------------------------------------------------------------- /predict

#[derive(Debug, Clone, Deserialize)]
pub struct PredictRequest {
    pub inputs: Inputs,
}

/// One label's probability — `/predict`'s per-input list covers ALL
/// labels (full softmax), sorted by `score` descending.
#[derive(Debug, Clone, Serialize)]
pub struct LabelScore {
    pub label: String,
    pub score: f32,
}

// ----------------------------------------------------------- /v1/classify

/// Takes the same string-or-array [`Inputs`] as every other endpoint.
///
/// This originally accepted an array ONLY, reasoning that `/v1/classify` is
/// our own contract rather than a TEI-compat shim and so owes no client
/// bug-compatibility. That reasoning was sound and still produced the wrong
/// answer, because `/v1/spans` is equally our own contract and accepts the
/// bare-string form — so the rule wasn't "our endpoints are strict," it was
/// "this one endpoint is different," which is just an inconsistency wearing
/// a justification.
///
/// The cost was real and measured on a live caller: `{"inputs": "ls"}`
/// succeeded against `/predict` and `/v1/spans` and returned 400 from
/// `/v1/classify`, in the same session where the same field name had
/// already been fumbled once. An API is allowed to be strict, but not
/// unpredictably strict — a caller should learn the request shape once.
///
/// The response is unchanged: always an array, one entry per input, batch
/// of one for a single string.
#[derive(Debug, Clone, Deserialize)]
pub struct ClassifyRequest {
    pub inputs: Inputs,
}

/// One input's full result: the full softmax (`scores`, every label), the
/// argmax (`top`), and the audit pair (`model_id`, `weight_hash`) — see the
/// crate root docs on why `/v1/classify` carries these in-body while
/// `/embed`/`/predict` carry them as response headers instead.
#[derive(Debug, Clone, Serialize)]
pub struct ClassifyResult {
    pub scores: BTreeMap<String, f32>,
    pub top: String,
    pub model_id: String,
    pub weight_hash: String,
}

// -------------------------------------------------------------- /v1/route

/// Singular `input`, not `inputs` — one prompt scored against many routes
/// in one forward pass (matching [`lfm2_encoder::Lfm2SequenceRouter::route_cosines`]).
#[derive(Debug, Clone, Deserialize)]
pub struct RouteRequest {
    pub input: String,
    pub routes: Vec<String>,
}

/// One route's RAW cosine — no softmax probability anywhere on this
/// endpoint. The router's softmax is route-count arithmetic and carries no
/// confidence information (see `src/routing.rs`'s module docs, "the
/// softmax is saturated"); exposing it here would just be a footgun with a
/// plausible-looking field name.
#[derive(Debug, Clone, Serialize)]
pub struct RouteScore {
    pub route: String,
    pub cosine: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct RouteResponse {
    pub model_id: String,
    pub weight_hash: String,
    pub routes: Vec<RouteScore>,
}

// ------------------------------------------------------------ /v1/cascade

/// `{"clauses": [...]}` only — no `routes`, no `severe_labels` in the
/// request body. Both are fixed server-side configuration (`--cascade-route`
/// / `--cascade-severe-label`), not per-request input; see the crate root
/// docs' "cascade configuration is server-side" section for why.
#[derive(Debug, Clone, Deserialize)]
pub struct CascadeRequest {
    pub clauses: Vec<String>,
}

/// The winning (highest-severity) clause. NO `top_severity` field here —
/// unlike [`CascadeClause`] — because "winner" already names which clause
/// won; adding a second "top" concept at this level would just be noise.
#[derive(Debug, Clone, Serialize)]
pub struct CascadeWinner {
    pub index: usize,
    pub clause: String,
    pub severity_scores: BTreeMap<String, f32>,
}

/// The winning clause's argmax route — the cascade's single "route this
/// statement to X" answer.
#[derive(Debug, Clone, Serialize)]
pub struct CascadeLane {
    pub route: String,
    pub cosine: f32,
}

/// One clause's full per-label breakdown, plus which single label scored
/// highest (`top_severity`) — the ranking SUM used to pick the winner
/// (`severity_score` in [`lfm2_encoder::ClauseVerdict`], the caller's
/// configured severe-label set summed) is deliberately not repeated here;
/// see the crate root docs.
#[derive(Debug, Clone, Serialize)]
pub struct CascadeClause {
    pub index: usize,
    pub clause: String,
    pub severity_scores: BTreeMap<String, f32>,
    pub top_severity: String,
}

/// One model's audit identity — `/v1/cascade`'s `models` array carries one
/// entry per head involved (classifier, router).
#[derive(Debug, Clone, Serialize)]
pub struct CascadeModelRef {
    pub model_id: String,
    pub weight_hash: String,
}

/// Moderations-FLAVORED but with NO `flagged` boolean and NO threshold
/// anywhere — see the crate root docs and `src/cascade.rs`'s module docs on
/// why: no global severity cutoff exists by measurement (benign max 0.3415
/// vs data-critical min 0.3440), so a caller ranks clauses WITHIN a
/// statement rather than gating on an absolute number this API would
/// otherwise seem to bless.
#[derive(Debug, Clone, Serialize)]
pub struct CascadeResponse {
    pub winner: CascadeWinner,
    pub lane: CascadeLane,
    pub clauses: Vec<CascadeClause>,
    pub models: Vec<CascadeModelRef>,
}

// -------------------------------------------------------------- /v1/spans

/// `{"inputs": str | [str]}` (TEI-style, same [`Inputs`] normalization as
/// `/embed`/`/predict`), plus an OPTIONAL `"model"` to pick which loaded
/// token-classification head answers this call. Required only when 2+
/// `--token-classifier-dir` heads are loaded — see `server::spans`'s doc
/// comment and the crate root docs' "N token heads" section for the
/// disambiguation rule.
#[derive(Debug, Clone, Deserialize)]
pub struct SpansRequest {
    pub inputs: Inputs,
    #[serde(default)]
    pub model: Option<String>,
}

/// One detected span, on the wire. `POST /v1/spans` and
/// `POST /v1/spans/credentials` respond with `Vec<Vec<SpanResult>>` — a
/// bare array of arrays (TEI-shaped, no wrapper object; audit pair travels
/// as `X-Model-Id`/`X-Model-Weight-Hash` headers same as `/embed`/
/// `/predict` — see the crate root docs' "two response conventions").
///
/// # NEVER add the matched text to this type
///
/// However tempting a `word`/`quote`/`text` field looks (Presidio and GCP
/// DLP both ship one, and it IS convenient for a caller who wants to log
/// what matched) — do not add it, not even behind an opt-in flag. The
/// caller already has the full input text it just sent us; all this field
/// could ever do is duplicate a piece of it verbatim into every response
/// this endpoint sends back, and this endpoint's entire reason to exist is
/// finding credentials — echoing one back is a second place for it to leak
/// into a log, a cache, a debugger's "pretty-print this JSON" pane, etc.
/// GCP DLP ships an opt-in "no matched text" mode as a special case; this
/// API is stricter on purpose and makes that the ONLY mode.
#[derive(Debug, Clone, Serialize)]
pub struct SpanResult {
    /// Byte offset of the span's first byte into the UTF-8 input string
    /// that was sent — NOT a codepoint/char index, NOT a UTF-16 code-unit
    /// index. `Lfm2TokenClassifier::Span` (the library type this is built
    /// from) documents itself as byte offsets and this type passes that
    /// value through unmodified — see `engine_real.rs`'s `spans_outcome`.
    /// A Rust caller (kaibo, the first consumer) can slice
    /// `&text[start..end]` directly. A Python/JS caller must NOT index its
    /// own string with these numbers directly — Python `str` and JS
    /// strings are codepoint/UTF-16 indexed, not byte-indexed — it must
    /// re-encode to UTF-8 bytes (or operate on the raw bytes it already
    /// sent) before slicing.
    pub start: usize,
    /// Byte offset one past the span's last byte — same units as `start`.
    pub end: usize,
    /// Entity type with the BIOES prefix stripped, e.g. `credential.api_key`
    /// — matches [`lfm2_encoder::Span::label`] and the strings
    /// `entity_types()` / `GET /v1/models`'s `labels` enumerate.
    pub entity: String,
    /// Confidence in `[0, 1]`: the **minimum** softmax probability across
    /// the span's tokens, passed straight through from
    /// [`lfm2_encoder::Span::score`].
    ///
    /// Minimum rather than mean because a span is a CONJUNCTION of
    /// per-token decisions — it is wrong if any one token is wrong — so a
    /// single coin-flip token inside an otherwise confident credential is
    /// the signal, and averaging would hide it. Amy's ruling, 2026-08-11.
    ///
    /// **These numbers read systematically LOWER than other PII services'**
    /// (Hugging Face's grouped-entity pipeline averages; Presidio reports a
    /// recognizer's own confidence). Do not compare them across tools, and
    /// do not "fix" them upward here.
    ///
    /// As everywhere else in this API: a ranking signal, not a calibrated
    /// absolute. Nothing in lfm2d thresholds it, and a caller that invents
    /// a global cutoff is repeating the mistake measured in
    /// `rank-within-dont-threshold-across`.
    pub score: f32,
}

// ----------------------------------------------------------------- errors

/// `{"error": {"message", "type"}}` — every non-2xx response body.
#[derive(Debug, Clone, Serialize)]
pub struct ApiError {
    pub error: ApiErrorBody,
}

#[derive(Debug, Clone, Serialize)]
pub struct ApiErrorBody {
    pub message: String,
    #[serde(rename = "type")]
    pub kind: String,
}

impl ApiError {
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self { error: ApiErrorBody { message: message.into(), kind: "bad_request".to_string() } }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self { error: ApiErrorBody { message: message.into(), kind: "internal".to_string() } }
    }
}
