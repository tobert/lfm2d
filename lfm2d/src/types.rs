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

// -------------------------------------------------------- /v1/adjudicate distributions
//
// `docs/field-requests.md` decision 5: every model-filled field carries its
// distribution, in LOG-SPACE (values saturate — 0.993 vs 0.927 is typical,
// live winner margins have been seen at 5e-9, and probability floats smear
// an ordering logprobs keep), and a named token set's mass is reported RAW
// — never renormalized over the set. Renormalizing a near-zero tail turned
// an "unasked" question into a confident-looking 85/733 "accuracy" that
// measured nothing. There is deliberately no renormalized-distribution type
// anywhere in this module: the only way to see mass inside a set is
// [`SetMass`], and it always carries the raw (denominator = full
// vocabulary) value, never a value renormalized over just that set.

/// One named request for per-generated-token distribution data on
/// `POST /v1/adjudicate`. Optional and additive: a request that omits
/// `distributions` entirely gets today's `AdjudicateResponse` shape,
/// byte-identical (see [`crate::adjudicator::AdjudicateResponse::distributions`]).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DistributionRequest {
    /// How many top logprobs to report per generated-token step, sorted
    /// descending, ties broken by ascending token id (matching
    /// [`candle_nn::sampling::GreedySampler`]'s own tie rule). `0` is a
    /// valid request meaning "no top-k, only `token_sets`".
    #[serde(default = "default_distribution_top_k")]
    pub top_k: usize,
    /// Named sets of token ids. Each step's response reports the RAW mass
    /// (full-vocabulary denominator) those ids hold, never renormalized
    /// over the set — see the module note above. Every set must be
    /// non-empty and its ids must not repeat (a repeat would silently
    /// double-count mass, which is the data-corruption case the daemon
    /// refuses rather than papering over).
    #[serde(default)]
    pub token_sets: BTreeMap<String, Vec<u32>>,
}

fn default_distribution_top_k() -> usize {
    5
}

/// Hard cap on `top_k`. Not a soft default — a request above this is
/// rejected, not silently clamped.
pub const MAX_DISTRIBUTION_TOP_K: usize = 20;

impl DistributionRequest {
    /// Structural checks available without a loaded model (bounds, empty
    /// sets, duplicate ids). Vocabulary-bound checks need [`Self::validate_vocab`].
    pub fn validate(&self) -> Result<(), String> {
        if self.top_k > MAX_DISTRIBUTION_TOP_K {
            return Err(format!(
                "distributions.top_k must be 0..={MAX_DISTRIBUTION_TOP_K}"
            ));
        }
        for (name, ids) in &self.token_sets {
            if name.trim().is_empty() {
                return Err("distributions.token_sets name must not be empty".into());
            }
            if ids.is_empty() {
                return Err(format!("distributions.token_sets['{name}'] must not be empty"));
            }
            let unique: std::collections::BTreeSet<_> = ids.iter().collect();
            if unique.len() != ids.len() {
                return Err(format!(
                    "distributions.token_sets['{name}'] has a duplicate token id"
                ));
            }
        }
        Ok(())
    }

    /// Every token id named anywhere in this request must be a real row in
    /// the loaded model's vocabulary. Needs the model, so it runs
    /// separately from [`Self::validate`], after a model is in hand.
    pub fn validate_vocab(&self, vocab_size: usize) -> Result<(), String> {
        for (name, ids) in &self.token_sets {
            if ids.iter().any(|&id| id as usize >= vocab_size) {
                return Err(format!(
                    "distributions.token_sets['{name}'] has a token id outside the model vocabulary"
                ));
            }
        }
        Ok(())
    }
}

/// One vocabulary row's logprob at some step — always `<= 0`, always the
/// RAW log-softmax value (full-vocabulary denominator).
#[derive(Debug, Clone, Serialize)]
pub struct TokenLogprob {
    pub token: u32,
    /// `None` for a GGUF vocabulary padding row with no tokenizer text
    /// (the checkpoint pads rows beyond the tokenizer's usable vocabulary;
    /// see `Adjudicator::load`'s vocab-agreement check) — reported honestly
    /// rather than guessed at.
    pub text: Option<String>,
    pub logprob: f32,
}

/// A named token set's RAW mass at one generated-token step: how much of
/// the FULL vocabulary's probability landed on this set, without
/// renormalizing over just the set. Low mass means the model was never
/// steered toward this vocabulary — "unasked", not "wrong" — see the module
/// note above; nothing in this daemon ever turns this into a
/// renormalized-over-the-set value.
#[derive(Debug, Clone, Serialize)]
pub struct SetMass {
    /// `log(sum(exp(logprob_i)))` over the set's raw per-token logprobs.
    /// Always `<= 0`.
    pub logprob: f32,
    /// `exp(logprob)`, for convenience. Computed from `logprob`, not a
    /// separately-derived value — see [`step_distribution`]'s tests, which
    /// assert this rather than leaving it a silently-unpopulated `0.0`
    /// (the hazard a prior implementation shipped).
    pub prob: f32,
}

/// One generated token's full distribution context: the token actually
/// sampled, the top-k alternatives it beat, and the raw mass any requested
/// named token sets held at that position. `AdjudicateResponse::distributions`
/// carries one of these per generated token (including a trailing eos, if
/// any) in generation order.
#[derive(Debug, Clone, Serialize)]
pub struct StepDistribution {
    pub token: u32,
    pub text: String,
    /// The sampled token's own raw logprob. `<= 0`.
    pub logprob: f32,
    pub top_logprobs: Vec<TokenLogprob>,
    pub set_mass: BTreeMap<String, SetMass>,
}

fn logsumexp(values: impl Iterator<Item = f32>) -> f32 {
    let values: Vec<f32> = values.collect();
    let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        // Empty (validated against elsewhere) or every value is -inf.
        return max;
    }
    max + values.iter().map(|&v| (v - max).exp()).sum::<f32>().ln()
}

/// Indices of the `k` largest values in `log_probs`, strictly descending,
/// ties broken by ascending index — matching
/// [`candle_nn::sampling::GreedySampler::sample`]'s own tie rule so a
/// reported top-1 always agrees with what greedy sampling would pick.
/// `O(n)` selection plus `O(k log k)` to order the winners, not a full sort.
fn top_k_indices(log_probs: &[f32], k: usize) -> Vec<usize> {
    let k = k.min(log_probs.len());
    if k == 0 {
        return Vec::new();
    }
    let mut idx: Vec<usize> = (0..log_probs.len()).collect();
    let cmp = |log_probs: &[f32], &a: &usize, &b: &usize| {
        log_probs[b]
            .partial_cmp(&log_probs[a])
            .expect("non-finite logit reached top_k_indices; the sampler should have rejected it")
            .then(a.cmp(&b))
    };
    idx.select_nth_unstable_by(k - 1, |a, b| cmp(log_probs, a, b));
    idx.truncate(k);
    idx.sort_unstable_by(|a, b| cmp(log_probs, a, b));
    idx
}

/// Compute one step's [`StepDistribution`] from raw model logits (NOT
/// pre-softmaxed) over the whole vocabulary. `token_text` resolves a
/// vocabulary id to its tokenizer spelling (`None` for an unused GGUF
/// padding row) — injected so this stays pure and testable without a
/// tokenizer or a loaded model.
///
/// Errors rather than silently misreporting: non-finite logits, a sampled
/// token or a `token_sets` id outside `logits`' length, an empty or
/// duplicate-id token set, or a sampled token with no vocabulary text are
/// all rejected. Crashing the request is preferred to shipping a number
/// that looks plausible and is not.
pub fn step_distribution(
    logits: &[f32],
    sampled_token: u32,
    top_k: usize,
    token_sets: &BTreeMap<String, Vec<u32>>,
    token_text: impl Fn(u32) -> Option<String>,
) -> Result<StepDistribution, String> {
    if logits.is_empty() {
        return Err("step_distribution: empty logits".into());
    }
    let sampled = sampled_token as usize;
    if sampled >= logits.len() {
        return Err(format!(
            "step_distribution: sampled token {sampled_token} outside vocabulary of {}",
            logits.len()
        ));
    }
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if !max_logit.is_finite() {
        return Err("step_distribution: non-finite logits".into());
    }
    let log_z = max_logit + logits.iter().map(|&l| (l - max_logit).exp()).sum::<f32>().ln();
    let log_probs: Vec<f32> = logits.iter().map(|&l| l - log_z).collect();

    let text = token_text(sampled_token).ok_or_else(|| {
        format!("step_distribution: sampled token {sampled_token} has no vocabulary text")
    })?;

    let top_logprobs = top_k_indices(&log_probs, top_k)
        .into_iter()
        .map(|i| TokenLogprob {
            token: i as u32,
            text: token_text(i as u32),
            logprob: log_probs[i],
        })
        .collect();

    let mut set_mass = BTreeMap::new();
    for (name, ids) in token_sets {
        if ids.is_empty() {
            return Err(format!("step_distribution: token set '{name}' is empty"));
        }
        let mut seen = std::collections::BTreeSet::new();
        for &id in ids {
            if id as usize >= log_probs.len() {
                return Err(format!(
                    "step_distribution: token set '{name}' has token id {id} outside vocabulary"
                ));
            }
            if !seen.insert(id) {
                return Err(format!(
                    "step_distribution: token set '{name}' has duplicate token id {id}"
                ));
            }
        }
        let logprob = logsumexp(ids.iter().map(|&id| log_probs[id as usize]));
        set_mass.insert(name.clone(), SetMass { logprob, prob: logprob.exp() });
    }

    Ok(StepDistribution {
        token: sampled_token,
        text,
        logprob: log_probs[sampled],
        top_logprobs,
        set_mass,
    })
}

#[cfg(test)]
mod distribution_tests {
    use super::*;

    fn text(id: u32) -> Option<String> {
        Some(format!("<{id}>"))
    }

    #[test]
    fn logprobs_are_logprobs_and_exponentiate_to_one() {
        let logits = [2.0_f32, -1.0, 0.5, 3.0, -3.0, 1.0];
        let step = step_distribution(&logits, 3, logits.len(), &BTreeMap::new(), text).unwrap();
        assert!(step.logprob <= 0.0);
        for t in &step.top_logprobs {
            assert!(t.logprob <= 0.0, "{t:?}");
        }
        assert_eq!(step.top_logprobs.len(), logits.len());
        let sum: f32 = step.top_logprobs.iter().map(|t| t.logprob.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-4, "sum={sum}");
    }

    #[test]
    fn top_k_is_actually_top_k_in_descending_order() {
        let logits = [1.0_f32, 5.0, 3.0, 3.0, -2.0, 4.0];
        let step = step_distribution(&logits, 1, 3, &BTreeMap::new(), text).unwrap();
        let ids: Vec<u32> = step.top_logprobs.iter().map(|t| t.token).collect();
        // index 1 (5.0) > index 5 (4.0) > tie between 2 and 3 (3.0), smaller id first.
        assert_eq!(ids, vec![1, 5, 2]);
        for w in step.top_logprobs.windows(2) {
            assert!(w[0].logprob >= w[1].logprob, "{:?} not descending", step.top_logprobs);
        }
    }

    #[test]
    fn set_mass_is_raw_not_renormalized() {
        // Two "label" tokens hold almost none of the mass; a third,
        // unrelated token dominates. Renormalizing over {label_a, label_b}
        // would manufacture a confident ~fifty-fifty split out of noise;
        // the raw mass must instead read as ~nothing.
        let logits = [10.0_f32, 0.0, 0.1];
        let mut sets = BTreeMap::new();
        sets.insert("labels".to_string(), vec![1u32, 2]);
        let step = step_distribution(&logits, 0, 0, &sets, text).unwrap();
        let raw = &step.set_mass["labels"];
        assert!(raw.prob < 0.001, "raw mass should be tiny, got {}", raw.prob);
        // What a (wrong) renormalization over just {1, 2} would have said:
        let renormalized_prob_of_1 = 1.0 / (1.0 + (0.1f32 - 0.0).exp());
        assert!(
            (raw.prob - renormalized_prob_of_1).abs() > 0.1,
            "raw mass must not resemble the renormalized-over-the-set answer"
        );
        assert!((raw.prob - raw.logprob.exp()).abs() < 1e-6, "prob must equal exp(logprob)");
    }

    #[test]
    fn determinism_same_input_same_output() {
        let logits = [0.3_f32, -1.2, 4.4, 2.2, -0.6, 1.1, 3.3];
        let mut sets = BTreeMap::new();
        sets.insert("s".to_string(), vec![0u32, 2, 4]);
        let a = step_distribution(&logits, 2, 4, &sets, text).unwrap();
        let b = step_distribution(&logits, 2, 4, &sets, text).unwrap();
        assert_eq!(a.logprob, b.logprob);
        assert_eq!(
            a.top_logprobs.iter().map(|t| (t.token, t.logprob)).collect::<Vec<_>>(),
            b.top_logprobs.iter().map(|t| (t.token, t.logprob)).collect::<Vec<_>>()
        );
        assert_eq!(a.set_mass["s"].logprob, b.set_mass["s"].logprob);
    }

    #[test]
    fn empty_or_duplicate_token_set_is_rejected_not_silently_corrupted() {
        let logits = [1.0_f32, 2.0, 3.0];
        let mut empty = BTreeMap::new();
        empty.insert("x".to_string(), Vec::<u32>::new());
        assert!(step_distribution(&logits, 0, 1, &empty, text).is_err());

        let mut dup = BTreeMap::new();
        dup.insert("x".to_string(), vec![0u32, 0]);
        assert!(step_distribution(&logits, 0, 1, &dup, text).is_err());
    }

    #[test]
    fn sampled_or_set_token_outside_vocabulary_is_rejected() {
        let logits = [1.0_f32, 2.0];
        assert!(step_distribution(&logits, 5, 1, &BTreeMap::new(), text).is_err());
        let mut sets = BTreeMap::new();
        sets.insert("x".to_string(), vec![5u32]);
        assert!(step_distribution(&logits, 0, 1, &sets, text).is_err());
    }

    #[test]
    fn distribution_request_validate_rejects_bad_shapes() {
        let mut too_big = DistributionRequest { top_k: MAX_DISTRIBUTION_TOP_K + 1, token_sets: BTreeMap::new() };
        assert!(too_big.validate().is_err());
        too_big.top_k = MAX_DISTRIBUTION_TOP_K;
        assert!(too_big.validate().is_ok());

        let mut empty_set = BTreeMap::new();
        empty_set.insert("x".to_string(), Vec::<u32>::new());
        let req = DistributionRequest { top_k: 5, token_sets: empty_set };
        assert!(req.validate().is_err());

        let mut dup_ids = BTreeMap::new();
        dup_ids.insert("x".to_string(), vec![1u32, 1]);
        let req = DistributionRequest { top_k: 5, token_sets: dup_ids };
        assert!(req.validate().is_err());

        let mut ok = BTreeMap::new();
        ok.insert("x".to_string(), vec![1u32, 2]);
        let req = DistributionRequest { top_k: 5, token_sets: ok };
        assert!(req.validate().is_ok());
        assert!(req.validate_vocab(2).is_err(), "id 2 is outside a vocab of size 2");
        assert!(req.validate_vocab(3).is_ok(), "id 2 is the last valid row in a vocab of size 3");
    }
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
