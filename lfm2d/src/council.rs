//! The council contract's pure layer: what `/council/v1` means before any
//! engine runs.
//!
//! The contract is kaijutsu's `docs/council-api.md` (and its OpenAPI file),
//! served by the megakernel beside its `/mk/v1` and here. This module holds
//! the parts that are a function of the request alone, so each is a unit
//! test and none needs weights:
//!
//! - context ids (client UUIDs) and the `PUT` diff that decides what a
//!   context update keeps and what it feeds ([`plan_put`]);
//! - the spec's validation, canonical JSON (RFC 8785) and id ([`spec_id`]);
//! - control text: found, reported, and split so an encoder can tokenize it
//!   as ordinary text ([`control_hits`], [`control_pieces`]);
//! - the numbers in an answer, with the contract's exact relations
//!   ([`read_numbers`], [`pooled_numbers`]).
//!
//! Vocabulary: a *context* is the client's mutable record under its UUID; a
//! *snapshot* is held model state for one build of its prefix, addressed by
//! the server (here, the sha256 ids of `state_store`). The two never share
//! an id.
use serde::ser::{SerializeMap, Serializer};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::chat::CONTROL_MARKERS;
use crate::hash::sha256_hex_bytes;
use crate::pool::{Method, PoolSettings, Weights};

// ---------------------------------------------------------------- errors

/// The contract's error body, `{"error": {"type", "message", "param"?, "head"?}}`,
/// and the status it travels under.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CouncilError {
    #[serde(skip)]
    pub status: u16,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub head: Option<String>,
}

impl CouncilError {
    fn new(status: u16, kind: &'static str, message: impl Into<String>) -> Self {
        Self { status, kind, message: message.into(), param: None, head: None }
    }
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self::new(400, "invalid_request_error", message)
    }
    pub fn param(mut self, param: impl Into<String>) -> Self {
        self.param = Some(param.into());
        self
    }
    pub fn head(mut self, head: impl Into<String>) -> Self {
        self.head = Some(head.into());
        self
    }
    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new(404, "not_found_error", message)
    }
    pub fn too_large(message: impl Into<String>, param: &str) -> Self {
        Self::new(413, "request_too_large", message).param(param)
    }
    /// `If-Match` named a head the context does not have.
    pub fn precondition_failed(current_head: &str) -> Self {
        Self::new(412, "precondition_failed", "the context's head is not the one If-Match named")
            .head(current_head)
    }
    /// A decision's `at` names a snapshot the server no longer holds.
    pub fn snapshot_gone(current_head: &str) -> Self {
        Self::new(409, "conflict_error", "the snapshot is no longer held").head(current_head)
    }
    pub fn to_body(&self) -> Value {
        serde_json::json!({ "error": self })
    }
}

type Result<T> = std::result::Result<T, CouncilError>;

// -------------------------------------------------------------- ordered map

/// An object that keeps its members in the order given. `serde_json::Value`
/// sorts keys, and an option's place in its question is part of the question
/// (ties go to the earlier option), so answers serialize through this.
#[derive(Clone, Debug, PartialEq)]
pub struct Ordered<T>(pub Vec<(String, T)>);

impl<T: Serialize> Serialize for Ordered<T> {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        let mut map = s.serialize_map(Some(self.0.len()))?;
        for (k, v) in &self.0 {
            map.serialize_entry(k, v)?;
        }
        map.end()
    }
}

// -------------------------------------------------------------- context ids

/// A context id is a UUID the client chose: 36 characters, lowercase hex in
/// 8-4-4-4-12 groups. The server never mints one and never rewrites one, so
/// a spelling it would have to normalise (uppercase, braces, no hyphens) is
/// refused rather than quietly mapped to another key.
pub fn parse_context_id(id: &str) -> Result<&str> {
    let ok = id.len() == 36
        && id.bytes().enumerate().all(|(i, b)| match i {
            8 | 13 | 18 | 23 => b == b'-',
            _ => matches!(b, b'0'..=b'9' | b'a'..=b'f'),
        });
    if ok {
        Ok(id)
    } else {
        Err(CouncilError::bad_request(format!(
            "{id:?} is not a context id: a lowercase UUID, 8-4-4-4-12 hex digits"
        ))
        .param("id"))
    }
}

// ------------------------------------------------------------------ limits

/// What this server holds to; `GET /council/v1/identity` reports it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Limits {
    pub contexts_per_decision: usize,
    pub questions_per_spec: usize,
    pub choice_options: usize,
    pub text_tokens: usize,
    pub state_bytes: usize,
    pub turns_per_context: usize,
    pub default_timeout_ms: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            contexts_per_decision: crate::opinion_api::MAX_CONTEXTS,
            questions_per_spec: 16,
            choice_options: 255,
            text_tokens: 256,
            state_bytes: crate::opinion_api::MAX_STATE_BYTES,
            turns_per_context: 512,
            default_timeout_ms: 30_000,
        }
    }
}

// ------------------------------------------------------------------ contexts

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

/// One turn of a context. `snap` marks a boundary: a turn the client expects
/// to keep while later turns change. `reasoning` is an assistant turn's own
/// thinking; it is content, so it is part of the diff and of every id.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Turn {
    pub role: Role,
    pub content: String,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub snap: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
}

impl Turn {
    /// Absent and empty reasoning render the same empty thinking region, so
    /// they are the same content: one turn, one set of snapshot ids.
    fn same_content(&self, other: &Self) -> bool {
        self.role == other.role
            && self.content == other.content
            && self.snap == other.snap
            && self.reasoning.as_deref().unwrap_or("") == other.reasoning.as_deref().unwrap_or("")
    }
}

/// `PUT /council/v1/contexts/{id}`'s body: the whole context.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContextBody {
    #[serde(default)]
    pub system: Option<String>,
    #[serde(default)]
    pub turns: Vec<Turn>,
    #[serde(default)]
    pub pin: Option<bool>,
    #[serde(default)]
    pub dry_run: bool,
    /// Held specs whose layers are rebuilt before the answer: up to 16, distinct.
    #[serde(default)]
    pub warm: Vec<String>,
}

impl ContextBody {
    pub fn validate(&self, limits: &Limits) -> Result<()> {
        if self.system.is_none() && self.turns.is_empty() {
            return Err(CouncilError::bad_request("a context needs a system message or a turn"));
        }
        if self.turns.len() > limits.turns_per_context {
            return Err(CouncilError::too_large(
                format!("{} turns; this server holds at most {}", self.turns.len(), limits.turns_per_context),
                "turns",
            ));
        }
        for (n, turn) in self.turns.iter().enumerate() {
            if turn.reasoning.is_some() && turn.role != Role::Assistant {
                return Err(CouncilError::bad_request(format!(
                    "turn {n} is a user turn; only an assistant turn carries reasoning"
                ))
                .param(format!("turns[{n}].reasoning")));
            }
            if turn.content.is_empty() && turn.reasoning.as_deref().unwrap_or("").is_empty() {
                return Err(CouncilError::bad_request(format!("turn {n} is empty"))
                    .param(format!("turns[{n}].content")));
            }
        }
        if self.warm.len() > 16 {
            return Err(CouncilError::bad_request("warm names at most 16 specs").param("warm"));
        }
        let mut seen = std::collections::BTreeSet::new();
        for spec in &self.warm {
            if !seen.insert(spec) {
                return Err(CouncilError::bad_request(format!("warm repeats {spec:?}")).param("warm"));
            }
        }
        Ok(())
    }
}

/// What a `PUT` keeps and what it feeds, in turns. `kept_turns` is how many
/// leading turns the held build already covers; the system message is part
/// of every boundary, so `kept_system` is false when it changed (and then
/// nothing is kept). `fed_from` is the first turn the build runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PutPlan {
    pub kept_system: bool,
    pub kept_turns: usize,
}

/// The first difference between a held context and a `PUT`, snapped back to
/// the last declared boundary: the system message or a turn marked `snap`.
///
/// An update extends only from a declared boundary, never from an unmarked
/// head, so a snapshot's bits are a function of the tokens and the `snap`
/// flags alone: the same content gives the same snapshots whether it
/// arrived one turn at a time or in one `PUT` after a restart. A turn whose
/// `snap` flag changed is an edit at that turn.
pub fn plan_put(held: Option<(&Option<String>, &[Turn])>, new: &ContextBody) -> PutPlan {
    let Some((system, turns)) = held else {
        return PutPlan { kept_system: false, kept_turns: 0 };
    };
    if *system != new.system {
        return PutPlan { kept_system: false, kept_turns: 0 };
    }
    let common = turns
        .iter()
        .zip(&new.turns)
        .take_while(|(a, b)| a.same_content(b))
        .count();
    // The held build has a boundary after turn j only when turn j is marked
    // snap (the same in both: it is inside the common prefix). A build
    // always has the system boundary.
    let kept_turns = (0..common).rev().find(|&j| new.turns[j].snap).map_or(0, |j| j + 1);
    PutPlan { kept_system: true, kept_turns }
}

// --------------------------------------------------------------------- specs

/// A spec question. The tag is the contract's `type`.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "lowercase", deny_unknown_fields)]
pub enum SpecQuestion {
    Choice { id: String, instructions: Value, criteria: Vec<ChoiceOption> },
    Score { id: String, instructions: Value, criteria: Vec<String> },
    Noul {
        id: String,
        instructions: Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        criteria: Option<Value>,
    },
    Text { id: String, instructions: Value, max_tokens: u64 },
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ChoiceOption {
    pub option: String,
    #[serde(default)]
    pub means: String,
}

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CouncilSpec {
    pub name: String,
    pub instructions: String,
    pub input_label: String,
    pub questions: Vec<SpecQuestion>,
}

impl SpecQuestion {
    pub fn id(&self) -> &str {
        match self {
            Self::Choice { id, .. } | Self::Score { id, .. } | Self::Noul { id, .. } | Self::Text { id, .. } => id,
        }
    }
    pub fn is_text(&self) -> bool {
        matches!(self, Self::Text { .. })
    }
    /// The answer names, in the question's order: options, levels as their
    /// level numbers, or `yes` then `no`. A text question has none.
    pub fn labels(&self) -> Vec<String> {
        match self {
            Self::Choice { criteria, .. } => criteria.iter().map(|c| c.option.clone()).collect(),
            Self::Score { criteria, .. } => (0..criteria.len()).map(|n| n.to_string()).collect(),
            Self::Noul { .. } => vec!["yes".into(), "no".into()],
            Self::Text { .. } => Vec::new(),
        }
    }
}

fn question_id_ok(id: &str) -> bool {
    let mut chars = id.chars();
    id.len() <= 64
        && matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Every number in a spec is an integer below 2^53: writing any other
/// canonically needs ES6 number formatting, so it is refused rather than
/// hashed under a guess.
fn check_integers(value: &Value, at: &str) -> Result<()> {
    match value {
        Value::Number(n) => match n.as_i64().map(i64::unsigned_abs).or_else(|| n.as_u64()) {
            Some(m) if m < (1u64 << 53) => Ok(()),
            _ => Err(CouncilError::bad_request(format!(
                "{at}: spec numbers are integers below 2^53 (got {n})"
            ))),
        },
        Value::Array(items) => items.iter().enumerate().try_for_each(|(i, v)| check_integers(v, &format!("{at}[{i}]"))),
        Value::Object(map) => map.iter().try_for_each(|(k, v)| check_integers(v, &format!("{at}.{k}"))),
        _ => Ok(()),
    }
}

impl CouncilSpec {
    /// Parse and check a spec body. `Err` is a `400` naming the problem.
    pub fn parse(body: &Value, limits: &Limits) -> Result<Self> {
        check_integers(body, "spec")?;
        let spec: Self = serde_json::from_value(body.clone())
            .map_err(|e| CouncilError::bad_request(format!("not a spec: {e}")))?;
        spec.validate(limits)?;
        Ok(spec)
    }

    fn validate(&self, limits: &Limits) -> Result<()> {
        let bad = |m: String| Err(CouncilError::bad_request(m));
        if self.name.is_empty() || self.name.len() > 64 {
            return bad("name is 1 to 64 bytes".into());
        }
        // The label heads the state in the case's user turn: one line, no
        // colon (the renderer adds its own), no control text.
        if self.input_label.is_empty()
            || self.input_label.len() > 64
            || self.input_label.contains([':', '\n', '\r'])
            || crate::chat::control_marker_in(&self.input_label).is_some()
        {
            return bad("input_label is one line of 1 to 64 bytes: no colon, no control text".into());
        }
        if self.questions.is_empty() || self.questions.len() > limits.questions_per_spec {
            return bad(format!("a spec has 1 to {} questions", limits.questions_per_spec));
        }
        let mut seen = std::collections::BTreeSet::new();
        for q in &self.questions {
            if !question_id_ok(q.id()) {
                return bad(format!("question id {:?} is not [A-Za-z_][A-Za-z0-9_]{{0,63}}", q.id()));
            }
            if !seen.insert(q.id()) {
                return bad(format!("question id {:?} repeats", q.id()));
            }
            match q {
                SpecQuestion::Choice { criteria, .. } => {
                    if criteria.len() < 2 || criteria.len() > limits.choice_options {
                        return bad(format!("{}: a choice has 2 to {} options", q.id(), limits.choice_options));
                    }
                    let mut options = std::collections::BTreeSet::new();
                    for c in criteria {
                        if c.option.is_empty() || !options.insert(&c.option) {
                            return bad(format!("{}: options are non-empty and distinct", q.id()));
                        }
                    }
                }
                SpecQuestion::Score { criteria, .. } => {
                    if !(2..=10).contains(&criteria.len()) || criteria.iter().any(String::is_empty) {
                        return bad(format!("{}: a score has 2 to 10 non-empty levels", q.id()));
                    }
                }
                SpecQuestion::Text { max_tokens, .. } => {
                    if *max_tokens == 0 || *max_tokens > limits.text_tokens as u64 {
                        return bad(format!("{}: max_tokens is 1 to {}", q.id(), limits.text_tokens));
                    }
                }
                SpecQuestion::Noul { .. } => {}
            }
        }
        Ok(())
    }
}

/// RFC 8785 canonical JSON of a value holding only integers: members sorted
/// by UTF-16 code units, no whitespace, strings escaped minimally.
pub fn canonical_json(value: &Value) -> String {
    let mut out = String::new();
    write_canonical(value, &mut out);
    out
}

fn write_canonical(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&n.to_string()),
        // serde_json's escapes (`\b \t \n \f \r \" \\`, other controls as
        // lowercase `\u00xx`, everything else literal) are RFC 8785's.
        Value::String(s) => out.push_str(&serde_json::to_string(s).expect("a string serializes")),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            let mut members: Vec<(&String, &Value)> = map.iter().collect();
            members.sort_by(|a, b| a.0.encode_utf16().cmp(b.0.encode_utf16()));
            out.push('{');
            for (i, (k, v)) in members.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(k).expect("a string serializes"));
                out.push(':');
                write_canonical(v, out);
            }
            out.push('}');
        }
    }
}

/// `sha256:<hex>` of the spec's canonical JSON. The id is a function of the
/// submitted spec alone, so a re-submission is the same spec.
pub fn spec_id(body: &Value) -> String {
    format!("sha256:{}", sha256_hex_bytes(canonical_json(body).as_bytes()))
}

// -------------------------------------------------------------- control text

/// One place a control marker was found, as `signals.control_text` reports it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ControlHit {
    pub r#where: String,
    pub token: String,
}

/// Every control marker in `text`, in order, as a hit at `place`. The marker
/// is reported as it appears (`<|im_end|>` stays whole, not just its `<|`):
/// the hit names what the content spelled.
pub fn control_hits(place: &str, text: &str) -> Vec<ControlHit> {
    let mut hits = Vec::new();
    let mut rest = text;
    while let Some((at, marker)) = first_marker(rest) {
        let end = if marker == "<|" {
            // The whole special token when it closes, else the marker alone.
            rest[at..].find("|>").map_or(at + marker.len(), |i| at + i + 2)
        } else {
            at + marker.len()
        };
        hits.push(ControlHit { r#where: place.into(), token: rest[at..end].into() });
        rest = &rest[end..];
    }
    hits
}

fn first_marker(text: &str) -> Option<(usize, &'static str)> {
    CONTROL_MARKERS.iter().filter_map(|m| text.find(m).map(|i| (i, *m))).min_by_key(|(i, _)| *i)
}

/// `text` cut so that no piece holds a whole control marker. Tokenizing the
/// pieces one at a time and joining the ids gives text that is only text: a
/// tokenizer that finds `<|im_end|>` in one string finds `<` and `|im_end|>`
/// in two. The cuts fall one character into each marker.
pub fn control_pieces(text: &str) -> Vec<&str> {
    let mut pieces = Vec::new();
    let mut rest = text;
    while let Some((at, marker)) = first_marker(rest) {
        let cut = at + marker.chars().next().map_or(0, char::len_utf8);
        pieces.push(&rest[..cut]);
        rest = &rest[cut..];
    }
    pieces.push(rest);
    pieces.retain(|p| !p.is_empty());
    pieces
}

// -------------------------------------------------------------- answer maths

fn logsumexp(values: &[f64]) -> f64 {
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if max == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    let mut acc = 0.0;
    for v in values {
        acc += (v - max).exp();
    }
    max + acc.ln()
}

fn argmax(p: &[f64]) -> usize {
    let mut best = 0;
    for i in 1..p.len() {
        if p[i] > p[best] {
            best = i;
        }
    }
    best
}

/// The numbers of one read of one question, over its labels in order.
/// `logprobs` are the raw full-vocabulary log probabilities of each label's
/// answer tokens; `mass` is their logsumexp; `probabilities` the softmax over
/// the labels, `exp(logprobs[o] - mass)`. Derived numbers are f64, summed in
/// label order, so a client recomputing in float64 agrees within 1e-9.
#[derive(Clone, Debug, PartialEq)]
pub struct ReadNumbers {
    pub labels: Vec<String>,
    pub logprobs: Vec<f64>,
    pub mass: f64,
    pub probabilities: Vec<f64>,
    /// `exp(mass) * max(probabilities)`.
    pub confidence: f64,
    /// Index of the highest probability; ties go to the earlier label.
    pub top: usize,
}

pub fn read_numbers(labels: &[String], logprobs: &[f64]) -> Result<ReadNumbers> {
    if labels.len() < 2 || labels.len() != logprobs.len() {
        return Err(CouncilError::bad_request("a read needs one logprob per label, at least two labels"));
    }
    if logprobs.iter().any(|v| v.is_nan() || *v > 0.0) {
        return Err(CouncilError::bad_request("a logprob is at most 0"));
    }
    let mass = logsumexp(logprobs);
    if !mass.is_finite() {
        return Err(CouncilError::bad_request("every label has probability 0: nothing to read"));
    }
    let probabilities: Vec<f64> = logprobs.iter().map(|l| (l - mass).exp()).collect();
    let top = argmax(&probabilities);
    Ok(ReadNumbers {
        labels: labels.to_vec(),
        logprobs: logprobs.to_vec(),
        mass,
        confidence: mass.exp() * probabilities[top],
        probabilities,
        top,
    })
}

/// A score's value: the probability-weighted level number.
pub fn score_value(probabilities: &[f64]) -> f64 {
    let mut acc = 0.0;
    for (k, p) in probabilities.iter().enumerate() {
        acc += k as f64 * p;
    }
    acc
}

/// One question's answer as a read carries it.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ReadAnswer {
    Choice {
        choice: String,
        probabilities: Ordered<f64>,
        confidence: f64,
        logprobs: Ordered<f64>,
        mass: f64,
    },
    Score {
        score: f64,
        legend: Ordered<String>,
        probabilities: Ordered<f64>,
        confidence: f64,
        logprobs: Ordered<f64>,
        mass: f64,
    },
    /// Carries no confidence: the contract's yes/no is one number.
    Noul { noul: f64, logprobs: Ordered<f64>, mass: f64 },
}

fn zip<T: Clone>(labels: &[String], values: &[T]) -> Ordered<T> {
    Ordered(labels.iter().cloned().zip(values.iter().cloned()).collect())
}

/// Shape a read of `question`. `Err` for a text question, which is written,
/// not read.
pub fn read_answer(question: &SpecQuestion, logprobs: &[f64]) -> Result<ReadAnswer> {
    let labels = question.labels();
    if labels.is_empty() {
        return Err(CouncilError::bad_request("a text question is written, not read"));
    }
    let n = read_numbers(&labels, logprobs)?;
    Ok(match question {
        SpecQuestion::Choice { .. } => ReadAnswer::Choice {
            choice: n.labels[n.top].clone(),
            probabilities: zip(&n.labels, &n.probabilities),
            confidence: n.confidence,
            logprobs: zip(&n.labels, &n.logprobs),
            mass: n.mass,
        },
        SpecQuestion::Score { criteria, .. } => ReadAnswer::Score {
            score: score_value(&n.probabilities),
            legend: zip(&n.labels, criteria),
            probabilities: zip(&n.labels, &n.probabilities),
            confidence: n.confidence,
            logprobs: zip(&n.labels, &n.logprobs),
            mass: n.mass,
        },
        SpecQuestion::Noul { .. } => ReadAnswer::Noul {
            noul: n.probabilities[0],
            logprobs: zip(&n.labels, &n.logprobs),
            mass: n.mass,
        },
        SpecQuestion::Text { .. } => unreachable!("labels were empty"),
    })
}

// ---------------------------------------------------------------------- pool

/// The contract's pool: `weights` is a word, and `given` takes `values`.
#[derive(Clone, Debug, Default, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CouncilPool {
    #[serde(default)]
    pub method: PoolMethod,
    #[serde(default)]
    pub weights: PoolWeights,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub values: Option<Vec<f64>>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum PoolMethod {
    #[default]
    Linear,
    Loglinear,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum PoolWeights {
    #[default]
    Uniform,
    Mass,
    Given,
}

impl CouncilPool {
    /// The engine's pool settings for `reads` reads. `values` go with `given`
    /// and with nothing else, one per read: a misspelled shape is a `400`.
    pub fn settings(&self, reads: usize) -> Result<PoolSettings> {
        let method = match self.method {
            PoolMethod::Linear => Method::Linear,
            PoolMethod::Loglinear => Method::Loglinear,
        };
        let weights = match (self.weights, &self.values) {
            (PoolWeights::Uniform, None) => Weights::Uniform,
            (PoolWeights::Mass, None) => Weights::Mass,
            (PoolWeights::Given, Some(values)) => Weights::Given(values.clone()),
            (PoolWeights::Given, None) => {
                return Err(CouncilError::bad_request("weights given needs values").param("pool.values"));
            }
            (_, Some(_)) => {
                return Err(CouncilError::bad_request("values go with weights given only").param("pool.values"));
            }
        };
        let settings = PoolSettings { method, weights };
        settings.validate(reads).map_err(|e| CouncilError::bad_request(e).param("pool"))?;
        Ok(settings)
    }
}

/// A pooled answer's numbers for one question.
#[derive(Clone, Debug, PartialEq)]
pub struct PooledNumbers {
    pub labels: Vec<String>,
    pub probabilities: Vec<f64>,
    pub top: usize,
    /// The pooled top probability times the pool-weighted mean of the reads'
    /// `exp(mass)`.
    pub confidence: f64,
    pub agree: bool,
    pub spread: f64,
    /// By read, in request order; `None` where no weighted read remains.
    pub leave_one_out: Vec<Option<Vec<f64>>>,
    pub weights: Vec<f64>,
}

/// Pool one question's reads. `reads` are the per-read numbers in request
/// order.
pub fn pooled_numbers(reads: &[ReadNumbers], settings: &PoolSettings) -> Result<PooledNumbers> {
    let first = reads.first().ok_or_else(|| CouncilError::bad_request("no reads to pool"))?;
    let logprobs: Vec<Vec<f64>> = reads.iter().map(|r| r.logprobs.clone()).collect();
    let mass: Vec<f64> = reads.iter().map(|r| r.mass.exp()).collect();
    let pooled = crate::pool::pool(&logprobs, &mass, settings).map_err(CouncilError::bad_request)?;
    let top = argmax(&pooled.probs);
    let mut mean_mass = 0.0;
    for (w, m) in pooled.weights.iter().zip(&mass) {
        mean_mass += w * m;
    }
    Ok(PooledNumbers {
        labels: first.labels.clone(),
        confidence: pooled.probs[top] * mean_mass,
        probabilities: pooled.probs,
        top,
        agree: pooled.agree,
        spread: pooled.spread,
        leave_one_out: pooled.leave_one_out,
        weights: pooled.weights,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const A: &str = "0199b3c4-6c1e-7a2b-9f00-3e5d1c2a7b11";

    fn turn(content: &str, snap: bool) -> Turn {
        Turn { role: Role::User, content: content.into(), snap, reasoning: None }
    }
    fn body(system: Option<&str>, turns: Vec<Turn>) -> ContextBody {
        ContextBody {
            system: system.map(String::from),
            turns,
            pin: None,
            dry_run: false,
            warm: vec![],
        }
    }

    // ---- ids

    #[test]
    fn a_context_id_is_a_lowercase_uuid_and_nothing_else() {
        assert_eq!(parse_context_id(A).unwrap(), A);
        for bad in [
            "",
            "0199B3C4-6C1E-7A2B-9F00-3E5D1C2A7B11", // uppercase would be rewritten
            "0199b3c46c1e7a2b9f003e5d1c2a7b11",     // no hyphens
            "{0199b3c4-6c1e-7a2b-9f00-3e5d1c2a7b11}",
            "0199b3c4-6c1e-7a2b-9f00-3e5d1c2a7b1",  // one short
            "0199b3c4-6c1e-7a2b-9f00-3e5d1c2a7b1g", // not hex
            &"a".repeat(64),                         // a snapshot-shaped id is not a context id
        ] {
            let e = parse_context_id(bad).unwrap_err();
            assert_eq!(e.status, 400, "{bad}");
            assert_eq!(e.param.as_deref(), Some("id"));
        }
    }

    // ---- PUT planning

    #[test]
    fn a_first_put_feeds_everything() {
        let new = body(Some("s"), vec![turn("a", true)]);
        assert_eq!(plan_put(None, &new), PutPlan { kept_system: false, kept_turns: 0 });
    }

    #[test]
    fn an_update_extends_from_the_last_snap_inside_the_common_prefix() {
        let held = vec![turn("a", true), turn("b", false), turn("c", true), turn("d", false)];
        let sys = Some("s".to_string());
        // An appended turn: the held build had boundaries after a and c.
        let mut turns = held.clone();
        turns.push(turn("e", false));
        assert_eq!(
            plan_put(Some((&sys, &held)), &body(Some("s"), turns)),
            PutPlan { kept_system: true, kept_turns: 3 },
            "keeps through c, the last declared boundary; d is re-fed"
        );
        // An edit at b: only a's boundary survives.
        let mut turns = held.clone();
        turns[1] = turn("B", false);
        assert_eq!(
            plan_put(Some((&sys, &held)), &body(Some("s"), turns)),
            PutPlan { kept_system: true, kept_turns: 1 }
        );
        // An edit at the first turn leaves only the system boundary.
        let mut turns = held.clone();
        turns[0] = turn("A", true);
        assert_eq!(
            plan_put(Some((&sys, &held)), &body(Some("s"), turns)),
            PutPlan { kept_system: true, kept_turns: 0 }
        );
    }

    #[test]
    fn toggling_a_snap_flag_is_an_edit_at_that_turn() {
        let sys = Some("s".to_string());
        let held = vec![turn("a", true), turn("b", true)];
        let new = body(Some("s"), vec![turn("a", true), turn("b", false)]);
        assert_eq!(plan_put(Some((&sys, &held)), &new).kept_turns, 1, "b's boundary is gone");
        let new = body(Some("s"), vec![turn("a", false), turn("b", true)]);
        assert_eq!(plan_put(Some((&sys, &held)), &new).kept_turns, 0, "a's flag is content");
    }

    #[test]
    fn a_changed_system_message_keeps_nothing() {
        let held = vec![turn("a", true)];
        let sys = Some("s".to_string());
        let new = body(Some("S"), vec![turn("a", true)]);
        assert_eq!(plan_put(Some((&sys, &held)), &new), PutPlan { kept_system: false, kept_turns: 0 });
        let new = body(None, vec![turn("a", true)]);
        assert!(!plan_put(Some((&sys, &held)), &new).kept_system);
    }

    #[test]
    fn absent_and_empty_reasoning_are_the_same_turn() {
        let sys = Some("s".to_string());
        let mk = |reasoning: Option<&str>| Turn {
            role: Role::Assistant,
            content: "x".into(),
            snap: true,
            reasoning: reasoning.map(String::from),
        };
        let held = vec![mk(None)];
        assert_eq!(plan_put(Some((&sys, &held)), &body(Some("s"), vec![mk(Some("")) ])).kept_turns, 1);
        assert_eq!(
            plan_put(Some((&sys, &held)), &body(Some("s"), vec![mk(Some("thinking"))])).kept_turns,
            0,
            "real reasoning is content"
        );
    }

    #[test]
    fn a_context_body_refuses_what_the_contract_refuses() {
        let limits = Limits::default();
        assert!(body(Some("s"), vec![]).validate(&limits).is_ok());
        assert!(body(None, vec![]).validate(&limits).is_err());
        let mut b = body(Some("s"), vec![Turn {
            role: Role::User,
            content: "x".into(),
            snap: false,
            reasoning: Some("why".into()),
        }]);
        let e = b.validate(&limits).unwrap_err();
        assert_eq!((e.status, e.param.as_deref()), (400, Some("turns[0].reasoning")));
        b.turns[0].reasoning = None;
        b.warm = vec!["x".into(), "x".into()];
        assert!(b.validate(&limits).is_err(), "warm repeats");
        let unknown = serde_json::from_value::<ContextBody>(json!({"system": "s", "tunrs": []}));
        assert!(unknown.is_err(), "an unknown field is a 400, never ignored");
    }

    // ---- specs

    fn spec_json() -> Value {
        json!({
            "name": "shell-gate",
            "instructions": "Judge the proposed statement.",
            "input_label": "Proposed statement",
            "questions": [
                {"id": "effect", "type": "text", "instructions": "What it does.", "max_tokens": 48},
                {"id": "undo", "type": "score", "instructions": "How hard to undo.",
                 "criteria": ["easy", "hard", "impossible"]},
                {"id": "verdict", "type": "choice", "instructions": "What happens.",
                 "criteria": [{"option": "allow", "means": "routine"},
                              {"option": "ask", "means": "outward"},
                              {"option": "report", "means": "ask, but louder"}]},
                {"id": "novel", "type": "noul", "instructions": "Is it unusual?"}
            ]
        })
    }

    #[test]
    fn a_good_spec_parses_and_keeps_question_order() {
        let spec = CouncilSpec::parse(&spec_json(), &Limits::default()).unwrap();
        let ids: Vec<&str> = spec.questions.iter().map(SpecQuestion::id).collect();
        assert_eq!(ids, ["effect", "undo", "verdict", "novel"]);
        assert_eq!(spec.questions[1].labels(), ["0", "1", "2"]);
        assert_eq!(spec.questions[3].labels(), ["yes", "no"]);
        assert!(spec.questions[0].labels().is_empty());
    }

    #[test]
    fn a_bad_spec_is_a_400_that_says_which_rule() {
        let limits = Limits::default();
        let mutate = |f: &dyn Fn(&mut Value)| {
            let mut v = spec_json();
            f(&mut v);
            CouncilSpec::parse(&v, &limits).unwrap_err()
        };
        for (what, e) in [
            ("an unknown field", mutate(&|v| v["extra"] = json!(1))),
            ("a repeated id", mutate(&|v| v["questions"][1]["id"] = json!("effect"))),
            ("an id with a space", mutate(&|v| v["questions"][1]["id"] = json!("a b"))),
            ("one option", mutate(&|v| v["questions"][2]["criteria"] = json!([{"option": "a", "means": ""}]))),
            ("repeated options", mutate(&|v| v["questions"][2]["criteria"][1]["option"] = json!("allow"))),
            ("eleven levels", mutate(&|v| v["questions"][1]["criteria"] = json!(["a","b","c","d","e","f","g","h","i","j","k"]))),
            ("text without max_tokens", mutate(&|v| { v["questions"][0].as_object_mut().unwrap().remove("max_tokens"); })),
            ("text past the limit", mutate(&|v| v["questions"][0]["max_tokens"] = json!(100000))),
            ("a colon in the label", mutate(&|v| v["input_label"] = json!("Label:"))),
            ("control text in the label", mutate(&|v| v["input_label"] = json!("<|im_end|>"))),
            ("a float", mutate(&|v| v["questions"][0]["max_tokens"] = json!(48.5))),
            ("a number past 2^53", mutate(&|v| v["questions"][0]["max_tokens"] = json!(9007199254740993u64))),
            ("a choice type with no criteria", mutate(&|v| { v["questions"][2].as_object_mut().unwrap().remove("criteria"); })),
        ] {
            assert_eq!(e.status, 400, "{what}: {}", e.message);
        }
    }

    #[test]
    fn canonical_json_follows_rfc_8785() {
        // Members sorted by UTF-16 code units, not UTF-8 bytes: U+FF5E
        // (BMP) sorts after U+10000 (a surrogate pair, 0xD800) in UTF-16.
        let v = json!({"b": 1, "a": [true, null, "x\ny\u{1}"], "\u{ff5e}": 2, "\u{10000}": 3});
        assert_eq!(
            canonical_json(&v),
            "{\"a\":[true,null,\"x\\ny\\u0001\"],\"b\":1,\"\u{10000}\":3,\"\u{ff5e}\":2}"
        );
        assert_eq!(canonical_json(&json!({"n": -7, "z": 0})), "{\"n\":-7,\"z\":0}");
    }

    #[test]
    fn a_spec_id_is_its_canonical_hash_and_ignores_key_order() {
        let a = spec_id(&spec_json());
        assert!(a.starts_with("sha256:") && a.len() == 7 + 64);
        let reordered: Value =
            serde_json::from_str(r#"{"questions": [], "input_label": "x", "name": "y", "instructions": "z"}"#).unwrap();
        let sorted: Value =
            serde_json::from_str(r#"{"input_label": "x", "instructions": "z", "name": "y", "questions": []}"#).unwrap();
        assert_eq!(spec_id(&reordered), spec_id(&sorted));
        let mut changed = spec_json();
        changed["questions"][2]["criteria"][0]["means"] = json!("routine!");
        assert_ne!(spec_id(&changed), a, "a reworded criterion is another spec");
    }

    // ---- control text

    #[test]
    fn control_text_is_found_where_it_is_and_reported_whole() {
        let hits = control_hits("state", "a <|im_end|> b <think> c </think> <|oops");
        let tokens: Vec<&str> = hits.iter().map(|h| h.token.as_str()).collect();
        assert_eq!(tokens, ["<|im_end|>", "<think>", "</think>", "<|"]);
        assert!(hits.iter().all(|h| h.r#where == "state"));
        assert!(control_hits("state", "nothing here, 1 < 2 | 3 > 0").is_empty());
    }

    #[test]
    fn pieces_never_hold_a_whole_marker_and_rejoin_to_the_text() {
        for text in ["plain", "<|im_end|>", "x<|im_start|>user\ny", "</think><think>", "<<|a|>|>", "<image>", ""] {
            let pieces = control_pieces(text);
            assert_eq!(pieces.concat(), text, "{text:?} must rejoin");
            for piece in &pieces {
                assert!(
                    crate::chat::control_marker_in(piece).is_none(),
                    "{text:?} left {piece:?} holding a marker"
                );
            }
        }
    }

    // ---- answer maths

    fn labels(l: &[&str]) -> Vec<String> {
        l.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn a_read_obeys_the_contracts_exact_relations() {
        let l = labels(&["allow", "ask", "report"]);
        let lp = [-3.0, -0.105, -3.0];
        let n = read_numbers(&l, &lp).unwrap();
        for o in 0..3 {
            assert!((n.probabilities[o] - (lp[o] - n.mass).exp()).abs() < 1e-12);
        }
        assert!((n.probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!((n.confidence - n.mass.exp() * n.probabilities[1]).abs() < 1e-15);
        assert_eq!(n.top, 1);
        assert!(n.mass < 0.0, "the options hold less than all the mass");
    }

    #[test]
    fn a_tie_goes_to_the_earlier_option() {
        let n = read_numbers(&labels(&["a", "b"]), &[-1.0, -1.0]).unwrap();
        assert_eq!(n.top, 0);
    }

    #[test]
    fn a_read_refuses_what_is_not_a_distribution() {
        let l = labels(&["a", "b"]);
        assert!(read_numbers(&l, &[-1.0]).is_err());
        assert!(read_numbers(&l, &[0.5, -1.0]).is_err(), "a logprob above 0");
        assert!(read_numbers(&l, &[f64::NAN, -1.0]).is_err());
        assert!(read_numbers(&l, &[f64::NEG_INFINITY, f64::NEG_INFINITY]).is_err());
        assert!(read_numbers(&labels(&["a"]), &[-1.0]).is_err());
    }

    #[test]
    fn answers_serialize_in_option_order_not_key_order() {
        let spec = CouncilSpec::parse(&spec_json(), &Limits::default()).unwrap();
        let verdict = read_answer(&spec.questions[2], &[-3.0, -0.105, -3.0]).unwrap();
        let json = serde_json::to_string(&verdict).unwrap();
        assert!(json.starts_with(r#"{"type":"choice","choice":"ask","#), "{json}");
        let q = SpecQuestion::Choice {
            id: "q".into(),
            instructions: json!("x"),
            criteria: ["zebra", "apple"].map(|o| ChoiceOption { option: o.into(), means: String::new() }).to_vec(),
        };
        let answer = serde_json::to_string(&read_answer(&q, &[-0.5, -1.5]).unwrap()).unwrap();
        assert!(answer.find("zebra").unwrap() < answer.find("apple").unwrap(), "{answer}");
    }

    #[test]
    fn a_score_is_the_probability_weighted_level_number() {
        let spec = CouncilSpec::parse(&spec_json(), &Limits::default()).unwrap();
        let p = [0.5f64, 0.25, 0.25];
        let lp: Vec<f64> = p.iter().map(|x| x.ln()).collect();
        match read_answer(&spec.questions[1], &lp).unwrap() {
            ReadAnswer::Score { score, legend, .. } => {
                assert!((score - 0.75).abs() < 1e-12);
                assert_eq!(legend.0[0], ("0".to_string(), "easy".to_string()));
                assert_eq!(legend.0[2].1, "impossible");
            }
            other => panic!("not a score: {other:?}"),
        }
    }

    #[test]
    fn a_noul_is_the_probability_of_yes_and_carries_no_confidence() {
        let spec = CouncilSpec::parse(&spec_json(), &Limits::default()).unwrap();
        let answer = read_answer(&spec.questions[3], &[(0.8f64).ln(), (0.2f64).ln()]).unwrap();
        let json = serde_json::to_value(&answer).unwrap();
        assert!((json["noul"].as_f64().unwrap() - 0.8).abs() < 1e-12);
        assert!(json.get("confidence").is_none());
        assert_eq!(json["logprobs"].as_object().unwrap().len(), 2);
        assert!(read_answer(&spec.questions[0], &[-1.0, -1.0]).is_err(), "text is written, not read");
    }

    // ---- pool

    #[test]
    fn the_contracts_pool_maps_onto_the_engines() {
        let p: CouncilPool = serde_json::from_value(json!({"method": "loglinear", "weights": "mass"})).unwrap();
        let s = p.settings(2).unwrap();
        assert_eq!((s.method, s.weights), (Method::Loglinear, Weights::Mass));
        let given: CouncilPool =
            serde_json::from_value(json!({"weights": "given", "values": [1.0, 3.0]})).unwrap();
        assert_eq!(given.settings(2).unwrap().weights, Weights::Given(vec![1.0, 3.0]));
        assert!(given.settings(3).is_err(), "one value per read");
        let missing: CouncilPool = serde_json::from_value(json!({"weights": "given"})).unwrap();
        assert!(missing.settings(2).is_err());
        let stray: CouncilPool = serde_json::from_value(json!({"values": [1.0]})).unwrap();
        assert!(stray.settings(1).is_err(), "values without given");
        assert!(serde_json::from_value::<CouncilPool>(json!({"weights": [1, 2]})).is_err());
        assert_eq!(CouncilPool::default().settings(1).unwrap(), PoolSettings::default());
    }

    #[test]
    fn a_pool_replays_from_its_reads_and_its_confidence_is_the_contracts() {
        let l = labels(&["allow", "ask"]);
        let a = read_numbers(&l, &[(0.3f64).ln(), (0.5f64).ln()]).unwrap();
        let b = read_numbers(&l, &[(0.05f64).ln(), (0.9f64).ln()]).unwrap();
        let settings = CouncilPool { weights: PoolWeights::Mass, ..Default::default() }.settings(2).unwrap();
        let pooled = pooled_numbers(&[a.clone(), b.clone()], &settings).unwrap();
        assert_eq!(pooled, pooled_numbers(&[a.clone(), b.clone()], &settings).unwrap(), "bit for bit");
        assert!((pooled.probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        let mean_mass: f64 = pooled.weights.iter().zip([&a, &b]).map(|(w, r)| w * r.mass.exp()).sum();
        assert!((pooled.confidence - pooled.probabilities[pooled.top] * mean_mass).abs() < 1e-12);
        assert_eq!(pooled.leave_one_out.len(), 2);
        // Without b, the pool is a's own distribution.
        let without_b = pooled.leave_one_out[1].as_ref().unwrap();
        assert!((without_b[1] - a.probabilities[1]).abs() < 1e-12);
        assert!(pooled.agree && pooled.top == 1);
    }

    #[test]
    fn one_read_pools_to_itself_with_no_leave_one_out() {
        let l = labels(&["a", "b"]);
        let r = read_numbers(&l, &[(0.2f64).ln(), (0.7f64).ln()]).unwrap();
        let pooled = pooled_numbers(&[r.clone()], &PoolSettings::default()).unwrap();
        assert!(pooled.leave_one_out.is_empty());
        assert!(pooled.agree && pooled.spread == 0.0);
        assert!((pooled.probabilities[1] - r.probabilities[1]).abs() < 1e-12);
    }

    // ---- errors

    #[test]
    fn an_error_body_is_the_contracts() {
        let e = CouncilError::precondition_failed("snap:abc");
        assert_eq!(e.status, 412);
        assert_eq!(
            e.to_body(),
            json!({"error": {"type": "precondition_failed",
                             "message": "the context's head is not the one If-Match named",
                             "head": "snap:abc"}})
        );
        let e = CouncilError::bad_request("x").param("turns[0]");
        assert_eq!(e.to_body()["error"]["param"], "turns[0]");
    }
}
