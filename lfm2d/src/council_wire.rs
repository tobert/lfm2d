//! The council contract's response bodies, as typed values that serialize to
//! exactly the shapes `council-api.openapi.yaml` describes.
//!
//! `tests/council_contract.rs` validates every one of these against the
//! vendored schemas, so a field renamed here, or a number out of range, is a
//! red test and not a client's decode error. The numbers come from
//! [`crate::council`] (`read_answer`, `pooled_numbers`); this module only
//! shapes them, and keeps option order everywhere ([`Ordered`]).
//!
//! Names on the wire are the client's: a context is its UUID, a snapshot is
//! `snap:<64 hex>`, a spec is `sha256:<64 hex>`. The sha256 content ids the
//! engine keeps internally (`state_store`, `chat_session`) never appear
//! except as the hex inside a `snap:` id.
use serde::Serialize;

use crate::council::{
    ControlHit, CouncilError, CouncilSpec, Limits, Ordered, PoolMethod, PoolWeights, PooledNumbers,
    SpecQuestion, score_value,
};

pub use crate::council::ReadAnswer;

type Result<T> = std::result::Result<T, CouncilError>;

/// `snap:` and 64 lowercase hex digits: the opaque address of one build of a
/// context prefix. Built from the engine's internal content id, so the same
/// build always has the same wire id.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct SnapshotId(String);

impl SnapshotId {
    /// From the 64-hex digest the engine keys the build by.
    pub fn from_digest(hex: &str) -> Result<Self> {
        if hex.len() == 64 && hex.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) {
            Ok(Self(format!("snap:{hex}")))
        } else {
            Err(CouncilError::internal(format!("{hex:?} is not a 64-digit lowercase hex digest")))
        }
    }

    /// A snapshot id a client sent (a decision's `at`, an `If-Match`).
    pub fn parse(id: &str) -> Result<Self> {
        id.strip_prefix("snap:")
            .map(Self::from_digest)
            .transpose()
            .ok()
            .flatten()
            .ok_or_else(|| CouncilError::bad_request(format!("{id:?} is not a snapshot id: snap: and 64 hex digits")))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

// ---------------------------------------------------------------- contexts

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Layer {
    System,
    Turn,
    Spec,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SnapshotInfo {
    pub id: SnapshotId,
    pub tokens: usize,
    pub layer: Layer,
    /// For a turn snapshot, the index of the turn it ends.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spec_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parked: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pinned: Option<bool>,
}

/// `GET /council/v1/contexts/{id}`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ContextState {
    pub id: String,
    pub head: SnapshotId,
    pub tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pinned: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub persist: Option<bool>,
    pub snapshots: Vec<SnapshotInfo>,
}

/// `PUT /council/v1/contexts/{id}`'s answer: the state, and what the build
/// kept and fed. With `dry_run`, nothing was built and `head` is the id the
/// build would have.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ContextPutResult {
    #[serde(flatten)]
    pub state: ContextState,
    pub kept: usize,
    pub fed: usize,
    pub dry_run: bool,
}

// ------------------------------------------------------------------- specs

/// `POST /council/v1/specs`'s answer, and an element of `GET /specs`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct HeldSpec {
    pub spec_id: String,
    pub spec: CouncilSpec,
    /// How this server compiles and renders the spec; part of identity.
    pub template: String,
}

// ---------------------------------------------------------------- identity

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    Park,
    Warm,
    DryRun,
    Persist,
    Describe,
    LeaveOneOut,
}

/// `GET /council/v1/identity`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ServerIdentity {
    pub model: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub aliases: Vec<String>,
    pub weight_hash: String,
    pub tokenizer_hash: String,
    pub template: String,
    /// Required by the contract's typed client, and information only here:
    /// nothing on our side gates on it (Amy, 2026-10-07).
    pub engine: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
    /// With `describe`, the rule that ends a `text` answer.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_stop: Option<String>,
    pub limits: Limits,
    pub capabilities: Vec<Capability>,
}

/// The fields a decision repeats from the server's identity.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DecisionIdentity {
    pub model: String,
    pub weight_hash: String,
    pub tokenizer_hash: String,
    pub template: String,
    pub engine: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spec_id: Option<String>,
}

// ----------------------------------------------------------------- answers

/// A choice or score's `leave_one_out`: the pooled probabilities without each
/// read, keyed by context id; `null` where no weighted read remains.
pub type LeaveOneOut = Ordered<Option<Ordered<f64>>>;

/// One question pooled across the reads. `agree` and `spread` are the
/// contract's; there is no winner beyond the argmax a Decisions client wants,
/// which is not a decision.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum PooledAnswer {
    Choice {
        choice: String,
        probabilities: Ordered<f64>,
        confidence: f64,
        agree: bool,
        spread: f64,
        #[serde(skip_serializing_if = "Option::is_none")]
        leave_one_out: Option<LeaveOneOut>,
    },
    Score {
        score: f64,
        legend: Ordered<String>,
        probabilities: Ordered<f64>,
        confidence: f64,
        agree: bool,
        spread: f64,
        #[serde(skip_serializing_if = "Option::is_none")]
        leave_one_out: Option<LeaveOneOut>,
    },
    /// The probability of yes. Carries no confidence.
    Noul {
        noul: f64,
        agree: bool,
        spread: f64,
        #[serde(skip_serializing_if = "Option::is_none")]
        leave_one_out: Option<Ordered<Option<f64>>>,
    },
}

fn zip<T: Clone>(labels: &[String], values: &[T]) -> Ordered<T> {
    Ordered(labels.iter().cloned().zip(values.iter().cloned()).collect())
}

/// Each context's leave-one-out row, in request order, `None` where the
/// pool had no weighted read left without it.
fn keyed<T>(contexts: &[String], rows: &[Option<Vec<f64>>], f: impl Fn(&Vec<f64>) -> T) -> Ordered<Option<T>> {
    Ordered(contexts.iter().zip(rows).map(|(id, row)| (id.clone(), row.as_ref().map(&f))).collect())
}

/// Shape one question's pooled numbers. `contexts` are the reads' context
/// ids in request order; `leave_one_out` says whether the server lists that
/// capability, and it is emitted only with two or more reads.
pub fn pooled_answer(
    question: &SpecQuestion,
    pooled: &PooledNumbers,
    contexts: &[String],
    leave_one_out: bool,
) -> Result<PooledAnswer> {
    let emit = leave_one_out && contexts.len() >= 2;
    if emit && pooled.leave_one_out.len() != contexts.len() {
        return Err(CouncilError::internal(format!(
            "{} leave-one-out rows for {} contexts",
            pooled.leave_one_out.len(),
            contexts.len()
        )));
    }
    let probabilities = zip(&pooled.labels, &pooled.probabilities);
    Ok(match question {
        SpecQuestion::Choice { .. } => PooledAnswer::Choice {
            choice: pooled.labels[pooled.top].clone(),
            probabilities,
            confidence: pooled.confidence,
            agree: pooled.agree,
            spread: pooled.spread,
            leave_one_out: emit.then(|| keyed(contexts, &pooled.leave_one_out, |p| zip(&pooled.labels, p))),
        },
        SpecQuestion::Score { criteria, .. } => PooledAnswer::Score {
            score: score_value(&pooled.probabilities),
            legend: zip(&pooled.labels, criteria),
            probabilities,
            confidence: pooled.confidence,
            agree: pooled.agree,
            spread: pooled.spread,
            leave_one_out: emit.then(|| keyed(contexts, &pooled.leave_one_out, |p| zip(&pooled.labels, p))),
        },
        SpecQuestion::Noul { .. } => PooledAnswer::Noul {
            noul: pooled.probabilities[0],
            agree: pooled.agree,
            spread: pooled.spread,
            leave_one_out: emit.then(|| keyed(contexts, &pooled.leave_one_out, |p| p[0])),
        },
        SpecQuestion::Text { .. } => {
            return Err(CouncilError::bad_request("a text question is written, not pooled"));
        }
    })
}

// ---------------------------------------------------------------- decision

/// One context's read of the case, or the spec's alone (`context: null`).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Read {
    pub context: Option<String>,
    /// The snapshot this read started from: the spec layer when there is a
    /// spec; `null` for inline questions with no context.
    pub snapshot: Option<SnapshotId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub described: Option<Ordered<String>>,
    pub answers: Ordered<ReadAnswer>,
    pub rendered_sha256: String,
    /// The context's length in tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ms: Option<f64>,
}

/// The pool as run: the method and weights echoed, and by question the
/// weight each read pooled with, in read order.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PoolEcho {
    pub method: PoolMethod,
    pub weights: PoolWeights,
    pub normalized: Ordered<Vec<f64>>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Signals {
    pub control_text: Vec<ControlHit>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Usage {
    /// The case as rendered.
    pub input_tokens: usize,
    /// The text answers written.
    pub output_tokens: usize,
    /// Tokens run past held snapshots, spec-layer rebuilds included.
    pub fed_tokens: usize,
}

/// `POST /council/v1/decisions`'s answer.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DecisionResponse {
    pub model: String,
    pub answers: Ordered<PooledAnswer>,
    pub reads: Vec<Read>,
    pub pool: PoolEcho,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signals: Option<Signals>,
    pub identity: DecisionIdentity,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub queue_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ms: Option<f64>,
}
