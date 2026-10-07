//! `POST /council/v1/decisions`, as pure functions: validate a request,
//! plan which questions are read and over which options, build the engine's
//! opinion request, and turn the engine's reads back into the contract's
//! answer. Nothing here touches the engine or a lock; [`crate::council_api`]
//! resolves names to held things around it.
//!
//! The engine reads one context at a time ([`crate::adjudicator`]'s
//! multi-context opinion) and this module re-derives every number the
//! contract promises from the reads' raw log probabilities, in float64,
//! summing in request order ([`crate::council::read_numbers`],
//! [`crate::council::pooled_numbers`]). The engine's own pool is never
//! echoed: it would be the same numbers by a second route.
//!
//! Deviations from the contract, each a choice to revisit:
//!
//! - inline `questions` (the Decisions API form) are refused with a `400`:
//!   hold a spec and name it. A compatibility layer is a separate surface;
//! - a read's `snapshot` is the context snapshot it started from. This
//!   server holds no spec-layer snapshots, and a decision with no contexts
//!   has none to name (`null`);
//! - a read's `rendered_sha256` is the engine's hash for its last asked
//!   slot, the longest text that read was conditioned on;
//! - `usage` is approximate: `input_tokens` is the read turn after the
//!   context (the spec's head and the state), `output_tokens` the described
//!   text written across reads, `fed_tokens` what the engine ran past what
//!   it already held;
//! - an object or array `state` is JSON in request order with the
//!   template's `", "` and `": "` separators.
use std::collections::{BTreeMap, HashSet};

use serde::Deserialize;
use serde_json::{Value, json};

use crate::chat::TemplateValue;
use crate::council::{
    CouncilError, CouncilPool, CouncilSpec, Limits, Ordered, ReadAnswer, SpecQuestion, is_spec_id, parse_context_id,
    pooled_numbers, read_answer, read_numbers,
};
use crate::council_compile::choices;
use crate::council_wire::{
    DecisionIdentity, DecisionResponse, PoolEcho, Read, ServerIdentity, Signals, SnapshotId, Usage, pooled_answer,
};
use crate::opinion_api::{OpinionRequest, OpinionResponse};
use crate::pool::PoolSettings;

type Result<T> = std::result::Result<T, CouncilError>;

/// A context a decision reads after, optionally as of one of its snapshots.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContextRef {
    pub id: String,
    #[serde(default)]
    pub at: Option<String>,
}

/// `POST /council/v1/decisions`'s body.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub state: TemplateValue,
    #[serde(default)]
    pub spec_id: Option<String>,
    #[serde(default)]
    pub ask: Option<Vec<String>>,
    #[serde(default)]
    pub options: Option<BTreeMap<String, Vec<String>>>,
    /// Inline Decisions API questions: accepted by the schema, refused here.
    #[serde(default)]
    pub questions: Option<Value>,
    #[serde(default)]
    pub contexts: Option<Vec<ContextRef>>,
    #[serde(default)]
    pub pool: Option<CouncilPool>,
    #[serde(default)]
    pub timeout_ms: Option<u64>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub trace: Option<Value>,
    #[serde(default)]
    pub provider: Option<Value>,
}

/// A request that passed every check that needs no held state.
#[derive(Clone, Debug)]
pub struct Validated {
    pub spec_id: String,
    pub state: String,
    pub contexts: Vec<(String, Option<SnapshotId>)>,
    pub pool: CouncilPool,
    /// The engine's settings for this many reads.
    pub settings: PoolSettings,
    pub timeout_ms: u64,
}

impl DecisionRequest {
    pub fn validate(&self, identity: &ServerIdentity, limits: &Limits) -> Result<Validated> {
        if self.questions.is_some() {
            return Err(CouncilError::bad_request(
                "inline questions are not served here: hold a spec (POST /council/v1/specs) and name it with spec_id",
            )
            .param("questions"));
        }
        let spec_id = self
            .spec_id
            .clone()
            .ok_or_else(|| CouncilError::bad_request("a decision names a held spec with spec_id").param("spec_id"))?;
        if !is_spec_id(&spec_id) {
            return Err(CouncilError::bad_request(format!(
                "{spec_id:?} is not a spec id: sha256: and 64 lowercase hex digits"
            ))
            .param("spec_id"));
        }
        if let Some(model) = &self.model
            && *model != identity.model
            && !identity.aliases.contains(model)
        {
            return Err(CouncilError::bad_request(format!(
                "this server is {:?}, not {model:?}",
                identity.model
            ))
            .param("model"));
        }
        if self.session_id.as_deref().is_some_and(|s| s.len() > 256) {
            return Err(CouncilError::bad_request("session_id is at most 256 characters").param("session_id"));
        }
        let state = match &self.state {
            TemplateValue::Str(s) => s.clone(),
            TemplateValue::Map(_) | TemplateValue::List(_) => self.state.to_json(),
            _ => return Err(CouncilError::bad_request("state is a string, an object or an array").param("state")),
        };
        if state.trim().is_empty() {
            return Err(CouncilError::bad_request("state is empty").param("state"));
        }
        if state.len() > limits.state_bytes {
            return Err(CouncilError::too_large(
                format!("the state is {} bytes; this server reads at most {}", state.len(), limits.state_bytes),
                "state",
            ));
        }
        let refs = self.contexts.clone().unwrap_or_default();
        if refs.len() > limits.contexts_per_decision {
            return Err(CouncilError::bad_request(format!(
                "a decision reads after at most {} contexts, got {}",
                limits.contexts_per_decision,
                refs.len()
            ))
            .param("contexts"));
        }
        let mut seen = HashSet::new();
        let mut contexts = Vec::with_capacity(refs.len());
        for (n, r) in refs.iter().enumerate() {
            parse_context_id(&r.id).map_err(|e| e.param(format!("contexts[{n}].id")))?;
            if !seen.insert(r.id.as_str()) {
                return Err(CouncilError::bad_request(format!("context {:?} is named twice", r.id))
                    .param(format!("contexts[{n}].id")));
            }
            let at = r
                .at
                .as_deref()
                .map(SnapshotId::parse)
                .transpose()
                .map_err(|e| e.param(format!("contexts[{n}].at")))?;
            contexts.push((r.id.clone(), at));
        }
        let pool = self.pool.clone().unwrap_or_default();
        let settings = pool.settings(contexts.len().max(1))?;
        let timeout_ms = self.timeout_ms.unwrap_or(limits.default_timeout_ms);
        if timeout_ms == 0 || timeout_ms > 600_000 {
            return Err(CouncilError::bad_request("timeout_ms must be 1..=600000").param("timeout_ms"));
        }
        Ok(Validated { spec_id, state, contexts, pool, settings, timeout_ms })
    }
}

/// One question this decision reads: as the spec has it, narrowed to the
/// options asked for, and the options the engine is told to score (`None`:
/// all of them).
#[derive(Clone, Debug)]
pub struct Asked {
    pub question: SpecQuestion,
    pub engine_options: Option<Vec<String>>,
}

/// Which of the spec's questions are read, and over which options. Spec
/// order, whatever order the request names them in.
pub fn plan(
    spec: &CouncilSpec,
    ask: &Option<Vec<String>>,
    options: &Option<BTreeMap<String, Vec<String>>>,
) -> Result<Vec<Asked>> {
    let readable: Vec<&SpecQuestion> = spec.questions.iter().filter(|q| !matches!(q, SpecQuestion::Text { .. })).collect();
    if let Some(ask) = ask {
        if ask.is_empty() {
            return Err(CouncilError::bad_request("ask names no question").param("ask"));
        }
        let mut seen = HashSet::new();
        for id in ask {
            if !seen.insert(id.as_str()) {
                return Err(CouncilError::bad_request(format!("ask repeats {id:?}")).param("ask"));
            }
            match spec.questions.iter().find(|q| q.id() == id) {
                None => {
                    return Err(CouncilError::bad_request(format!("the spec has no question {id:?}")).param("ask"));
                }
                Some(SpecQuestion::Text { .. }) => {
                    return Err(CouncilError::bad_request(format!(
                        "{id:?} is a text question: it is written, not read"
                    ))
                    .param("ask"));
                }
                Some(_) => {}
            }
        }
    }
    let asked: Vec<&SpecQuestion> = readable
        .into_iter()
        .filter(|q| ask.as_ref().is_none_or(|ask| ask.iter().any(|id| id == q.id())))
        .collect();
    if asked.is_empty() {
        return Err(CouncilError::bad_request("the spec has no question to read").param("ask"));
    }
    if let Some(options) = options {
        for id in options.keys() {
            if !asked.iter().any(|q| q.id() == id) {
                return Err(CouncilError::bad_request(format!(
                    "options names {id:?}, which is not a question this decision reads"
                ))
                .param(format!("options.{id}")));
            }
        }
    }
    asked
        .into_iter()
        .map(|question| {
            let narrowing = options.as_ref().and_then(|o| o.get(question.id()));
            let Some(subset) = narrowing else {
                return Ok(Asked { question: question.clone(), engine_options: None });
            };
            let param = format!("options.{}", question.id());
            let SpecQuestion::Choice { id, instructions, criteria } = question else {
                return Err(CouncilError::bad_request(format!("{:?} is not a choice question", question.id()))
                    .param(param));
            };
            let mut seen = HashSet::new();
            for option in subset {
                if !criteria.iter().any(|c| &c.option == option) {
                    return Err(CouncilError::bad_request(format!("{id:?} has no option {option:?}")).param(param));
                }
                if !seen.insert(option.as_str()) {
                    return Err(CouncilError::bad_request(format!("{option:?} repeats")).param(param));
                }
            }
            if subset.len() < 2 {
                return Err(CouncilError::bad_request("narrow to at least two options").param(param));
            }
            // The engine scores the subset in the spec's order, so the
            // narrowed question keeps that order too.
            let kept: Vec<_> = criteria.iter().filter(|c| seen.contains(c.option.as_str())).cloned().collect();
            let engine_options = kept.iter().map(|c| c.option.clone()).collect();
            Ok(Asked {
                question: SpecQuestion::Choice { id: id.clone(), instructions: instructions.clone(), criteria: kept },
                engine_options: Some(engine_options),
            })
        })
        .collect()
}

/// The engine's opinion request for this decision. `contexts` are the
/// engine's ids of the snapshots to read after, in request order; none is a
/// read of the spec's own prompt.
pub fn engine_request(
    engine_spec_id: &str,
    state: &str,
    asked: &[Asked],
    contexts: &[String],
    settings: &PoolSettings,
    timeout_ms: u64,
) -> Result<OpinionRequest> {
    let questions: Vec<Value> = asked
        .iter()
        .map(|a| match &a.engine_options {
            Some(options) => json!({"field": a.question.id(), "options": options}),
            None => json!({"field": a.question.id()}),
        })
        .collect();
    let mut body = json!({
        "spec": engine_spec_id,
        "state": {"input": state},
        "questions": questions,
        "use_cache": true,
        "timeout_ms": timeout_ms,
    });
    if !contexts.is_empty() {
        body["contexts"] = json!(contexts);
        body["pool"] = serde_json::to_value(settings)
            .map_err(|e| CouncilError::internal(format!("the pool settings do not serialize: {e}")))?;
    }
    serde_json::from_value(body).map_err(|e| CouncilError::internal(format!("not an opinion request: {e}")))
}

/// What the engine scored for `asked` in one read, in the order the
/// question's own labels follow.
fn scores_of(asked: &Asked, read: &OpinionResponse) -> Result<Vec<f64>> {
    let id = asked.question.id();
    let answer = read
        .answers
        .iter()
        .find(|a| a.field == id)
        .ok_or_else(|| CouncilError::internal(format!("the engine did not answer {id:?}")))?;
    let names = choices(&asked.question).unwrap_or_default();
    names
        .iter()
        .map(|name| {
            answer
                .read
                .options
                .iter()
                .find(|o| &o.option == name)
                .map(|o| f64::from(o.logprob))
                .ok_or_else(|| CouncilError::internal(format!("the engine did not score {name:?} for {id:?}")))
        })
        .collect()
}

/// A reference to one context read: the client's id and the snapshot read.
pub type ContextRead = (String, SnapshotId);

/// Everything the contract's answer needs that is not in the engine's reads.
pub struct Assembly<'a> {
    pub model: String,
    pub identity: DecisionIdentity,
    pub spec: &'a CouncilSpec,
    pub asked: &'a [Asked],
    /// One per read, in order; empty for a decision with no contexts.
    pub contexts: &'a [ContextRead],
    pub pool: &'a CouncilPool,
    pub settings: &'a PoolSettings,
    pub queue_ms: f64,
    pub ms: f64,
}

/// The contract's answer from the engine's reads (one per context in request
/// order, or one for a decision with none).
pub fn assemble(a: &Assembly<'_>, reads: &[OpinionResponse]) -> Result<DecisionResponse> {
    let expected = a.contexts.len().max(1);
    if reads.len() != expected {
        return Err(CouncilError::internal(format!("{} reads for {expected} contexts", reads.len())));
    }
    let text_ids: HashSet<&str> = a
        .spec
        .questions
        .iter()
        .filter(|q| matches!(q, SpecQuestion::Text { .. }))
        .map(SpecQuestion::id)
        .collect();
    let context_ids: Vec<String> = a.contexts.iter().map(|(id, _)| id.clone()).collect();

    let mut out_reads = Vec::with_capacity(reads.len());
    for (n, read) in reads.iter().enumerate() {
        let mut answers: Vec<(String, ReadAnswer)> = Vec::with_capacity(a.asked.len());
        for asked in a.asked {
            answers.push((asked.question.id().to_owned(), read_answer(&asked.question, &scores_of(asked, read)?)?));
        }
        let described = (!text_ids.is_empty()).then(|| {
            Ordered(
                read.described
                    .iter()
                    .filter(|d| text_ids.contains(d.field.as_str()))
                    .map(|d| {
                        let text = match &d.value {
                            Value::String(s) => s.clone(),
                            other => other.to_string(),
                        };
                        (d.field.clone(), text)
                    })
                    .collect(),
            )
        });
        let rendered_sha256 = read
            .answers
            .last()
            .map(|answer| answer.read.rendered_sha256.clone())
            .ok_or_else(|| CouncilError::internal("a read answered no question"))?;
        out_reads.push(Read {
            context: a.contexts.get(n).map(|(id, _)| id.clone()),
            snapshot: a.contexts.get(n).map(|(_, snapshot)| snapshot.clone()),
            described,
            answers: Ordered(answers),
            rendered_sha256,
            tokens: read.context_tokens,
            ms: Some(read.prefill_ms + read.describe_ms + read.read_ms),
        });
    }

    let mut answers = Vec::with_capacity(a.asked.len());
    let mut normalized = Vec::with_capacity(a.asked.len());
    for asked in a.asked {
        let per_read: Vec<_> = reads
            .iter()
            .map(|read| read_numbers(&asked.question.labels(), &scores_of(asked, read)?))
            .collect::<Result<_>>()?;
        let pooled = pooled_numbers(&per_read, a.settings)?;
        normalized.push((asked.question.id().to_owned(), pooled.weights.clone()));
        answers.push((asked.question.id().to_owned(), pooled_answer(&asked.question, &pooled, &context_ids, true)?));
    }

    let case_tokens = |r: &OpinionResponse| r.prompt_tokens - r.context_tokens.unwrap_or(0).min(r.prompt_tokens);
    let usage = Usage {
        input_tokens: reads.first().map_or(0, case_tokens),
        output_tokens: reads.iter().map(|r| r.described_tokens).sum(),
        fed_tokens: reads.iter().map(|r| r.prompt_tokens - r.cached_tokens.min(r.prompt_tokens) + r.described_tokens).sum(),
    };
    Ok(DecisionResponse {
        model: a.model.clone(),
        answers: Ordered(answers),
        reads: out_reads,
        pool: PoolEcho { method: a.pool.method, weights: a.pool.weights, normalized: Ordered(normalized) },
        signals: Some(Signals { control_text: Vec::new() }),
        identity: a.identity.clone(),
        usage: Some(usage),
        queue_ms: Some(a.queue_ms),
        ms: Some(a.ms),
    })
}
