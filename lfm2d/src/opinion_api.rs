//! `/v1/opinion`: the typed-decision surface over the resident adjudicator.
//!
//! An opinion request names a loaded prompt spec, hands over the state (the
//! command and, optionally, the facts an app built for it) and asks one of
//! the spec's choice fields. The daemon renders the same prompt the
//! generative path renders, generates the fields that precede the question
//! under the output grammar — the *description* — stops at the question's
//! value slot, and teacher-forces every option there ([`crate::opinion`]).
//! Nothing after the slot is decoded.
//!
//! Why describe first: a read at the verdict slot before the model has
//! described the command carries nothing on this checkpoint (F9: AUC ~0.5
//! at 99% mass); after `effect`/`scope`/`undo` the same slot separates the
//! gold at AUC 0.74. The description is therefore part of the answer and is
//! echoed, so a caller can see and log what the read was conditioned on.
//!
//! What the response never carries: a winner. The caller picks, from its own
//! thresholds on its own data, reading the renormalised `prob` beside the
//! raw `sequence_mass` (a low mass means the model was never asked this
//! question here, and a renormalised number over it is noise that looks
//! like an answer). Questions come from the spec on disk, never from request
//! text: framing words in a rendered prompt were measured to move label
//! tokens by an order of magnitude, and a prompt the instruments never
//! scored is not one this daemon should serve.
//!
//! Vocabulary: this is the *opinion read* in code; "System 1" is discussion
//! language and belongs in the README.
use crate::adjudicator::{PrefixInfo, PromptSpec, validate_text};
use crate::opinion::OpinionRead;
use serde::{Deserialize, Serialize};

const MAX_STATE_BYTES: usize = 65536;

fn yes() -> bool {
    true
}
fn default_timeout() -> u64 {
    30000
}

/// The state an opinion is asked about. Rendered into the user turn exactly
/// as the evaluation harnesses render it, so a paired generative run and an
/// opinion read see the same bytes: the facts block verbatim (an app builds
/// it; the daemon never does), then `Command:` and the command.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpinionState {
    pub command: String,
    /// Evidence about the command, prepended verbatim. Ends with a newline
    /// if the caller wants one between it and `Command:`.
    #[serde(default)]
    pub facts: Option<String>,
}

impl OpinionState {
    pub fn render(&self) -> String {
        format!(
            "{}Command:\n{}",
            self.facts.as_deref().unwrap_or(""),
            self.command
        )
    }
    fn validate(&self) -> Result<(), String> {
        validate_text(&self.command).map_err(|e| format!("state.command: {e}"))?;
        if let Some(facts) = &self.facts
            && (facts.contains("<|") || facts.contains("<think>") || facts.contains("</think>"))
        {
            return Err("state.facts: literal model control tokens are not allowed".into());
        }
        if self.render().len() > MAX_STATE_BYTES {
            return Err(format!("state exceeds {MAX_STATE_BYTES} bytes"));
        }
        Ok(())
    }
}

/// One question: a choice field of the spec, optionally narrowed to a subset
/// of its options. The subset is scored in the spec's order, not the
/// request's.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Question {
    pub field: String,
    #[serde(default)]
    pub options: Option<Vec<String>>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpinionRequest {
    /// A loaded spec's name (`GET /v1/opinion/specs`).
    pub spec: String,
    pub state: OpinionState,
    /// Reserved for a shared block between the prefix and the state. v1
    /// accepts only `null`; anything else is refused so the field's meaning
    /// is settled before a caller depends on it.
    #[serde(default)]
    pub context: Option<serde_json::Value>,
    /// One or more choice fields. They share one description: the engine
    /// walks the grammar once and reads each slot as it passes it, so the
    /// answers come back in emission order whatever order they were asked.
    pub questions: Vec<Question>,
    /// Return the exact text the options continue (`rendered` on the
    /// response). Off by default: `rendered_sha256` is the audit trail, and
    /// the text carries the whole spec prefix on every request.
    #[serde(default)]
    pub rendered: bool,
    #[serde(default = "yes")]
    pub use_cache: bool,
    #[serde(default = "default_timeout")]
    pub timeout_ms: u64,
}

impl OpinionRequest {
    /// Shape checks that need no spec: the menu check is
    /// [`SpecMenuEntry::resolve`].
    pub fn validate(&self) -> Result<(), String> {
        if self.spec.is_empty() {
            return Err("spec must name a loaded prompt spec".into());
        }
        self.state.validate()?;
        if self.context.is_some() {
            return Err("context is reserved and must be null".into());
        }
        if self.questions.is_empty() {
            return Err("ask at least one question".into());
        }
        if self.timeout_ms == 0 || self.timeout_ms > 120000 {
            return Err("timeout_ms must be 1..=120000".into());
        }
        Ok(())
    }
}

/// What kind of field a spec's schema declares, as the menu reports it.
/// Only `choice` fields can be asked.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FieldKind {
    Text,
    Choice,
    Boolean,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FieldInfo {
    pub field: String,
    pub kind: FieldKind,
    /// The enum, in schema order; empty unless `kind` is `choice`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub options: Vec<String>,
}

/// One loaded spec as `GET /v1/opinion/specs` lists it. Read this at
/// runtime; never hard-code a field name or an option.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpecMenuEntry {
    pub spec: String,
    pub snapshot_id: String,
    /// How many described states the daemon keeps for this spec.
    pub described_cache_capacity: usize,
    /// The schema's fields in `required` order, which is emission order.
    pub fields: Vec<FieldInfo>,
}

/// A question checked against its spec: what the engine generates before
/// the slot, the options it scores there, and the bytes that close each one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedQuestion {
    pub field: String,
    /// Fields generated before the question, in order.
    pub describe: Vec<String>,
    /// Options to score, in spec order.
    pub options: Vec<String>,
    /// Written after each option: the grammar's separator when another field
    /// follows, the object's close when this is the last field.
    pub close: String,
}

impl ResolvedQuestion {
    /// The text the generation must end with to stand at the slot: the key
    /// as the grammar writes it (`serde_json` escaping, so a quote or a
    /// backslash in a field name still matches), the separator, the value's
    /// opening quote.
    pub fn slot_text(&self) -> String {
        format!(
            "{}: \"",
            serde_json::to_string(&self.field).expect("a string always serializes")
        )
    }
}

impl SpecMenuEntry {
    /// Read the menu entry off a spec. A spec without an output schema lists
    /// no fields and can answer no question.
    pub fn from_prompt(
        name: &str,
        prompt: &PromptSpec,
        snapshot_id: &str,
        described_cache_capacity: usize,
    ) -> Result<Self, String> {
        let mut fields = Vec::new();
        if let Some(schema) = &prompt.output_schema {
            let order = schema["required"]
                .as_array()
                .ok_or("output_schema has no required list")?;
            let properties = schema["properties"]
                .as_object()
                .ok_or("output_schema has no properties")?;
            for name in order {
                let name = name
                    .as_str()
                    .ok_or("required field names must be strings")?;
                let spec = properties
                    .get(name)
                    .ok_or_else(|| format!("required names a missing property {name:?}"))?;
                let (kind, options) = if spec["type"] == "boolean" {
                    (FieldKind::Boolean, vec![])
                } else if let Some(choices) = spec.get("enum").and_then(|e| e.as_array()) {
                    let options = choices
                        .iter()
                        .map(|c| c.as_str().map(str::to_string))
                        .collect::<Option<Vec<_>>>()
                        .ok_or_else(|| format!("field {name:?} has a non-string enum"))?;
                    (FieldKind::Choice, options)
                } else {
                    (FieldKind::Text, vec![])
                };
                fields.push(FieldInfo {
                    field: name.to_string(),
                    kind,
                    options,
                });
            }
        }
        Ok(Self {
            spec: name.to_string(),
            snapshot_id: snapshot_id.to_string(),
            described_cache_capacity,
            fields,
        })
    }

    /// Resolve every question and put them in emission order, the order the
    /// engine meets their slots. A field asked twice is refused: its two
    /// reads would be the same slot.
    pub fn resolve_all(&self, questions: &[Question]) -> Result<Vec<ResolvedQuestion>, String> {
        if questions.is_empty() {
            return Err("ask at least one question".into());
        }
        let mut resolved = Vec::with_capacity(questions.len());
        for question in questions {
            if resolved
                .iter()
                .any(|r: &ResolvedQuestion| r.field == question.field)
            {
                return Err(format!("field {:?} is asked more than once", question.field));
            }
            resolved.push(self.resolve(question)?);
        }
        resolved.sort_by_key(|r| r.describe.len());
        Ok(resolved)
    }

    /// Check a question against this spec and fix what the engine needs.
    pub fn resolve(&self, question: &Question) -> Result<ResolvedQuestion, String> {
        let position = self
            .fields
            .iter()
            .position(|f| f.field == question.field)
            .ok_or_else(|| format!("spec {:?} has no field {:?}", self.spec, question.field))?;
        let field = &self.fields[position];
        if field.kind != FieldKind::Choice {
            return Err(format!(
                "field {:?} is {:?}; only a choice field can be asked",
                question.field,
                serde_json::to_value(field.kind).map_err(|e| e.to_string())?
            ));
        }
        let options: Vec<String> = match &question.options {
            None => field.options.clone(),
            Some(subset) => {
                let mut seen = std::collections::BTreeSet::new();
                for option in subset {
                    if !field.options.contains(option) {
                        return Err(format!(
                            "field {:?} has no option {option:?}",
                            question.field
                        ));
                    }
                    if !seen.insert(option) {
                        return Err(format!("option {option:?} repeats"));
                    }
                }
                field
                    .options
                    .iter()
                    .filter(|o| seen.contains(o))
                    .cloned()
                    .collect()
            }
        };
        if options.len() < 2 {
            return Err("a question needs at least two options".into());
        }
        let last = position + 1 == self.fields.len();
        Ok(ResolvedQuestion {
            field: question.field.clone(),
            describe: self.fields[..position]
                .iter()
                .map(|f| f.field.clone())
                .collect(),
            options,
            close: if last { "\"}" } else { "\"," }.into(),
        })
    }
}

/// One described field, in emission order. A list rather than an object
/// because `serde_json` sorts object keys, and the order here is evidence.
#[derive(Clone, Debug, Serialize)]
pub struct DescribedField {
    pub field: String,
    pub value: serde_json::Value,
}

/// One question's read at its slot. `margin` is the renormalised gap between
/// the top two options — the one Jev-shaped number added here — and is blind
/// to "never asked", which is what `sequence_mass` beside it is for.
#[derive(Clone, Debug, Serialize)]
pub struct Answer {
    pub field: String,
    #[serde(flatten)]
    pub read: OpinionRead,
    pub margin: f32,
}

/// Each cache layer's outcome for this request: `hit`, `miss`, `bypass`
/// (the request asked for a cold path) or `skipped` (a described hit never
/// consults the prompt checkpoint).
#[derive(Clone, Debug, Serialize)]
pub struct CacheOutcome {
    /// The spec's resident system prefix.
    pub prefix: String,
    /// The exact rendered prompt, before any description.
    pub state: String,
    /// The prompt plus its generated description, at the slot.
    pub described: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct OpinionResponse {
    #[serde(flatten)]
    pub prefix: PrefixInfo,
    pub spec: String,
    /// The fields generated before the slot, as the model wrote them.
    pub described: Vec<DescribedField>,
    pub answers: Vec<Answer>,
    /// Only when the request asked: the rendered prompt, chat-template
    /// control tokens included, plus the description up to and including
    /// the slot — the bytes each answer's `rendered_sha256` hashes. Absent,
    /// not null, when unasked.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rendered: Option<String>,
    pub cache: CacheOutcome,
    /// Tokens in the rendered prompt (prefix and user turn), before the
    /// description.
    pub prompt_tokens: usize,
    /// How many of those were already resident.
    pub cached_tokens: usize,
    /// Tokens the description took, whether generated now or served from
    /// the described cache (`cache.described` says which).
    pub described_tokens: usize,
    pub queue_ms: f64,
    pub prefill_ms: f64,
    pub describe_ms: f64,
    pub read_ms: f64,
}

/// `prob[top] - prob[runner-up]` over renormalised option probabilities.
pub fn margin(probs: &[f32]) -> f32 {
    let mut sorted = probs.to_vec();
    sorted.sort_by(|a, b| b.total_cmp(a));
    match sorted.as_slice() {
        [top, next, ..] => top - next,
        [_] => 1.0,
        [] => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_text_is_the_key_as_the_grammar_writes_it() {
        let q = |field: &str| ResolvedQuestion {
            field: field.into(),
            describe: vec![],
            options: vec!["a".into(), "b".into()],
            close: "\"}".into(),
        };
        assert_eq!(q("verdict").slot_text(), "\"verdict\": \"");
        // A quote or a backslash in a field name is escaped exactly as the
        // grammar (serde_json) writes the key, so the stop still matches.
        assert_eq!(q("a\"b").slot_text(), "\"a\\\"b\": \"");
        assert_eq!(q("a\\b").slot_text(), "\"a\\\\b\": \"");
    }

    fn entry() -> SpecMenuEntry {
        let choice = |f: &str, o: &[&str]| FieldInfo {
            field: f.into(),
            kind: FieldKind::Choice,
            options: o.iter().map(|s| s.to_string()).collect(),
        };
        SpecMenuEntry {
            spec: "s".into(),
            snapshot_id: "x".into(),
            described_cache_capacity: 16,
            fields: vec![
                FieldInfo { field: "effect".into(), kind: FieldKind::Text, options: vec![] },
                choice("scope", &["nothing", "project", "home"]),
                choice("undo", &["easy", "hard"]),
                choice("verdict", &["allow", "ask", "review"]),
            ],
        }
    }

    #[test]
    fn several_questions_resolve_into_emission_order() {
        let q = |f: &str| Question { field: f.into(), options: None };
        let got = entry()
            .resolve_all(&[q("verdict"), q("scope"), q("undo")])
            .unwrap();
        let fields: Vec<&str> = got.iter().map(|r| r.field.as_str()).collect();
        assert_eq!(fields, ["scope", "undo", "verdict"], "answers follow the slots, not the request");
        assert_eq!(got[2].describe, ["effect", "scope", "undo"]);
        assert_eq!(got[2].close, "\"}");
        let err = entry().resolve_all(&[q("undo"), q("undo")]).unwrap_err();
        assert!(err.contains("more than once"), "{err}");
        assert!(entry().resolve_all(&[]).is_err());
        assert!(entry().resolve_all(&[q("scope"), q("effect")]).is_err(), "text is never asked");
    }

    #[test]
    fn margin_is_the_gap_between_the_top_two() {
        assert!((margin(&[0.7, 0.2, 0.1]) - 0.5).abs() < 1e-6);
        assert!((margin(&[0.1, 0.2, 0.7]) - 0.5).abs() < 1e-6);
        assert_eq!(margin(&[1.0]), 1.0);
        assert_eq!(margin(&[]), 0.0);
    }

    #[test]
    fn a_request_needs_at_least_one_question_and_no_context() {
        let parse = |s: &str| serde_json::from_str::<OpinionRequest>(s).unwrap();
        let good = parse(r#"{"spec":"s","state":{"command":"x"},"questions":[{"field":"v"}]}"#);
        assert!(good.validate().is_ok());
        assert!(good.use_cache && good.timeout_ms == 30000);
        let two = parse(
            r#"{"spec":"s","state":{"command":"x"},"questions":[{"field":"v"},{"field":"w"}]}"#,
        );
        assert!(two.validate().is_ok(), "several questions share one description");
        let none = parse(r#"{"spec":"s","state":{"command":"x"},"questions":[]}"#);
        assert!(none.validate().is_err());
        let ctx = parse(
            r#"{"spec":"s","state":{"command":"x"},"context":1,"questions":[{"field":"v"}]}"#,
        );
        assert!(ctx.validate().is_err());
        let null_ctx = parse(
            r#"{"spec":"s","state":{"command":"x"},"context":null,"questions":[{"field":"v"}]}"#,
        );
        assert!(null_ctx.validate().is_ok());
    }
}
