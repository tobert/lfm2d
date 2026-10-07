//! Compile a council spec onto the engine's `PromptSpec`.
//!
//! The council spec is a question set (typed questions, in order); the
//! engine's spec is a prompt (a system text, an input label and an
//! `output_schema` whose `required` order is the emission order). This is
//! the one place the two meet, so what the model is shown is decided here
//! and nowhere else.
//!
//! The rendering is the council "v3" shape, reused as it stands (Amy,
//! 2026-10-07: "reuse v3 as is, we can iterate later"): the spec's
//! `instructions` are the system text, `input_label` is the label, each
//! question is one schema property in spec order, a text question is a free
//! string and every other question is an enum the engine reads at its slot.
//! `demo/web/static/council-{describe,verdict}-v3.json` compile from their
//! council-spec form byte for byte (`tests/council_compile.rs`).
//!
//! How the other question types map, because the engine can only ask a
//! choice (enum) field:
//!
//! - `choice`: an enum of its options, in order.
//! - `score`: an enum of its level names, lowest first; the answer's labels
//!   are the level numbers, and the score is computed from them.
//! - `noul`: an enum of `yes` and `no`; the answer is the probability of yes.
//!
//! Known deviations from the contract's prose (the wire shapes are exact;
//! these are about what the model is shown and how it is asked), each a
//! choice to revisit:
//!
//! - a question's id is the JSON key the model writes, as v3's `effect` and
//!   `verdict` are; the contract says ids are never shown to the model;
//! - every field before a slot is generated before it, so an earlier
//!   non-text answer (its greedy argmax) conditions a later question, where
//!   the contract says non-text answers never condition a later one;
//! - a text question's `max_tokens` is accepted and not enforced: the
//!   engine's schema has no per-field limit;
//! - a choice's `means` has no v3 slot. It is folded into the field's
//!   description, unmeasured; with no `means` anywhere the description is
//!   exactly the question's instructions, as in v3;
//! - empty `instructions`, and an `input_label` the engine cannot render
//!   (a colon, a newline, control text), are refused with a `400`.
//!
//! The `template` is a hash of [`COMPILE_RULES`]. `tests/council_compile.rs`
//! pins a golden compile, so a change to what the model is shown fails a
//! test until the rules text, and with it the template, is changed on
//! purpose.
use serde_json::{Map, Value, json};

use crate::adjudicator::PromptSpec;
use crate::council::{CouncilError, CouncilSpec, SpecQuestion, canonical_json};
use crate::hash::sha256_hex_bytes;

type Result<T> = std::result::Result<T, CouncilError>;

/// What a change to the compile has to change, so that it changes the
/// template. Written as the rules, not as a version number.
pub const COMPILE_RULES: &str = "council-compile-1: system=spec.instructions; input_label=spec.input_label; \
one string property per question keyed by its id, in spec order, all required; text: description=instructions; \
choice: enum=options, description=instructions + ' Options: ' + 'option (means)' joined by '; ' when any means is \
non-empty; score: enum=level names, description=instructions; noul: enum=[yes,no], description=instructions + \
' Guidance: ' + criteria text when criteria is not null; non-string instructions or criteria are canonical JSON";

/// The rendering's id: how this server compiles and renders a spec, part of
/// its identity.
pub fn template() -> String {
    format!("lfm25-council-v3:{}", &sha256_hex_bytes(COMPILE_RULES.as_bytes())[..16])
}

/// An instruction or criteria value as text: a string as it is, anything
/// else as canonical JSON (so the same value is the same bytes).
fn text_of(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => canonical_json(other),
    }
}

fn description(question: &SpecQuestion) -> String {
    match question {
        SpecQuestion::Text { instructions, .. } | SpecQuestion::Score { instructions, .. } => text_of(instructions),
        SpecQuestion::Choice { instructions, criteria, .. } => {
            let mut text = text_of(instructions);
            if criteria.iter().any(|c| !c.means.is_empty()) {
                let options: Vec<String> = criteria
                    .iter()
                    .map(|c| if c.means.is_empty() { c.option.clone() } else { format!("{} ({})", c.option, c.means) })
                    .collect();
                text.push_str(" Options: ");
                text.push_str(&options.join("; "));
            }
            text
        }
        SpecQuestion::Noul { instructions, criteria, .. } => {
            let mut text = text_of(instructions);
            if let Some(guidance) = criteria.as_ref().filter(|c| !c.is_null()) {
                text.push_str(" Guidance: ");
                text.push_str(&text_of(guidance));
            }
            text
        }
    }
}

/// The enum a question is read over, in order; `None` for a text question.
fn choices(question: &SpecQuestion) -> Option<Vec<String>> {
    match question {
        SpecQuestion::Text { .. } => None,
        SpecQuestion::Choice { criteria, .. } => Some(criteria.iter().map(|c| c.option.clone()).collect()),
        SpecQuestion::Score { criteria, .. } => Some(criteria.clone()),
        SpecQuestion::Noul { .. } => Some(vec!["yes".into(), "no".into()]),
    }
}

/// A council spec as the engine's prompt, and the template that names how.
#[derive(Clone, Debug)]
pub struct Compiled {
    pub prompt: PromptSpec,
    pub template: String,
}

pub fn compile(spec: &CouncilSpec) -> Result<Compiled> {
    if spec.instructions.trim().is_empty() {
        return Err(CouncilError::bad_request("this server needs instructions: the spec layer's text")
            .param("instructions"));
    }
    let mut properties = Map::new();
    let mut required = Vec::new();
    for question in &spec.questions {
        let mut property = Map::new();
        property.insert("type".into(), json!("string"));
        property.insert("description".into(), json!(description(question)));
        if let Some(options) = choices(question) {
            let distinct: std::collections::BTreeSet<&String> = options.iter().collect();
            if distinct.len() != options.len() {
                return Err(CouncilError::bad_request(format!("question {:?} repeats a name", question.id()))
                    .param(format!("questions.{}", question.id())));
            }
            property.insert("enum".into(), json!(options));
        }
        properties.insert(question.id().to_owned(), Value::Object(property));
        required.push(json!(question.id()));
    }
    let prompt: PromptSpec = serde_json::from_value(json!({
        "input_label": spec.input_label,
        "system": spec.instructions,
        "output_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": false,
        },
    }))
    .map_err(|e| CouncilError::internal(format!("the compiled spec is not a prompt spec: {e}")))?;
    prompt
        .render_prefix()
        .map_err(|e| CouncilError::bad_request(format!("this server cannot render the spec: {e}")))?;
    Ok(Compiled { prompt, template: template() })
}
