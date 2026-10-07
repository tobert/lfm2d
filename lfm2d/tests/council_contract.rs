//! The council wire contract, checked against kaijutsu's OpenAPI file.
//!
//! `tests/fixtures/council/components.json` is that file's schemas (see
//! `PROVENANCE.md`). Two directions, both of which must hold:
//!
//! - what we SEND validates against the contract's schema (`Error`, and the
//!   response shapes as they are written);
//! - what we ACCEPT agrees with the schema about the requests it describes:
//!   a body the schema allows parses, a body it forbids is refused.
//!
//! Each table row states the expected verdict by hand AND is checked against
//! the schema, so the table cannot drift from the oracle and the oracle
//! cannot drift from our parse unnoticed. Semantic rules the schema does not
//! express (limits that depend on the server, duplicate warm entries) live
//! in `council.rs`'s unit tests, not here.
use lfm2d::council::{ContextBody, CouncilError, CouncilSpec, Limits};
use serde_json::{Value, json};

fn components() -> Value {
    let text = include_str!("fixtures/council/components.json");
    serde_json::from_str::<Value>(text).expect("components.json parses")["components"].clone()
}

/// A validator for one named schema in the contract's `components.schemas`.
/// The whole `components` object rides along so local `$ref`s resolve.
fn schema(name: &str) -> jsonschema::Validator {
    let root = json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$ref": format!("#/components/schemas/{name}"),
        "components": components(),
    });
    jsonschema::validator_for(&root).unwrap_or_else(|e| panic!("schema {name}: {e}"))
}

fn limits() -> Limits {
    Limits::new(4096)
}

/// Does the schema allow `body`? On refusal, say why, for the failure message.
fn schema_says(name: &str, body: &Value) -> Result<(), String> {
    let validator = schema(name);
    let errors: Vec<String> = validator.iter_errors(body).map(|e| format!("{e} at {}", e.instance_path())).collect();
    if errors.is_empty() { Ok(()) } else { Err(errors.join("; ")) }
}

fn assert_valid(name: &str, body: &Value) {
    if let Err(why) = schema_says(name, body) {
        panic!("{name} refuses what we send: {why}\n{body}");
    }
}

// ------------------------------------------------------------------ errors

#[test]
fn every_error_we_send_is_the_contracts_error_body() {
    let errors = [
        CouncilError::bad_request("x"),
        CouncilError::bad_request("x").param("turns[0]"),
        CouncilError::not_found("x"),
        CouncilError::too_large("x", "turns"),
        CouncilError::precondition_failed(&format!("snap:{}", "a".repeat(64))),
        CouncilError::snapshot_gone(&format!("snap:{}", "a".repeat(64))),
        CouncilError::busy("x"),
        CouncilError::unavailable("x"),
        CouncilError::timeout("x"),
        CouncilError::pin_budget("x"),
        CouncilError::internal("x"),
    ];
    for e in errors {
        assert_valid("Error", &e.to_body());
    }
}

// ---------------------------------------------------------------- contexts

/// `(name, body, the contract allows it)`.
fn put_cases() -> Vec<(&'static str, Value, bool)> {
    let turn = |v: Value| json!({"system": "s", "turns": [v]});
    vec![
        ("empty turns", json!({"system": "s", "turns": []}), true),
        ("user turn", turn(json!({"role": "user", "content": "a"})), true),
        ("assistant turn with reasoning and snap",
            turn(json!({"role": "assistant", "content": "a", "reasoning": "why", "snap": true})), true),
        ("tool turn", turn(json!({"role": "tool", "content": "a"})), true),
        ("every optional field",
            json!({"system": "s", "turns": [], "pin": true, "dry_run": true, "persist": false,
                   "warm": [format!("sha256:{}", "a".repeat(64))]}), true),
        ("no system", json!({"turns": []}), false),
        ("no turns", json!({"system": "s"}), false),
        ("unknown field", json!({"system": "s", "turns": [], "from": "x"}), false),
        ("system role", turn(json!({"role": "system", "content": "a"})), false),
        ("turn without content", turn(json!({"role": "user"})), false),
        ("turn with an unknown field", turn(json!({"role": "user", "content": "a", "extra": 1})), false),
        ("reasoning on a user turn", turn(json!({"role": "user", "content": "a", "reasoning": "why"})), false),
        ("snap that is not a boolean", turn(json!({"role": "user", "content": "a", "snap": "yes"})), false),
        ("warm that is not a spec id", json!({"system": "s", "turns": [], "warm": ["verdict"]}), false),
        ("persist that is not a boolean", json!({"system": "s", "turns": [], "persist": "no"}), false),
    ]
}

#[test]
fn the_put_table_agrees_with_the_contracts_schema() {
    for (name, body, allowed) in put_cases() {
        assert_eq!(schema_says("ContextPut", &body).is_ok(), allowed, "the table is wrong about {name:?}: {body}");
    }
}

#[test]
fn a_put_is_accepted_exactly_when_the_contract_allows_it() {
    let mut disagreements = Vec::new();
    for (name, body, allowed) in put_cases() {
        let ours = serde_json::from_value::<ContextBody>(body.clone())
            .map_err(|e| e.to_string())
            .and_then(|b| b.validate(&limits()).map_err(|e| e.message));
        if ours.is_ok() != allowed {
            disagreements.push(format!("{name:?}: contract allowed={allowed}, we said {ours:?}"));
        }
    }
    assert!(disagreements.is_empty(), "{}", disagreements.join("\n"));
}

// ------------------------------------------------------------------- specs

fn spec(questions: Value) -> Value {
    json!({"name": "gate", "instructions": "Judge it.", "input_label": "Statement", "questions": questions})
}

fn choice(options: usize) -> Value {
    let criteria: Vec<Value> = (0..options).map(|n| json!({"option": format!("o{n}"), "means": "m"})).collect();
    json!({"id": "verdict", "type": "choice", "instructions": "What happens.", "criteria": criteria})
}

fn spec_cases() -> Vec<(&'static str, Value, bool)> {
    let levels = |n: usize| (0..n).map(|i| json!(format!("l{i}"))).collect::<Vec<_>>();
    vec![
        ("one choice", spec(json!([choice(2)])), true),
        ("every question type",
            spec(json!([
                {"id": "effect", "type": "text", "instructions": "What it does.", "max_tokens": 48},
                {"id": "undo", "type": "score", "instructions": "How hard.", "criteria": ["easy", "hard"]},
                choice(3),
                {"id": "novel", "type": "noul", "instructions": "Is it new?"},
            ])), true),
        ("noul with null criteria",
            spec(json!([{"id": "n", "type": "noul", "instructions": "x", "criteria": null}])), true),
        ("noul with object criteria",
            spec(json!([{"id": "n", "type": "noul", "instructions": ["a", "b"], "criteria": {"true": "t", "false": "f"}}])), true),
        ("score with ten levels",
            spec(json!([{"id": "s", "type": "score", "instructions": "x", "criteria": levels(10)}])), true),
        ("no questions", spec(json!([])), false),
        ("one choice option", spec(json!([choice(1)])), false),
        ("score with one level",
            spec(json!([{"id": "s", "type": "score", "instructions": "x", "criteria": levels(1)}])), false),
        ("score with eleven levels",
            spec(json!([{"id": "s", "type": "score", "instructions": "x", "criteria": levels(11)}])), false),
        ("text without max_tokens", spec(json!([{"id": "t", "type": "text", "instructions": "x"}])), false),
        ("unknown question type", spec(json!([{"id": "t", "type": "rank", "instructions": "x"}])), false),
        ("question id starting with a digit", spec(json!([{"id": "1q", "type": "noul", "instructions": "x"}])), false),
        ("name past 64", json!({"name": "n".repeat(65), "instructions": "i", "input_label": "l",
                                "questions": [choice(2)]}), false),
        ("input_label past 64", json!({"name": "n", "instructions": "i", "input_label": "l".repeat(65),
                                       "questions": [choice(2)]}), false),
        ("unknown spec field", json!({"name": "n", "instructions": "i", "input_label": "l",
                                      "questions": [choice(2)], "extra": 1}), false),
        ("choice option without means", spec(json!([{"id": "c", "type": "choice", "instructions": "x",
                                                     "criteria": [{"option": "a"}, {"option": "b"}]}])), false),
    ]
}

#[test]
fn the_spec_table_agrees_with_the_contracts_schema() {
    for (name, body, allowed) in spec_cases() {
        assert_eq!(schema_says("Spec", &body).is_ok(), allowed, "the table is wrong about {name:?}: {body}");
    }
}

#[test]
fn a_spec_is_accepted_exactly_when_the_contract_allows_it() {
    let mut disagreements = Vec::new();
    for (name, body, allowed) in spec_cases() {
        let ours = CouncilSpec::parse(&body, &limits());
        if ours.is_ok() != allowed {
            disagreements.push(format!("{name:?}: contract allowed={allowed}, we said {:?}", ours.err().map(|e| e.message)));
        }
    }
    assert!(disagreements.is_empty(), "{}", disagreements.join("\n"));
}
