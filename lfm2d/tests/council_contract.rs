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
use lfm2d::council::{
    ContextBody, ControlHit, CouncilError, CouncilPool, CouncilSpec, Limits, Ordered, PoolMethod, PoolWeights,
    SpecQuestion, pooled_numbers, read_answer, read_numbers, spec_id,
};
use lfm2d::council_wire::{
    Capability, ContextPutResult, ContextState, DecisionIdentity, DecisionResponse, HeldSpec, Layer, PoolEcho, Read,
    ServerIdentity, Signals, SnapshotId, SnapshotInfo, Usage, pooled_answer,
};
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

// --------------------------------------------------------------- responses

const A: &str = "0199b3c4-6c1e-7a2b-9f00-3e5d1c2a7b11";
const B: &str = "0199b3c4-6c1e-7a2b-9f00-3e5d1c2a7b12";

fn snap(n: u8) -> SnapshotId {
    SnapshotId::from_digest(&format!("{n:02x}").repeat(32)).unwrap()
}

fn gate_body() -> Value {
    json!({
        "name": "shell-gate", "instructions": "Judge it.", "input_label": "Proposed statement",
        "questions": [
            {"id": "effect", "type": "text", "instructions": "What it does.", "max_tokens": 48},
            {"id": "undo", "type": "score", "instructions": "How hard.", "criteria": ["easy", "hard", "impossible"]},
            {"id": "verdict", "type": "choice", "instructions": "What happens.",
             "criteria": [{"option": "zebra", "means": "z"}, {"option": "apple", "means": "a"},
                          {"option": "mango", "means": "m"}]},
            {"id": "novel", "type": "noul", "instructions": "Is it new?"},
        ]
    })
}

/// One context's raw log probabilities per non-text question, in spec order.
fn logprobs(context: usize) -> Vec<(&'static str, Vec<f64>)> {
    let lp = |p: &[f64], shift: f64| p.iter().map(|x| x.ln() + shift).collect::<Vec<_>>();
    match context {
        0 => vec![("undo", lp(&[0.2, 0.7, 0.1], -0.05)), ("verdict", lp(&[0.1, 0.8, 0.1], -0.02)), ("novel", lp(&[0.3, 0.7], -0.1))],
        _ => vec![("undo", lp(&[0.6, 0.3, 0.1], -0.2)), ("verdict", lp(&[0.5, 0.2, 0.3], -0.3)), ("novel", lp(&[0.9, 0.1], -0.01))],
    }
}

/// A whole decision built through the real constructors.
fn decision(contexts: &[&str], pool: CouncilPool, leave_one_out: bool) -> DecisionResponse {
    let spec = CouncilSpec::parse(&gate_body(), &limits()).unwrap();
    let settings = pool.settings(contexts.len().max(1)).unwrap();
    let ids: Vec<String> = contexts.iter().map(|c| c.to_string()).collect();
    let questions: Vec<&SpecQuestion> = spec.questions.iter().filter(|q| !q.labels().is_empty()).collect();
    let mut reads = Vec::new();
    for c in 0..contexts.len().max(1) {
        let answers: Vec<(String, _)> = logprobs(c)
            .into_iter()
            .map(|(id, lp)| {
                let q = questions.iter().find(|q| q.id() == id).unwrap();
                (id.to_string(), read_answer(q, &lp).unwrap())
            })
            .collect();
        reads.push(Read {
            context: contexts.get(c).map(|c| c.to_string()),
            snapshot: contexts.get(c).map(|_| snap(c as u8 + 1)),
            described: Some(Ordered(vec![("effect".into(), "Publishes local commits.".into())])),
            answers: Ordered(answers),
            rendered_sha256: "e3".repeat(32),
            tokens: Some(96),
            ms: Some(12.5),
        });
    }
    let mut answers = Vec::new();
    let mut normalized = Vec::new();
    for q in &questions {
        let per_read: Vec<_> = (0..contexts.len().max(1))
            .map(|c| {
                let lp = &logprobs(c).into_iter().find(|(id, _)| *id == q.id()).unwrap().1;
                read_numbers(&q.labels(), lp).unwrap()
            })
            .collect();
        let pooled = pooled_numbers(&per_read, &settings).unwrap();
        normalized.push((q.id().to_string(), pooled.weights.clone()));
        answers.push((q.id().to_string(), pooled_answer(q, &pooled, &ids, leave_one_out).unwrap()));
    }
    DecisionResponse {
        model: "lfm2.5-8b-a1b".into(),
        answers: Ordered(answers),
        reads,
        pool: PoolEcho { method: pool.method, weights: pool.weights, normalized: Ordered(normalized) },
        signals: Some(Signals { control_text: vec![ControlHit { r#where: "state".into(), token: "<|im_end|>".into() }] }),
        identity: DecisionIdentity {
            model: "lfm2.5-8b-a1b".into(),
            weight_hash: "w".repeat(8),
            tokenizer_hash: "t".repeat(8),
            template: "lfm25-describe-1:0123456789abcdef".into(),
            engine: "lfm2d/0.0.1".into(),
            spec_id: Some(spec_id(&gate_body())),
        },
        usage: Some(Usage { input_tokens: 41, output_tokens: 26, fed_tokens: 212 }),
        queue_ms: Some(0.5),
        ms: Some(240.0),
    }
}

fn to_value(r: &DecisionResponse) -> Value {
    serde_json::to_value(r).unwrap()
}

#[test]
fn a_decision_over_two_contexts_is_the_contracts_decision_response() {
    for (method, weights) in [
        (PoolMethod::Linear, PoolWeights::Uniform),
        (PoolMethod::Loglinear, PoolWeights::Mass),
    ] {
        let pool = CouncilPool { method, weights, values: None };
        assert_valid("DecisionResponse", &to_value(&decision(&[A, B], pool, true)));
    }
}

#[test]
fn a_decision_with_given_weights_may_carry_a_null_leave_one_out() {
    // Weights [1, 0]: without the first read nothing weighted remains.
    let pool = CouncilPool { method: PoolMethod::Linear, weights: PoolWeights::Given, values: Some(vec![1.0, 0.0]) };
    let v = to_value(&decision(&[A, B], pool, true));
    assert_valid("DecisionResponse", &v);
    assert!(v["answers"]["verdict"]["leave_one_out"][A].is_null(), "no weighted read left without {A}");
    assert!(v["answers"]["verdict"]["leave_one_out"][B].is_object());
    assert!(v["answers"]["novel"]["leave_one_out"][A].is_null());
    assert!(v["answers"]["novel"]["leave_one_out"][B].is_number(), "a noul's row is one number");
}

#[test]
fn a_plain_decision_with_no_contexts_has_one_read_with_null_context_and_snapshot() {
    let v = to_value(&decision(&[], CouncilPool::default(), true));
    assert_valid("DecisionResponse", &v);
    assert_eq!(v["reads"].as_array().unwrap().len(), 1);
    assert!(v["reads"][0]["context"].is_null() && v["reads"][0]["snapshot"].is_null());
    assert!(v["answers"]["verdict"].get("leave_one_out").is_none(), "one read has no leave-one-out");
    assert_eq!((v["answers"]["verdict"]["agree"].as_bool(), v["answers"]["verdict"]["spread"].as_f64()), (Some(true), Some(0.0)));
}

#[test]
fn leave_one_out_appears_only_when_the_server_lists_the_capability() {
    let v = to_value(&decision(&[A, B], CouncilPool::default(), false));
    assert_valid("DecisionResponse", &v);
    for q in ["undo", "verdict", "novel"] {
        assert!(v["answers"][q].get("leave_one_out").is_none(), "{q}");
    }
}

#[test]
fn options_keep_the_specs_order_not_the_alphabets() {
    let text = serde_json::to_string(&decision(&[A, B], CouncilPool::default(), true)).unwrap();
    let verdict = &text[text.find("\"verdict\":{\"type\":\"choice\"").unwrap()..];
    // From the probabilities object: the argmax `choice` value precedes it and names `apple`.
    let verdict = &verdict[verdict.find("\"probabilities\":{").unwrap()..];
    let (z, a, m) = (verdict.find("\"zebra\"").unwrap(), verdict.find("\"apple\"").unwrap(), verdict.find("\"mango\"").unwrap());
    assert!(z < a && a < m, "zebra, apple, mango as the spec lists them: {verdict}");
}

#[test]
fn the_oracle_refuses_a_response_that_is_wrong() {
    let good = to_value(&decision(&[A, B], CouncilPool::default(), true));
    assert_valid("DecisionResponse", &good);
    let mutations: [(&str, &dyn Fn(&mut Value)); 5] = [
        ("no identity", &|v| { v.as_object_mut().unwrap().remove("identity"); }),
        ("no engine", &|v| { v["identity"].as_object_mut().unwrap().remove("engine"); }),
        ("confidence past 1", &|v| v["answers"]["verdict"]["confidence"] = json!(1.5)),
        ("a positive logprob", &|v| v["reads"][0]["answers"]["verdict"]["logprobs"]["zebra"] = json!(0.5)),
        ("a read with no rendered_sha256", &|v| { v["reads"][0].as_object_mut().unwrap().remove("rendered_sha256"); }),
    ];
    for (name, mutate) in mutations {
        let mut v = good.clone();
        mutate(&mut v);
        assert!(schema_says("DecisionResponse", &v).is_err(), "the schema let {name} through");
    }
}

// ---------------------------------------------------------- the other bodies

#[test]
fn server_identity_is_the_contracts_identity() {
    let identity = ServerIdentity {
        model: "lfm2.5-8b-a1b".into(),
        aliases: vec!["lfm25".into()],
        weight_hash: "w".into(),
        tokenizer_hash: "t".into(),
        template: "lfm25-describe-1:0123456789abcdef".into(),
        engine: "lfm2d/0.0.1".into(),
        device: Some("rocm:gfx1151".into()),
        text_stop: Some("the closing quote of the field".into()),
        limits: limits(),
        capabilities: vec![Capability::Describe, Capability::LeaveOneOut],
    };
    let v = serde_json::to_value(&identity).unwrap();
    assert_valid("ServerIdentity", &v);
    assert_eq!(v["capabilities"], json!(["describe", "leave_one_out"]));
    // Optional members are absent, not null.
    let bare = ServerIdentity { aliases: vec![], device: None, text_stop: None, ..identity };
    let v = serde_json::to_value(&bare).unwrap();
    assert_valid("ServerIdentity", &v);
    assert!(v.get("aliases").is_none() && v.get("device").is_none() && v.get("text_stop").is_none());
}

fn state() -> ContextState {
    ContextState {
        id: A.into(),
        head: snap(3),
        tokens: 96,
        pinned: Some(false),
        persist: None,
        snapshots: vec![
            SnapshotInfo { id: snap(1), tokens: 12, layer: Layer::System, turn: None, spec_id: None, bytes: Some(4096), parked: None, pinned: None },
            SnapshotInfo { id: snap(2), tokens: 60, layer: Layer::Turn, turn: Some(0), spec_id: None, bytes: None, parked: None, pinned: Some(false) },
            SnapshotInfo { id: snap(3), tokens: 96, layer: Layer::Spec, turn: None, spec_id: Some(spec_id(&gate_body())), bytes: None, parked: Some(false), pinned: None },
        ],
    }
}

#[test]
fn a_context_state_and_a_put_result_are_the_contracts() {
    assert_valid("ContextState", &serde_json::to_value(state()).unwrap());
    for dry_run in [false, true] {
        let result = ContextPutResult { state: state(), kept: 60, fed: 36, dry_run };
        let v = serde_json::to_value(&result).unwrap();
        assert_valid("ContextPutResult", &v);
        assert_eq!((v["kept"].as_u64(), v["fed"].as_u64(), v["dry_run"].as_bool()), (Some(60), Some(36), Some(dry_run)));
        assert_eq!(v["head"], v["snapshots"][2]["id"], "the state is flattened beside kept and fed");
    }
}

#[test]
fn a_held_spec_is_the_contracts_and_its_id_is_the_canonical_hash() {
    let body = gate_body();
    let held = HeldSpec {
        spec_id: spec_id(&body),
        spec: CouncilSpec::parse(&body, &limits()).unwrap(),
        template: "lfm25-describe-1:0123456789abcdef".into(),
    };
    let v = serde_json::to_value(&held).unwrap();
    assert_valid("HeldSpec", &v);
    assert_eq!(v["spec"], body, "the spec echoes as submitted");
}

#[test]
fn snapshot_ids_are_snap_and_64_hex() {
    assert!(SnapshotId::from_digest(&"ab".repeat(32)).is_ok());
    for bad in ["", "ab", &"AB".repeat(32), &"zz".repeat(32), &"ab".repeat(33)] {
        assert!(SnapshotId::from_digest(bad).is_err(), "{bad:?}");
    }
    let id = snap(7);
    assert_eq!(SnapshotId::parse(id.as_str()).unwrap(), id);
    for bad in ["snap:", "sha256:abab", &"ab".repeat(32), &format!("snap:{}", "AB".repeat(32))] {
        assert_eq!(SnapshotId::parse(bad).unwrap_err().status, 400, "{bad:?}");
    }
}
