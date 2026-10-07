//! The council spec compiles onto the engine's prompt spec, and the v3
//! rendering is reproduced exactly: `demo/web/static/council-*-v3.json` are
//! the shipped, measured specs, and their council-spec form must compile to
//! them. A change to what the model is shown fails here until
//! `COMPILE_RULES` (and with it the template) is changed on purpose.
use lfm2d::council::{CouncilSpec, Limits};
use lfm2d::council_compile::{COMPILE_RULES, compile, template};
use serde_json::{Value, json};

fn limits() -> Limits {
    Limits::new(4096)
}

fn v3_file(name: &str) -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../demo/web/static/");
    let text = std::fs::read_to_string(format!("{path}{name}")).unwrap_or_else(|e| panic!("{name}: {e}"));
    serde_json::from_str(&text).unwrap()
}

fn parse(body: Value) -> CouncilSpec {
    CouncilSpec::parse(&body, &limits()).unwrap_or_else(|e| panic!("{}", e.message))
}

fn compiled_json(spec: &CouncilSpec) -> Value {
    serde_json::to_value(compile(spec).unwrap_or_else(|e| panic!("{}", e.message)).prompt).unwrap()
}

/// The v3 shape as council specs: the file's own words.
fn v3_as_council(file: &Value, with_text: bool) -> Value {
    let schema = &file["output_schema"];
    let mut questions: Vec<Value> = Vec::new();
    for key in schema["required"].as_array().unwrap() {
        let key = key.as_str().unwrap();
        let prop = &schema["properties"][key];
        let instructions = prop["description"].clone();
        if let Some(options) = prop.get("enum") {
            let criteria: Vec<Value> =
                options.as_array().unwrap().iter().map(|o| json!({"option": o, "means": ""})).collect();
            questions.push(json!({"id": key, "type": "choice", "instructions": instructions, "criteria": criteria}));
        } else if with_text {
            questions.push(json!({"id": key, "type": "text", "instructions": instructions, "max_tokens": 48}));
        }
    }
    json!({
        "name": "council-v3", "instructions": file["system"], "input_label": file["input_label"],
        "questions": questions,
    })
}

#[test]
fn the_describe_v3_spec_compiles_to_itself() {
    let file = v3_file("council-describe-v3.json");
    let ours = compiled_json(&parse(v3_as_council(&file, true)));
    assert_eq!(ours["system"], file["system"]);
    assert_eq!(ours["input_label"], file["input_label"]);
    assert_eq!(ours["output_schema"], file["output_schema"], "the schema, `required` order included");
    assert_eq!(
        ours["output_schema"]["required"],
        json!(["effect", "this_source_says", "verdict"]),
        "emission order is spec order"
    );
}

#[test]
fn the_verdict_v3_spec_compiles_to_itself() {
    let file = v3_file("council-verdict-v3.json");
    let ours = compiled_json(&parse(v3_as_council(&file, false)));
    assert_eq!((&ours["system"], &ours["input_label"]), (&file["system"], &file["input_label"]));
    assert_eq!(ours["output_schema"], file["output_schema"]);
}

/// Every other question type, with means, guidance and structured
/// instructions: a golden. If this fails the model is being shown something
/// else, so change `COMPILE_RULES` (the template) deliberately.
#[test]
fn the_other_question_types_compile_to_a_golden() {
    let spec = parse(json!({
        "name": "gate", "instructions": "Judge the statement.", "input_label": "Statement",
        "questions": [
            {"id": "effect", "type": "text", "instructions": "What it does.", "max_tokens": 48},
            {"id": "undo", "type": "score", "instructions": {"ask": "How hard to undo"}, "criteria": ["easy", "hard", "impossible"]},
            {"id": "verdict", "type": "choice", "instructions": "What happens.",
             "criteria": [{"option": "allow", "means": "routine"}, {"option": "ask", "means": ""},
                          {"option": "report", "means": "ask, but louder"}]},
            {"id": "novel", "type": "noul", "instructions": "Is it new?", "criteria": ["seen before", "common"]},
            {"id": "plain", "type": "noul", "instructions": "Is it local?", "criteria": null},
        ]
    }));
    let ours = compiled_json(&spec);
    assert_eq!(
        ours,
        json!({
            "input_label": "Statement",
            "system": "Judge the statement.",
            "tools": [],
            "reasoning": "closed",
            "opinion": null,
            "output_schema": {
                "type": "object",
                "properties": {
                    "effect": {"type": "string", "description": "What it does."},
                    "undo": {"type": "string", "description": "{\"ask\":\"How hard to undo\"}",
                             "enum": ["easy", "hard", "impossible"]},
                    "verdict": {"type": "string",
                                "description": "What happens. Options: allow (routine); ask; report (ask, but louder)",
                                "enum": ["allow", "ask", "report"]},
                    "novel": {"type": "string", "description": "Is it new? Guidance: [\"seen before\",\"common\"]",
                              "enum": ["yes", "no"]},
                    "plain": {"type": "string", "description": "Is it local?", "enum": ["yes", "no"]},
                },
                "required": ["effect", "undo", "verdict", "novel", "plain"],
                "additionalProperties": false,
            },
        })
    );
}

#[test]
fn the_template_names_the_rules() {
    let t = template();
    assert!(t.starts_with("lfm25-council-v3:") && t.len() == "lfm25-council-v3:".len() + 16, "{t}");
    assert!(COMPILE_RULES.starts_with("council-compile-1"));
    assert_eq!(t, template(), "a template is a function of the rules alone");
}

/// What this server refuses that the contract allows, as a `400`; each is a
/// deviation named in `council_compile.rs`. Empty instructions fail in the
/// compile; a label the engine cannot render fails at parse.
#[test]
fn what_the_engine_cannot_render_is_refused_loudly() {
    let body = |instructions: &str, label: &str| {
        json!({
            "name": "g", "instructions": instructions, "input_label": label,
            "questions": [{"id": "v", "type": "noul", "instructions": "x"}],
        })
    };
    let e = compile(&parse(body("", "Label"))).expect_err("empty instructions");
    assert_eq!((e.status, e.param.as_deref()), (400, Some("instructions")));
    for (what, label) in [
        ("a colon in the label", "Label: x"),
        ("a newline in the label", "A\nB"),
        ("control text in the label", "<|im_start|>"),
    ] {
        let e = CouncilSpec::parse(&body("Judge.", label), &limits()).expect_err(what);
        assert_eq!(e.status, 400, "{what}");
    }
}

#[test]
fn a_score_with_a_repeated_level_name_is_refused() {
    let spec = parse(json!({
        "name": "g", "instructions": "Judge.", "input_label": "Label",
        "questions": [{"id": "s", "type": "score", "instructions": "x", "criteria": ["same", "same"]}],
    }));
    let e = compile(&spec).expect_err("two levels with one name cannot be told apart");
    assert_eq!((e.status, e.param.as_deref()), (400, Some("questions.s")));
}
