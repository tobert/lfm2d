//! `/council/v1/specs` at the wire, over a fake engine: what a client posts,
//! what it gets back, and what the engine was asked underneath. Bodies are
//! validated against the contract's schemas (`tests/fixtures/council/`).
//! The compile itself is `council_compile.rs`; the wire shapes are
//! `council_contract.rs`; this file is the routes, the council-spec <-> engine
//! prompt-spec bookkeeping (idempotence, sharing, eviction) and the errors.
use axum::body::Body;
use axum::http::Request;
use lfm2d::adjudicator::{
    AdjudicateRequest, AdjudicateResponse, Failure, Generator, Handle, PrefixInfo, PromptSpec, RegisterOutcome,
    UnregisterOutcome,
};
use lfm2d::council::{Limits, spec_id};
use lfm2d::council_compile::template;
use lfm2d::opinion_api::{OpinionRequest, OpinionResponse, ResolvedQuestion, SpecMenuEntry};
use serde_json::{Value, json};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use tower::ServiceExt;

fn info() -> PrefixInfo {
    PrefixInfo {
        model_id: "fixture".into(),
        weight_hash: "hash".into(),
        tokenizer_hash: "tok".into(),
        template_version: "test".into(),
        snapshot_id: "snapshot".into(),
        prefix_tokens: 7,
        input_cache_capacity: 1,
        context_limit: 4096,
        backend: "cpu".into(),
        device: "cpu".into(),
        candle_rev: "test".into(),
        dtype: "f32".into(),
        sampling: "greedy".into(),
        weight_dtypes: vec!["F32".into()],
    }
}

/// The text a compiled spec's system carries to make the fake's load fail.
const REFUSE: &str = "REFUSE this fixture";

struct Inner {
    boot: Vec<SpecMenuEntry>,
    uploaded: VecDeque<SpecMenuEntry>,
    capacity: usize,
    loads: usize,
    unregisters: usize,
}
impl Inner {
    fn menu(&self) -> Vec<SpecMenuEntry> {
        self.boot.iter().chain(self.uploaded.iter()).cloned().collect()
    }
}

/// A fake engine that keeps an LRU of uploaded prompt specs like the real one.
#[derive(Clone)]
struct Fake(Arc<Mutex<Inner>>);

impl Generator for Fake {
    fn generate(
        &mut self,
        _: &AdjudicateRequest,
        _: &dyn lfm2d::adjudicator::YieldPoint<Self>,
    ) -> Result<AdjudicateResponse, Failure> {
        Err(Failure::Internal("this fake does not generate".into()))
    }
    fn opine(
        &mut self,
        _: &OpinionRequest,
        _: &[ResolvedQuestion],
        _: &dyn Fn() -> Result<(), Failure>,
    ) -> Result<OpinionResponse, Failure> {
        Err(Failure::Internal("this fake does not read".into()))
    }
    fn register(
        &mut self,
        id: String,
        prompt: PromptSpec,
        check: &dyn Fn() -> Result<(), Failure>,
    ) -> Result<RegisterOutcome, Failure> {
        check()?;
        if prompt.system.contains(REFUSE) {
            return Err(Failure::Unprocessable(format!("cannot load: {}", prompt.system)));
        }
        let mut inner = self.0.lock().unwrap();
        if let Some(entry) = inner.boot.iter().find(|e| e.id == id).cloned() {
            let menu = inner.menu();
            return Ok(RegisterOutcome { entry, newly_loaded: false, evicted: None, load_ms: 0., menu });
        }
        if let Some(pos) = inner.uploaded.iter().position(|e| e.id == id) {
            let entry = inner.uploaded.remove(pos).unwrap();
            inner.uploaded.push_back(entry.clone());
            let menu = inner.menu();
            return Ok(RegisterOutcome { entry, newly_loaded: false, evicted: None, load_ms: 0., menu });
        }
        inner.loads += 1;
        let entry = SpecMenuEntry::from_prompt(&id, &id, &prompt, "snap", 16).expect("a compiled spec is a menu entry");
        let evicted = if inner.uploaded.len() >= inner.capacity { inner.uploaded.pop_front().map(|e| e.id) } else { None };
        inner.uploaded.push_back(entry.clone());
        let menu = inner.menu();
        Ok(RegisterOutcome { entry, newly_loaded: true, evicted, load_ms: 0.5, menu })
    }
    fn unregister(&mut self, id: &str) -> UnregisterOutcome {
        let mut inner = self.0.lock().unwrap();
        inner.unregisters += 1;
        if inner.boot.iter().any(|e| e.id == id) {
            return UnregisterOutcome::BootSpec;
        }
        match inner.uploaded.iter().position(|e| e.id == id) {
            Some(pos) => {
                let entry = inner.uploaded.remove(pos).unwrap();
                let menu = inner.menu();
                UnregisterOutcome::Deleted { entry, menu }
            }
            None => UnregisterOutcome::NotFound,
        }
    }
    fn probe(
        &mut self,
        _: &lfm2d::probe_api::ProbeRequest,
        _: &dyn Fn() -> Result<(), Failure>,
    ) -> Result<lfm2d::probe_api::ProbeResponse, Failure> {
        Err(Failure::Internal("this fake does not probe".into()))
    }
}

struct Harness {
    router: axum::Router,
    fake: Fake,
}

fn harness(capacity: usize, boot: Vec<SpecMenuEntry>) -> Harness {
    let fake = Fake(Arc::new(Mutex::new(Inner { boot: boot.clone(), uploaded: VecDeque::new(), capacity, loads: 0, unregisters: 0 })));
    let handle = Handle::spawn(fake.clone(), (&info()).into()).with_menu(boot);
    Harness { router: lfm2d::council_api::router(handle, (&info()).into()), fake }
}

async fn send(router: &axum::Router, request: Request<Body>) -> (u16, Value, String) {
    let response = router.clone().oneshot(request).await.unwrap();
    let status = response.status().as_u16();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let text = String::from_utf8_lossy(&bytes).into_owned();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null), text)
}

async fn post(router: &axum::Router, body: &Value) -> (u16, Value, String) {
    send(router, Request::post("/council/v1/specs").body(Body::from(body.to_string())).unwrap()).await
}
async fn post_raw(router: &axum::Router, body: Vec<u8>) -> (u16, Value, String) {
    send(router, Request::post("/council/v1/specs").body(Body::from(body)).unwrap()).await
}
async fn get(router: &axum::Router, path: &str) -> (u16, Value, String) {
    send(router, Request::get(path).body(Body::empty()).unwrap()).await
}
async fn delete(router: &axum::Router, path: &str) -> (u16, Value, String) {
    send(router, Request::delete(path).body(Body::empty()).unwrap()).await
}

// ---- the contract's schemas, as the oracle

fn schema(name: &str) -> jsonschema::Validator {
    let text = include_str!("fixtures/council/components.json");
    let components = serde_json::from_str::<Value>(text).unwrap()["components"].clone();
    let root = json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$ref": format!("#/components/schemas/{name}"),
        "components": components,
    });
    jsonschema::validator_for(&root).unwrap()
}
fn assert_valid(name: &str, body: &Value) {
    let errors: Vec<String> = schema(name).iter_errors(body).map(|e| format!("{e} at {}", e.instance_path())).collect();
    assert!(errors.is_empty(), "{name} refuses: {}\n{body}", errors.join("; "));
}
fn assert_error(status: u16, body: &Value, want: u16, kind: &str) {
    assert_eq!(status, want, "{body}");
    assert_valid("Error", body);
    assert_eq!(body["error"]["type"], kind, "{body}");
}

// ---- specs

fn gate(name: &str) -> Value {
    json!({
        "name": name, "instructions": "Judge the statement.", "input_label": "Statement",
        "questions": [
            {"id": "effect", "type": "text", "instructions": "What it does.", "max_tokens": 48},
            {"id": "verdict", "type": "choice", "instructions": "What happens.",
             "criteria": [{"option": "allow", "means": "routine"}, {"option": "ask", "means": "outward"}]},
        ]
    })
}

#[tokio::test]
async fn posting_a_spec_holds_it_under_its_canonical_id_and_echoes_it() {
    let h = harness(4, vec![]);
    let body = gate("gate");
    let (status, held, _) = post(&h.router, &body).await;
    assert_eq!(status, 200, "{held}");
    assert_valid("HeldSpec", &held);
    assert_eq!(held["spec_id"], spec_id(&body));
    assert_eq!(held["spec"], body, "the spec echoes as submitted");
    assert_eq!(held["template"], template());

    let (status, got, _) = get(&h.router, &format!("/council/v1/specs/{}", spec_id(&body))).await;
    assert_eq!((status, &got), (200, &held));
}

#[tokio::test]
async fn the_same_spec_is_the_same_id_and_loads_once() {
    let h = harness(4, vec![]);
    let (_, first, _) = post(&h.router, &gate("gate")).await;
    let (status, second, _) = post(&h.router, &gate("gate")).await;
    assert_eq!(status, 200);
    assert_eq!(first, second);
    assert_eq!(h.fake.0.lock().unwrap().loads, 1, "a re-submission is a no-op for the engine");
    let (_, list, _) = get(&h.router, "/council/v1/specs").await;
    assert_eq!(list["specs"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn engine_ids_never_reach_the_wire() {
    let h = harness(4, vec![]);
    let (_, held, text) = post(&h.router, &gate("gate")).await;
    let engine_id = h.fake.0.lock().unwrap().uploaded[0].id.clone();
    assert_eq!(engine_id.len(), 64);
    assert!(!text.contains(&engine_id), "the engine's content id is internal: {held}");
    let (_, _, list) = get(&h.router, "/council/v1/specs").await;
    assert!(!list.contains(&engine_id));
}

#[tokio::test]
async fn the_list_is_the_contracts_and_in_id_order() {
    let h = harness(4, vec![]);
    for name in ["c", "a", "b"] {
        post(&h.router, &gate(name)).await;
    }
    let (status, list, _) = get(&h.router, "/council/v1/specs").await;
    assert_eq!(status, 200);
    let specs = list["specs"].as_array().unwrap();
    assert_eq!(specs.len(), 3);
    for spec in specs {
        assert_valid("HeldSpec", spec);
    }
    let ids: Vec<&str> = specs.iter().map(|s| s["spec_id"].as_str().unwrap()).collect();
    let mut sorted = ids.clone();
    sorted.sort();
    assert_eq!(ids, sorted, "stable, by id");
    // An empty list is still the contract's object.
    let (_, empty, _) = get(&harness(4, vec![]).router, "/council/v1/specs").await;
    assert_eq!(empty, json!({"specs": []}));
}

#[tokio::test]
async fn specs_that_compile_alike_share_one_engine_spec_and_delete_by_reference() {
    // `name` is not in the prompt, so these two are one engine spec.
    let h = harness(4, vec![]);
    let (_, a, _) = post(&h.router, &gate("alpha")).await;
    let (_, b, _) = post(&h.router, &gate("beta")).await;
    assert_ne!(a["spec_id"], b["spec_id"], "two council specs");
    assert_eq!(h.fake.0.lock().unwrap().loads, 1, "one engine spec");

    let (status, _, _) = delete(&h.router, &format!("/council/v1/specs/{}", a["spec_id"].as_str().unwrap())).await;
    assert_eq!(status, 204);
    assert_eq!(h.fake.0.lock().unwrap().unregisters, 0, "beta still needs the engine's spec");
    let (status, got, _) = get(&h.router, &format!("/council/v1/specs/{}", b["spec_id"].as_str().unwrap())).await;
    assert_eq!((status, &got), (200, &b));

    let (status, _, _) = delete(&h.router, &format!("/council/v1/specs/{}", b["spec_id"].as_str().unwrap())).await;
    assert_eq!(status, 204);
    assert_eq!(h.fake.0.lock().unwrap().unregisters, 1, "the last holder lets the engine drop it");
    assert!(h.fake.0.lock().unwrap().uploaded.is_empty());
}

#[tokio::test]
async fn deleting_a_spec_is_a_204_then_a_404() {
    let h = harness(4, vec![]);
    let (_, held, _) = post(&h.router, &gate("gate")).await;
    let path = format!("/council/v1/specs/{}", held["spec_id"].as_str().unwrap());
    let (status, body, text) = delete(&h.router, &path).await;
    assert_eq!((status, text.as_str(), &body), (204, "", &Value::Null));
    let (status, body, _) = delete(&h.router, &path).await;
    assert_error(status, &body, 404, "not_found");
    let (status, body, _) = get(&h.router, &path).await;
    assert_error(status, &body, 404, "not_found");
}

#[tokio::test]
async fn unknown_and_malformed_ids_are_404() {
    let h = harness(4, vec![]);
    for id in [format!("sha256:{}", "0".repeat(64)), "nope".into(), format!("sha256:{}", "A".repeat(64)), "sha256:ab".into()] {
        let path = format!("/council/v1/specs/{id}");
        for (status, body, _) in [get(&h.router, &path).await, delete(&h.router, &path).await] {
            assert_error(status, &body, 404, "not_found");
        }
    }
}

#[tokio::test]
async fn an_evicted_engine_spec_is_a_spec_that_is_gone() {
    // Capacity one: the second upload evicts the first compiled prompt.
    let h = harness(1, vec![]);
    let (_, first, _) = post(&h.router, &gate("first")).await;
    let other = {
        let mut other = gate("second");
        other["instructions"] = json!("A different judge.");
        other
    };
    let (_, second, _) = post(&h.router, &other).await;
    let (status, body, _) = get(&h.router, &format!("/council/v1/specs/{}", first["spec_id"].as_str().unwrap())).await;
    assert_error(status, &body, 404, "not_found");
    let (_, list, _) = get(&h.router, "/council/v1/specs").await;
    assert_eq!(list["specs"], json!([second]), "only what the engine still holds is listed");
    // And it can be posted again.
    let (status, again, _) = post(&h.router, &gate("first")).await;
    assert_eq!((status, &again), (200, &first));
}

#[tokio::test]
async fn a_spec_that_matches_a_boot_spec_drops_without_touching_the_engines() {
    // Boot a prompt spec that is exactly what `gate` compiles to.
    let compiled = lfm2d::council_compile::compile(
        &lfm2d::council::CouncilSpec::parse(&gate("gate"), &Limits::new(4096)).unwrap(),
    )
    .unwrap();
    let bytes = serde_json::to_vec(&compiled.prompt).unwrap();
    let id = lfm2d::hash::sha256_hex_bytes(&bytes);
    let boot = SpecMenuEntry::from_prompt(&id, "boot", &compiled.prompt, "snap", 16).unwrap();
    let h = harness(4, vec![boot]);
    let (_, held, _) = post(&h.router, &gate("gate")).await;
    let path = format!("/council/v1/specs/{}", held["spec_id"].as_str().unwrap());
    let (status, _, _) = delete(&h.router, &path).await;
    assert_eq!(status, 204);
    assert_eq!(h.fake.0.lock().unwrap().boot.len(), 1, "the boot spec is still there");
    let (status, body, _) = get(&h.router, &path).await;
    assert_error(status, &body, 404, "not_found");
}

// ---- refusals

#[tokio::test]
async fn bad_bodies_are_400_in_the_contracts_error_shape() {
    let h = harness(4, vec![]);
    let mutate = |f: &dyn Fn(&mut Value)| {
        let mut v = gate("gate");
        f(&mut v);
        v
    };
    let (status, body, _) = post_raw(&h.router, b"not json".to_vec()).await;
    assert_error(status, &body, 400, "invalid_request");
    for (what, spec) in [
        ("an unknown field", mutate(&|v| v["extra"] = json!(1))),
        ("no questions", mutate(&|v| v["questions"] = json!([]))),
        ("a float in the spec", mutate(&|v| v["questions"][0]["max_tokens"] = json!(1.5))),
        ("a colon in the label", mutate(&|v| v["input_label"] = json!("A: b"))),
        ("empty instructions", mutate(&|v| v["instructions"] = json!(""))),
    ] {
        let (status, body, _) = post(&h.router, &spec).await;
        assert_error(status, &body, 400, "invalid_request");
        assert!(body["error"]["message"].as_str().is_some_and(|m| !m.is_empty()), "{what}: {body}");
    }
    assert_eq!(h.fake.0.lock().unwrap().loads, 0, "nothing reached the engine");
    let (_, list, _) = get(&h.router, "/council/v1/specs").await;
    assert_eq!(list, json!({"specs": []}));
}

#[tokio::test]
async fn an_oversized_body_is_a_413_in_the_contracts_shape() {
    let h = harness(4, vec![]);
    let mut spec = gate("gate");
    spec["instructions"] = json!("x".repeat(1_048_576));
    let (status, body, _) = post(&h.router, &spec).await;
    assert_error(status, &body, 413, "too_large");
    assert_eq!(body["error"]["param"], "body");
}

#[tokio::test]
async fn an_engine_refusal_is_a_400_carrying_the_engines_words() {
    let h = harness(4, vec![]);
    let mut spec = gate("gate");
    spec["instructions"] = json!(REFUSE);
    let (status, body, _) = post(&h.router, &spec).await;
    assert_error(status, &body, 400, "invalid_request");
    assert!(body["error"]["message"].as_str().unwrap().contains(REFUSE), "{body}");
    let (_, list, _) = get(&h.router, "/council/v1/specs").await;
    assert_eq!(list, json!({"specs": []}), "a refused spec is not held");
}

// ---- identity

#[tokio::test]
async fn identity_is_the_contracts_and_says_what_this_server_is() {
    let h = harness(4, vec![]);
    let (status, id, _) = get(&h.router, "/council/v1/identity").await;
    assert_eq!(status, 200, "{id}");
    assert_valid("ServerIdentity", &id);
    assert_eq!(id["model"], "fixture");
    assert_eq!((id["weight_hash"].as_str(), id["tokenizer_hash"].as_str()), (Some("hash"), Some("tok")));
    assert_eq!(id["device"], "cpu");
    assert_eq!(id["template"], template(), "the identity and a held spec name the same rendering");
    assert_eq!(id["limits"]["context_tokens"], 4096, "the engine's own context limit");
    assert_eq!(id["limits"]["contexts_per_decision"], 8);
    assert!(id["engine"].as_str().unwrap().starts_with("lfm2d/"), "{id}");
    assert!(id["text_stop"].as_str().is_some_and(|s| !s.is_empty()), "describe is listed, so text_stop is");
}

#[tokio::test]
async fn identity_lists_only_the_capabilities_this_server_has() {
    let h = harness(4, vec![]);
    let (_, id, _) = get(&h.router, "/council/v1/identity").await;
    assert_eq!(id["capabilities"], json!(["dry_run", "describe", "leave_one_out"]));
    // Not claimed: nothing here parks or warms, and persist is tolerated, not kept.
    for not in ["park", "warm", "persist"] {
        assert!(!id["capabilities"].as_array().unwrap().iter().any(|c| c == not), "{not}");
    }
}

#[tokio::test]
async fn identity_does_not_change_between_reads() {
    let h = harness(4, vec![]);
    let (_, a, _) = get(&h.router, "/council/v1/identity").await;
    post(&h.router, &gate("gate")).await;
    let (_, b, _) = get(&h.router, "/council/v1/identity").await;
    assert_eq!(a, b);
}
