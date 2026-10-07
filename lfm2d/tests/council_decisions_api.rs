//! `POST /council/v1/decisions` at the wire, over a fake engine that holds
//! specs and contexts the way the real one does and answers reads from
//! scripted preferences, so contexts can be made to disagree. Bodies are
//! validated against the contract's schemas, and the pooled numbers are
//! checked against the reads they came from. The real model is the
//! engine's own tests' job (`council_contexts_real.rs`, `opinion_real.rs`).
use axum::body::Body;
use axum::http::Request;
use lfm2d::adjudicator::{
    AdjudicateRequest, AdjudicateResponse, Failure, Generator, Handle, PrefixInfo, PromptSpec, RegisterOutcome,
    UnregisterOutcome, YieldPoint,
};
use lfm2d::council::{ContextBody, render_segments};
use lfm2d::council_context::{BuildOutcome, BuildRequest, BuiltSnapshot, HeldInfo};
use lfm2d::opinion::{OpinionRead, OptionScore};
use lfm2d::opinion_api::{
    Answer, CacheOutcome, DescribedField, OpinionRequest, OpinionResponse, ResolvedQuestion, SpecMenuEntry,
};
use serde_json::{Value, json};
use std::collections::{HashMap, VecDeque};
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
        context_limit: 100_000,
        backend: "cpu".into(),
        device: "cpu".into(),
        candle_rev: "test".into(),
        dtype: "f32".into(),
        sampling: "greedy".into(),
        weight_dtypes: vec!["F32".into()],
    }
}

// ------------------------------------------------------------ the fake engine

struct Item {
    id: String,
    tokens: usize,
    pinned: bool,
}
struct Inner {
    items: Vec<Item>,
    capacity: usize,
    specs: VecDeque<SpecMenuEntry>,
    /// Which option index a context's reads prefer, by the context's engine id.
    prefs: HashMap<String, usize>,
    /// What the engine was asked, in order.
    seen: Vec<(OpinionRequest, Vec<ResolvedQuestion>)>,
}
#[derive(Clone)]
struct Fake(Arc<Mutex<Inner>>);

fn engine_id(text: &str) -> String {
    lfm2d::hash::sha256_hex_bytes(format!("fake\0{text}").as_bytes())
}

impl Generator for Fake {
    fn generate(&mut self, _: &AdjudicateRequest, _: &dyn YieldPoint<Self>) -> Result<AdjudicateResponse, Failure> {
        Err(Failure::Internal("not used".into()))
    }
    fn opine(
        &mut self,
        request: &OpinionRequest,
        questions: &[ResolvedQuestion],
        _: &dyn Fn() -> Result<(), Failure>,
    ) -> Result<OpinionResponse, Failure> {
        let mut inner = self.0.lock().unwrap();
        inner.seen.push((request.clone(), questions.to_vec()));
        let checkpoint = request.context.as_ref().map(|c| c.checkpoint.clone());
        let context_tokens = match &checkpoint {
            Some(id) => Some(inner.items.iter().find(|i| &i.id == id).ok_or_else(|| Failure::NotFound(format!("unknown read context {id}")))?.tokens),
            None => None,
        };
        let pref = checkpoint.as_ref().and_then(|id| inner.prefs.get(id)).copied().unwrap_or(0);
        let answers = questions
            .iter()
            .map(|q| {
                let options: Vec<OptionScore> = q
                    .options
                    .iter()
                    .enumerate()
                    .map(|(k, o)| OptionScore {
                        option: o.clone(),
                        logprob: if k == pref % q.options.len() { -0.2 } else { -3.5 },
                        first_logprob: -0.1,
                        prob: 0.5,
                        tokens: vec![1],
                    })
                    .collect();
                Answer {
                    field: q.field.clone(),
                    read: OpinionRead {
                        options,
                        sequence_mass: -0.05,
                        first_token_mass: -0.05,
                        shared_tokens: 1,
                        scored_tokens: 1,
                        rendered_sha256: lfm2d::hash::sha256_hex_bytes(
                            format!("{checkpoint:?}|{}", q.field).as_bytes(),
                        ),
                    },
                    margin: 0.5,
                }
            })
            .collect();
        let described = questions
            .last()
            .map(|q| {
                q.describe
                    .iter()
                    .map(|f| DescribedField { field: f.clone(), value: json!(format!("text for {f}")) })
                    .collect()
            })
            .unwrap_or_default();
        let tokens = context_tokens.unwrap_or(0);
        Ok(OpinionResponse {
            prefix: info(),
            spec: request.spec.clone(),
            context: request.context.clone(),
            context_tokens,
            described,
            answers,
            rendered: None,
            rendered_token_ids: None,
            cache: CacheOutcome { prefix: "tail".into(), state: "miss".into(), described: "miss".into() },
            prompt_tokens: tokens + 25,
            cached_tokens: tokens,
            described_tokens: 7,
            queue_ms: 0.,
            prefill_ms: 1.,
            describe_ms: 2.,
            read_ms: 3.,
        })
    }
    fn register(
        &mut self,
        id: String,
        prompt: PromptSpec,
        _: &dyn Fn() -> Result<(), Failure>,
    ) -> Result<RegisterOutcome, Failure> {
        let mut inner = self.0.lock().unwrap();
        let menu = |inner: &Inner| inner.specs.iter().cloned().collect::<Vec<_>>();
        if let Some(entry) = inner.specs.iter().find(|e| e.id == id).cloned() {
            let menu = menu(&inner);
            return Ok(RegisterOutcome { entry, newly_loaded: false, evicted: None, load_ms: 0., menu });
        }
        let entry = SpecMenuEntry::from_prompt(&id, &id, &prompt, "snap", 16).expect("a compiled spec is a menu entry");
        inner.specs.push_back(entry.clone());
        let menu = menu(&inner);
        Ok(RegisterOutcome { entry, newly_loaded: true, evicted: None, load_ms: 0., menu })
    }
    fn unregister(&mut self, _: &str) -> UnregisterOutcome {
        UnregisterOutcome::NotFound
    }
    fn probe(
        &mut self,
        _: &lfm2d::probe_api::ProbeRequest,
        _: &dyn Fn() -> Result<(), Failure>,
    ) -> Result<lfm2d::probe_api::ProbeResponse, Failure> {
        Err(Failure::Internal("not used".into()))
    }
    fn council_put(&mut self, request: &BuildRequest, _: &dyn YieldPoint<Self>) -> Result<BuildOutcome, Failure> {
        request.validate().map_err(Failure::BadRequest)?;
        let mut inner = self.0.lock().unwrap();
        let mut ends = Vec::new();
        let mut total = 0;
        for s in &request.segments {
            total += s.len();
            ends.push(total);
        }
        let id_at = |b: usize| engine_id(&request.segments[..=b].concat());
        let base = request.hold_after.iter().rev().copied().find(|&b| inner.items.iter().any(|i| i.id == id_at(b)));
        let kept = base.map_or(0, |b| ends[b]);
        for &b in request.hold_after.iter().filter(|&&b| !request.dry_run && base.is_none_or(|k| b > k)) {
            while inner.items.len() >= inner.capacity {
                let Some(at) = inner.items.iter().position(|i| !i.pinned) else {
                    return Err(Failure::InsufficientStorage("every held context is pinned".into()));
                };
                inner.items.remove(at);
            }
            inner.items.push(Item { id: id_at(b), tokens: ends[b], pinned: false });
        }
        let snapshots = request
            .hold_after
            .iter()
            .map(|&b| {
                let id = id_at(b);
                let item = inner.items.iter().find(|i| i.id == id);
                BuiltSnapshot {
                    after_segment: b,
                    tokens: ends[b],
                    held: item.is_some(),
                    bytes: item.map_or(0, |i| i.tokens * 10),
                    pinned: item.is_some_and(|i| i.pinned),
                    engine_id: id,
                }
            })
            .collect();
        Ok(BuildOutcome { snapshots, tokens: total, kept, fed: total - kept, prefill_ms: 0.1 })
    }
    fn council_inspect(&mut self, ids: &[String]) -> Result<Vec<Option<HeldInfo>>, Failure> {
        let inner = self.0.lock().unwrap();
        Ok(ids
            .iter()
            .map(|id| {
                inner.items.iter().find(|i| &i.id == id).map(|i| HeldInfo { tokens: i.tokens, bytes: i.tokens * 10, pinned: i.pinned })
            })
            .collect())
    }
    fn council_unpin(&mut self, _: &str) -> Result<bool, Failure> {
        Ok(true)
    }
}

// ---------------------------------------------------------------- the harness

struct Harness {
    router: axum::Router,
    fake: Fake,
}
fn harness(capacity: usize) -> Harness {
    let fake = Fake(Arc::new(Mutex::new(Inner {
        items: vec![],
        capacity,
        specs: VecDeque::new(),
        prefs: HashMap::new(),
        seen: vec![],
    })));
    let handle = Handle::spawn(fake.clone(), (&info()).into()).with_menu(vec![]);
    Harness { router: lfm2d::council_api::router(handle, (&info()).into()), fake }
}

type Reply = (u16, Value, String);
async fn send(router: &axum::Router, request: Request<Body>) -> Reply {
    let response = router.clone().oneshot(request).await.unwrap();
    let status = response.status().as_u16();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null), String::from_utf8_lossy(&bytes).into_owned())
}
async fn post_spec(r: &axum::Router, body: &Value) -> Reply {
    send(r, Request::post("/council/v1/specs").body(Body::from(body.to_string())).unwrap()).await
}
async fn put_ctx(r: &axum::Router, id: &str, body: &Value) -> Reply {
    send(r, Request::put(format!("/council/v1/contexts/{id}")).body(Body::from(body.to_string())).unwrap()).await
}
async fn get_ctx(r: &axum::Router, id: &str) -> Reply {
    send(r, Request::get(format!("/council/v1/contexts/{id}")).body(Body::empty()).unwrap()).await
}
async fn decide(r: &axum::Router, body: &Value) -> Reply {
    decide_raw(r, body.to_string()).await
}
async fn decide_raw(r: &axum::Router, body: String) -> Reply {
    send(r, Request::post("/council/v1/decisions").body(Body::from(body)).unwrap()).await
}

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
fn assert_error(reply: &Reply, want: u16, kind: &str) {
    assert_eq!(reply.0, want, "{}", reply.1);
    assert_valid("Error", &reply.1);
    assert_eq!(reply.1["error"]["type"], kind, "{}", reply.1);
}

const A: &str = "0199b3c4-6c1e-7a2b-9f00-3e5d1c2a7b11";
const B: &str = "0199b3c4-6c1e-7a2b-9f00-3e5d1c2a7b12";
const C: &str = "0199b3c4-6c1e-7a2b-9f00-3e5d1c2a7b13";

fn gate() -> Value {
    json!({
        "name": "gate", "instructions": "Judge the statement.", "input_label": "Statement",
        "questions": [
            {"id": "effect", "type": "text", "instructions": "What it does.", "max_tokens": 48},
            {"id": "undo", "type": "score", "instructions": "How hard to undo.", "criteria": ["easy", "hard", "impossible"]},
            {"id": "verdict", "type": "choice", "instructions": "What happens.",
             "criteria": [{"option": "allow", "means": "routine"}, {"option": "ask", "means": "outward"},
                          {"option": "report", "means": "louder"}]},
            {"id": "novel", "type": "noul", "instructions": "Is it new?"},
        ]
    })
}
fn ctx(system: &str, turns: &[&str]) -> Value {
    let turns: Vec<Value> = turns.iter().map(|t| json!({"role": "user", "content": t, "snap": true})).collect();
    json!({"system": system, "turns": turns})
}
/// The fake engine's id for the head of a context body.
fn head_id(body: &Value) -> String {
    let body: ContextBody = serde_json::from_value(body.clone()).unwrap();
    engine_id(&render_segments(&body).unwrap().concat())
}

struct Ready {
    h: Harness,
    spec_id: String,
}
/// A held spec and two held contexts.
async fn ready() -> Ready {
    let h = harness(64);
    let (_, spec, _) = post_spec(&h.router, &gate()).await;
    put_ctx(&h.router, A, &ctx("Memory.", &["Never push."])).await;
    put_ctx(&h.router, B, &ctx("User.", &["Ask before posting."])).await;
    Ready { spec_id: spec["spec_id"].as_str().unwrap().into(), h }
}
fn seen(h: &Harness) -> Vec<(OpinionRequest, Vec<ResolvedQuestion>)> {
    h.fake.0.lock().unwrap().seen.clone()
}

// ------------------------------------------------------------------ the read

#[tokio::test]
async fn a_decision_over_two_contexts_is_the_contracts_and_pools_what_the_reads_say() {
    let r = ready().await;
    {
        let mut inner = r.h.fake.0.lock().unwrap();
        inner.prefs.insert(head_id(&ctx("Memory.", &["Never push."])), 2); // report
        inner.prefs.insert(head_id(&ctx("User.", &["Ask before posting."])), 1); // ask
    }
    let (status, v, reply_text) = decide(
        &r.h.router,
        &json!({"spec_id": r.spec_id, "state": "git push origin main",
                "contexts": [{"id": A}, {"id": B}], "pool": {"method": "linear", "weights": "uniform"}}),
    )
    .await;
    assert_eq!(status, 200, "{v}");
    assert_valid("DecisionResponse", &v);

    // Answers: the non-text questions, in spec order. (Through the raw text:
    // parsing into a Value sorts the keys.)
    let text = &reply_text;
    let answers = &text[text.find("\"answers\"").unwrap()..];
    let at = |key: &str| answers.find(&format!("\"{key}\":{{\"type\"")).unwrap_or_else(|| panic!("{key} in {answers}"));
    assert!(at("undo") < at("verdict") && at("verdict") < at("novel"), "spec order, not the alphabet's");
    assert!(!answers[..at("novel")].contains("\"effect\":{\"type\""), "a text question is written, not an answer");

    // One read per context, in request order, each at its context's head.
    let reads = v["reads"].as_array().unwrap();
    assert_eq!(reads.len(), 2);
    for (read, id) in reads.iter().zip([A, B]) {
        assert_eq!(read["context"], id);
        assert_eq!(read["snapshot"], get_ctx(&r.h.router, id).await.1["head"], "a read names the snapshot it started from");
    }

    // The pooled verdict is the mean of the reads' own probabilities.
    let verdict = &v["answers"]["verdict"];
    for option in ["allow", "ask", "report"] {
        let mean = reads.iter().map(|rd| rd["answers"]["verdict"]["probabilities"][option].as_f64().unwrap()).sum::<f64>() / 2.0;
        assert!((verdict["probabilities"][option].as_f64().unwrap() - mean).abs() < 1e-9, "{option}");
    }
    assert_eq!(verdict["agree"], false, "the contexts read different options");
    assert_eq!(reads[0]["answers"]["verdict"]["choice"], "report");
    assert_eq!(reads[1]["answers"]["verdict"]["choice"], "ask");
    let loo = verdict["leave_one_out"].as_object().unwrap();
    assert_eq!(loo.keys().map(String::as_str).collect::<std::collections::BTreeSet<_>>(), [A, B].into_iter().collect());
    assert_eq!(v["pool"]["method"], "linear");
    assert_eq!(v["pool"]["normalized"]["verdict"], json!([0.5, 0.5]));

    // What this decision says about itself.
    assert_eq!(v["identity"]["spec_id"], r.spec_id);
    assert_eq!(v["model"], "fixture");
    assert!(v["usage"]["input_tokens"].as_u64().unwrap() > 0 && v["usage"]["fed_tokens"].as_u64().unwrap() > 0);

    // What the engine was asked: the held spec, the case, the contexts as
    // engine ids in order, and the non-text questions.
    let asks = seen(&r.h);
    assert_eq!(asks.len(), 2, "one read per context, serially");
    for ((request, questions), ctx_body) in asks.iter().zip([ctx("Memory.", &["Never push."]), ctx("User.", &["Ask before posting."])]) {
        assert_eq!(request.state.input, "git push origin main");
        assert_eq!(request.context.as_ref().unwrap().checkpoint, head_id(&ctx_body));
        assert_eq!(questions.iter().map(|q| q.field.as_str()).collect::<Vec<_>>(), ["undo", "verdict", "novel"]);
    }
}

#[tokio::test]
async fn a_decision_with_no_contexts_is_a_plain_read_of_the_spec() {
    let r = ready().await;
    let (status, v, _) = decide(&r.h.router, &json!({"spec_id": r.spec_id, "state": "ls -la"})).await;
    assert_eq!(status, 200, "{v}");
    assert_valid("DecisionResponse", &v);
    let reads = v["reads"].as_array().unwrap();
    assert_eq!(reads.len(), 1);
    assert!(reads[0]["context"].is_null() && reads[0]["snapshot"].is_null());
    assert!(v["answers"]["verdict"].get("leave_one_out").is_none());
    assert_eq!((v["answers"]["verdict"]["agree"].as_bool(), v["answers"]["verdict"]["spread"].as_f64()), (Some(true), Some(0.0)));
    assert!(seen(&r.h)[0].0.context.is_none() && seen(&r.h)[0].0.contexts.is_none(), "the spec's own prompt");
}

#[tokio::test]
async fn text_answers_are_described_by_each_read_and_only_text_ids_appear() {
    let r = ready().await;
    let (_, v, _) = decide(&r.h.router, &json!({"spec_id": r.spec_id, "state": "x", "contexts": [{"id": A}, {"id": B}]})).await;
    for read in v["reads"].as_array().unwrap() {
        assert_eq!(read["described"], json!({"effect": "text for effect"}), "the engine also generated undo before verdict; it is not a text answer");
    }
    // Ask only the verdict: undo is still generated before it, and still not a text answer.
    let (_, v, _) = decide(&r.h.router, &json!({"spec_id": r.spec_id, "state": "x", "ask": ["verdict"], "contexts": [{"id": A}]})).await;
    assert_eq!(v["reads"][0]["described"], json!({"effect": "text for effect"}));
}

#[tokio::test]
async fn an_object_state_is_json_in_the_requests_member_order() {
    let r = ready().await;
    let body = format!(r#"{{"spec_id": "{}", "state": {{"b": 1, "a": [2, 3], "c": "x"}}}}"#, r.spec_id);
    let (status, v, _) = decide_raw(&r.h.router, body).await;
    assert_eq!(status, 200, "{v}");
    assert_eq!(seen(&r.h)[0].0.state.input, r#"{"b": 1, "a": [2, 3], "c": "x"}"#, "b before a: the request's order, not the alphabet's");
}

// -------------------------------------------------------------- ask, options

#[tokio::test]
async fn ask_reads_a_subset_in_spec_order() {
    let r = ready().await;
    let (status, v, _) = decide(&r.h.router, &json!({"spec_id": r.spec_id, "state": "x", "ask": ["novel", "undo"]})).await;
    assert_eq!(status, 200, "{v}");
    assert_valid("DecisionResponse", &v);
    let asks = seen(&r.h);
    assert_eq!(asks[0].1.iter().map(|q| q.field.as_str()).collect::<Vec<_>>(), ["undo", "novel"], "spec order, whatever the request's");
    let answers = v["answers"].as_object().unwrap();
    assert!(answers.contains_key("undo") && answers.contains_key("novel") && !answers.contains_key("verdict"));
    assert!(v["reads"][0]["answers"].as_object().unwrap().get("verdict").is_none());
}

#[tokio::test]
async fn options_narrow_a_choice_and_the_numbers_follow() {
    let r = ready().await;
    let (status, v, _) = decide(
        &r.h.router,
        &json!({"spec_id": r.spec_id, "state": "x", "ask": ["verdict"], "options": {"verdict": ["report", "allow"]}}),
    )
    .await;
    assert_eq!(status, 200, "{v}");
    assert_valid("DecisionResponse", &v);
    let asks = seen(&r.h);
    assert_eq!(asks[0].1[0].options, ["allow", "report"], "the engine scores the subset in the spec's order");
    let probabilities = v["answers"]["verdict"]["probabilities"].as_object().unwrap();
    assert_eq!(probabilities.keys().cloned().collect::<Vec<_>>(), ["allow", "report"]);
    assert!((probabilities.values().map(|p| p.as_f64().unwrap()).sum::<f64>() - 1.0).abs() < 1e-9);
}

#[tokio::test]
async fn ask_and_options_the_spec_cannot_serve_are_400s_naming_the_field() {
    let r = ready().await;
    let sid = &r.spec_id;
    for (what, body, param) in [
        ("an unknown question", json!({"spec_id": sid, "state": "x", "ask": ["nope"]}), "ask"),
        ("a text question", json!({"spec_id": sid, "state": "x", "ask": ["effect"]}), "ask"),
        ("a repeated question", json!({"spec_id": sid, "state": "x", "ask": ["undo", "undo"]}), "ask"),
        ("no question", json!({"spec_id": sid, "state": "x", "ask": []}), "ask"),
        ("options on a score", json!({"spec_id": sid, "state": "x", "options": {"undo": ["easy", "hard"]}}), "options.undo"),
        ("an option the spec lacks", json!({"spec_id": sid, "state": "x", "options": {"verdict": ["allow", "nope"]}}), "options.verdict"),
        ("one option", json!({"spec_id": sid, "state": "x", "options": {"verdict": ["allow"]}}), "options.verdict"),
        ("a repeated option", json!({"spec_id": sid, "state": "x", "options": {"verdict": ["allow", "allow"]}}), "options.verdict"),
        ("options on a question not asked", json!({"spec_id": sid, "state": "x", "ask": ["undo"], "options": {"verdict": ["allow", "ask"]}}), "options.verdict"),
    ] {
        let reply = decide(&r.h.router, &body).await;
        assert_error(&reply, 400, "invalid_request");
        assert_eq!(reply.1["error"]["param"], param, "{what}");
    }
    assert!(seen(&r.h).is_empty(), "nothing reached the engine");
}

// ---------------------------------------------------------------------- pool

#[tokio::test]
async fn the_pool_is_echoed_and_weights_are_normalized_per_question() {
    let r = ready().await;
    for pool in [
        json!({"method": "loglinear", "weights": "mass"}),
        json!({"method": "linear", "weights": "given", "values": [3.0, 1.0]}),
    ] {
        let (status, v, _) = decide(
            &r.h.router,
            &json!({"spec_id": r.spec_id, "state": "x", "contexts": [{"id": A}, {"id": B}], "pool": pool}),
        )
        .await;
        assert_eq!(status, 200, "{v}");
        assert_valid("DecisionResponse", &v);
        assert_eq!(v["pool"]["method"], pool["method"]);
        assert_eq!(v["pool"]["weights"], pool["weights"]);
    }
    let (_, v, _) = decide(
        &r.h.router,
        &json!({"spec_id": r.spec_id, "state": "x", "contexts": [{"id": A}, {"id": B}],
                "pool": {"weights": "given", "values": [3.0, 1.0]}}),
    )
    .await;
    assert_eq!(v["pool"]["normalized"]["verdict"], json!([0.75, 0.25]));
}

#[tokio::test]
async fn a_pool_that_cannot_be_run_is_a_400() {
    let r = ready().await;
    let sid = &r.spec_id;
    for (what, pool) in [
        ("given without values", json!({"weights": "given"})),
        ("values without given", json!({"weights": "mass", "values": [1.0, 1.0]})),
        ("the wrong number of values", json!({"weights": "given", "values": [1.0]})),
        ("negative values", json!({"weights": "given", "values": [1.0, -1.0]})),
        ("an unknown method", json!({"method": "geometric"})),
    ] {
        let reply = decide(&r.h.router, &json!({"spec_id": sid, "state": "x", "contexts": [{"id": A}, {"id": B}], "pool": pool})).await;
        assert_error(&reply, 400, "invalid_request");
        let _ = what;
    }
}

// ------------------------------------------------------------- the contexts

#[tokio::test]
async fn at_reads_a_named_snapshot_and_the_head_is_the_default() {
    let r = ready().await;
    let (_, first, _) = put_ctx(&r.h.router, C, &ctx("Memory.", &["one"])).await;
    let old_head = first["head"].as_str().unwrap().to_owned();
    // Different content from the first turn on: content-addressed, an
    // unchanged first turn would be the very snapshot the first version's head was.
    let (_, second, _) = put_ctx(&r.h.router, C, &ctx("Memory.", &["changed", "two"])).await;
    assert_ne!(old_head, second["head"]);

    // The default is the head now.
    decide(&r.h.router, &json!({"spec_id": r.spec_id, "state": "x", "contexts": [{"id": C}]})).await;
    assert_eq!(seen(&r.h).last().unwrap().0.context.as_ref().unwrap().checkpoint, head_id(&ctx("Memory.", &["changed", "two"])));

    // The system snapshot is still one of this context's, and reading `at` it reads it.
    let system = second["snapshots"][0]["id"].as_str().unwrap();
    let (status, v, _) = decide(&r.h.router, &json!({"spec_id": r.spec_id, "state": "x", "contexts": [{"id": C, "at": system}]})).await;
    assert_eq!(status, 200, "{v}");
    assert_eq!(v["reads"][0]["snapshot"], system);
    assert_ne!(v["reads"][0]["snapshot"], second["head"]);

    // A snapshot this context never had is a 409 naming its current head.
    let reply = decide(&r.h.router, &json!({"spec_id": r.spec_id, "state": "x", "contexts": [{"id": C, "at": old_head}]})).await;
    assert_error(&reply, 409, "snapshot_gone");
    assert_eq!(reply.1["error"]["head"], second["head"]);
}

#[tokio::test]
async fn a_snapshot_the_engine_evicted_is_a_409_and_an_evicted_head_is_a_context_that_is_gone() {
    let h = harness(3);
    let (_, spec, _) = post_spec(&h.router, &gate()).await;
    let sid = spec["spec_id"].as_str().unwrap();
    let (_, a, _) = put_ctx(&h.router, A, &ctx("Memory.", &["one"])).await;
    let system = a["snapshots"][0]["id"].as_str().unwrap().to_owned();
    // B's two snapshots push A's oldest (its system head) out of a store of three.
    put_ctx(&h.router, B, &ctx("User.", &["two"])).await;

    let reply = decide(&h.router, &json!({"spec_id": sid, "state": "x", "contexts": [{"id": A, "at": system}]})).await;
    assert_error(&reply, 409, "snapshot_gone");
    assert_eq!(reply.1["error"]["head"], a["head"]);

    // Evict A's head as well: no `at` named, so the context itself is gone.
    put_ctx(&h.router, C, &ctx("Third.", &["three"])).await;
    let reply = decide(&h.router, &json!({"spec_id": sid, "state": "x", "contexts": [{"id": A}]})).await;
    assert_error(&reply, 404, "not_found");
    assert!(seen(&Harness { router: h.router.clone(), fake: h.fake.clone() }).is_empty(), "the engine was never asked");
}

#[tokio::test]
async fn a_decision_changes_nothing_it_read() {
    let r = ready().await;
    let before = (get_ctx(&r.h.router, A).await.1, r.h.fake.0.lock().unwrap().items.len());
    decide(&r.h.router, &json!({"spec_id": r.spec_id, "state": "x", "contexts": [{"id": A}, {"id": B}]})).await;
    assert_eq!((get_ctx(&r.h.router, A).await.1, r.h.fake.0.lock().unwrap().items.len()), before);
}

// ------------------------------------------------------------------ refusals

#[tokio::test]
async fn what_cannot_be_read_is_refused_in_the_contracts_error_shape() {
    let r = ready().await;
    let sid = &r.spec_id;
    let unknown_spec = format!("sha256:{}", "0".repeat(64));
    let many: Vec<Value> = (0..9).map(|n| json!({"id": format!("0199b3c4-6c1e-7a2b-9f00-3e5d1c2a7b{n:02}")})).collect();
    let cases: Vec<(&str, Value, u16, &str, Option<&str>)> = vec![
        ("an unknown spec", json!({"spec_id": unknown_spec, "state": "x"}), 404, "not_found", Some("spec_id")),
        ("a malformed spec id", json!({"spec_id": "nope", "state": "x"}), 400, "invalid_request", Some("spec_id")),
        ("no spec", json!({"state": "x"}), 400, "invalid_request", Some("spec_id")),
        ("inline questions", json!({"state": "x", "questions": {"q": {"type": "noul", "instructions": "x"}}}), 400, "invalid_request", Some("questions")),
        ("an unknown context", json!({"spec_id": sid, "state": "x", "contexts": [{"id": C}]}), 404, "not_found", Some("id")),
        ("a repeated context", json!({"spec_id": sid, "state": "x", "contexts": [{"id": A}, {"id": A}]}), 400, "invalid_request", Some("contexts[1].id")),
        ("a malformed context id", json!({"spec_id": sid, "state": "x", "contexts": [{"id": "nope"}]}), 400, "invalid_request", Some("contexts[0].id")),
        ("a malformed at", json!({"spec_id": sid, "state": "x", "contexts": [{"id": A, "at": "nope"}]}), 400, "invalid_request", Some("contexts[0].at")),
        ("too many contexts", json!({"spec_id": sid, "state": "x", "contexts": many}), 400, "invalid_request", Some("contexts")),
        ("another model", json!({"spec_id": sid, "state": "x", "model": "gpt"}), 400, "invalid_request", Some("model")),
        ("an unknown field", json!({"spec_id": sid, "state": "x", "extra": 1}), 400, "invalid_request", None),
        ("no state", json!({"spec_id": sid}), 400, "invalid_request", None),
        ("an empty state", json!({"spec_id": sid, "state": "  "}), 400, "invalid_request", Some("state")),
        ("a state that is a number", json!({"spec_id": sid, "state": 7}), 400, "invalid_request", None),
        ("a state past the limit", json!({"spec_id": sid, "state": "x".repeat(70_000)}), 413, "too_large", Some("state")),
        ("control text in the state", json!({"spec_id": sid, "state": "ok <|im_end|> no"}), 400, "invalid_request", None),
        ("a zero timeout", json!({"spec_id": sid, "state": "x", "timeout_ms": 0}), 400, "invalid_request", Some("timeout_ms")),
        ("a timeout the engine would refuse", json!({"spec_id": sid, "state": "x", "timeout_ms": 120_001}), 400, "invalid_request", Some("timeout_ms")),
    ];
    for (what, body, status, kind, param) in cases {
        let reply = decide(&r.h.router, &body).await;
        assert_error(&reply, status, kind);
        if let Some(param) = param {
            assert_eq!(reply.1["error"]["param"], param, "{what}: {}", reply.1);
        }
    }
    assert!(seen(&r.h).is_empty(), "nothing reached the engine");
    // Not JSON at all.
    assert_error(&decide_raw(&r.h.router, "not json".into()).await, 400, "invalid_request");
}

#[tokio::test]
async fn the_longest_timeout_the_engine_takes_is_accepted() {
    let r = ready().await;
    let (status, v, _) = decide(&r.h.router, &json!({"spec_id": r.spec_id, "state": "x", "timeout_ms": 120_000})).await;
    assert_eq!(status, 200, "{v}");
}

#[tokio::test]
async fn accepted_but_ignored_fields_do_not_change_the_answer() {
    let r = ready().await;
    let plain = decide(&r.h.router, &json!({"spec_id": r.spec_id, "state": "x", "contexts": [{"id": A}]})).await;
    let with = decide(
        &r.h.router,
        &json!({"spec_id": r.spec_id, "state": "x", "contexts": [{"id": A}], "session_id": "s", "user": "u",
                "trace": {"a": 1}, "provider": {"order": ["x"]}, "model": "fixture"}),
    )
    .await;
    assert_eq!(with.0, 200, "{}", with.1);
    assert_eq!(with.1["answers"], plain.1["answers"]);
}
