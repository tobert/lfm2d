//! `/council/v1/contexts/{id}` at the wire, over a fake engine that keeps the
//! real one's rules: content-addressed snapshots, a snapshot held after each
//! declared boundary, resume from the largest declared boundary already
//! held, an LRU that never evicts a pin. Bodies are validated against the
//! contract's schemas. That the real engine computes the same bits however a
//! context arrived is `contexts_real.rs`' job (it needs weights).
use axum::body::Body;
use axum::http::Request;
use lfm2d::adjudicator::{
    AdjudicateRequest, AdjudicateResponse, Failure, Generator, Handle, PrefixInfo, PromptSpec, RegisterOutcome,
    UnregisterOutcome, YieldPoint,
};
use lfm2d::council::{ContextBody, plan_put, render_segments};
use lfm2d::council_context::{BuildOutcome, BuildRequest, BuiltSnapshot, HeldInfo};
use lfm2d::opinion_api::{OpinionRequest, OpinionResponse, ResolvedQuestion};
use serde_json::{Value, json};
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
struct Store {
    /// Oldest first.
    items: Vec<Item>,
    capacity: usize,
    puts: usize,
    /// The next this-many `council_unpin` calls fail, as a full queue would.
    unpin_failures: usize,
    /// At most this many pinned items: a pin past it is refused.
    pin_limit: Option<usize>,
    /// How long a build takes, so concurrent requests overlap.
    put_delay_ms: u64,
}
#[derive(Clone)]
struct Fake(Arc<Mutex<Store>>);

fn engine_id(text: &str) -> String {
    lfm2d::hash::sha256_hex_bytes(format!("fake\0{text}").as_bytes())
}

impl Store {
    fn find(&self, id: &str) -> Option<&Item> {
        self.items.iter().find(|i| i.id == id)
    }
    fn insert(&mut self, id: String, tokens: usize) -> Result<(), Failure> {
        while self.items.len() >= self.capacity {
            let Some(at) = self.items.iter().position(|i| !i.pinned) else {
                return Err(Failure::InsufficientStorage("every held context is pinned".into()));
            };
            self.items.remove(at);
        }
        self.items.push(Item { id, tokens, pinned: false });
        Ok(())
    }
}

impl Generator for Fake {
    fn generate(&mut self, _: &AdjudicateRequest, _: &dyn YieldPoint<Self>) -> Result<AdjudicateResponse, Failure> {
        Err(Failure::Internal("not used".into()))
    }
    fn opine(
        &mut self,
        _: &OpinionRequest,
        _: &[ResolvedQuestion],
        _: &dyn Fn() -> Result<(), Failure>,
    ) -> Result<OpinionResponse, Failure> {
        Err(Failure::Internal("not used".into()))
    }
    fn register(
        &mut self,
        _: String,
        _: PromptSpec,
        _: &dyn Fn() -> Result<(), Failure>,
    ) -> Result<RegisterOutcome, Failure> {
        Err(Failure::Internal("not used".into()))
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

    /// One token per character of rendered text.
    fn council_put(&mut self, request: &BuildRequest, at: &dyn YieldPoint<Self>) -> Result<BuildOutcome, Failure> {
        request.validate().map_err(Failure::BadRequest)?;
        at.check()?;
        let delay = self.0.lock().unwrap().put_delay_ms;
        if delay > 0 {
            std::thread::sleep(std::time::Duration::from_millis(delay));
        }
        let mut store = self.0.lock().unwrap();
        let mut ends = Vec::new();
        let mut total = 0;
        for s in &request.segments {
            total += s.len();
            ends.push(total);
        }
        let id_at = |b: usize| engine_id(&request.segments[..=b].concat());
        let base = request.hold_after.iter().rev().copied().find(|&b| store.find(&id_at(b)).is_some());
        let kept = base.map_or(0, |b| ends[b]);
        let head = request.segments.len() - 1;
        let mut published = Vec::new();
        if !request.dry_run {
            store.puts += 1;
            for &b in request.hold_after.iter().filter(|&&b| base.is_none_or(|k| b > k)) {
                store.insert(id_at(b), ends[b])?;
                published.push(id_at(b));
            }
            // A held boundary is used again: move it to the young end.
            if let Some(at) = store.items.iter().position(|i| i.id == id_at(head)) {
                let item = store.items.remove(at);
                store.items.push(item);
            }
            if request.pin == Some(true) {
                let id = id_at(head);
                let others = store.items.iter().filter(|i| i.pinned && i.id != id).count();
                if store.pin_limit.is_some_and(|limit| others >= limit) {
                    // A refusal changes nothing: what this request built goes.
                    store.items.retain(|i| !published.contains(&i.id));
                    return Err(Failure::InsufficientStorage("the pin budget is spent".into()));
                }
                store.items.iter_mut().find(|i| i.id == id).unwrap().pinned = true;
            }
        }
        let snapshots = request
            .hold_after
            .iter()
            .map(|&b| {
                let id = id_at(b);
                let item = store.find(&id);
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
        let store = self.0.lock().unwrap();
        Ok(ids
            .iter()
            .map(|id| store.find(id).map(|i| HeldInfo { tokens: i.tokens, bytes: i.tokens * 10, pinned: i.pinned }))
            .collect())
    }
    fn council_unpin(&mut self, id: &str) -> Result<bool, Failure> {
        let mut store = self.0.lock().unwrap();
        if store.unpin_failures > 0 {
            store.unpin_failures -= 1;
            return Err(Failure::Internal("the queue is full".into()));
        }
        Ok(store.items.iter_mut().find(|i| i.id == id).map(|i| i.pinned = false).is_some())
    }
}

// ---------------------------------------------------------------- the harness

struct Harness {
    router: axum::Router,
    fake: Fake,
}

fn harness(capacity: usize) -> Harness {
    harness_with(capacity, lfm2d::council::Limits::new(100_000))
}

fn harness_with(capacity: usize, limits: lfm2d::council::Limits) -> Harness {
    let fake = Fake(Arc::new(Mutex::new(Store {
        items: vec![],
        capacity,
        puts: 0,
        unpin_failures: 0,
        pin_limit: None,
        put_delay_ms: 0,
    })));
    let handle = Handle::spawn(fake.clone(), (&info()).into()).with_menu(vec![]);
    Harness { router: lfm2d::council_api::router_with_limits(handle, (&info()).into(), limits), fake }
}

type Reply = (u16, Value, String);

async fn send(router: &axum::Router, request: Request<Body>) -> Reply {
    let response = router.clone().oneshot(request).await.unwrap();
    let status = response.status().as_u16();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let text = String::from_utf8_lossy(&bytes).into_owned();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null), text)
}
async fn put(router: &axum::Router, id: &str, body: &Value) -> Reply {
    put_with(router, id, body, &[]).await
}
async fn put_with(router: &axum::Router, id: &str, body: &Value, headers: &[(&str, &str)]) -> Reply {
    let mut request = Request::put(format!("/council/v1/contexts/{id}"));
    for (k, v) in headers {
        request = request.header(*k, *v);
    }
    send(router, request.body(Body::from(body.to_string())).unwrap()).await
}
async fn get(router: &axum::Router, id: &str) -> Reply {
    send(router, Request::get(format!("/council/v1/contexts/{id}")).body(Body::empty()).unwrap()).await
}
async fn delete(router: &axum::Router, id: &str) -> Reply {
    send(router, Request::delete(format!("/council/v1/contexts/{id}")).body(Body::empty()).unwrap()).await
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

fn turn(text: &str, snap: bool) -> Value {
    json!({"role": "user", "content": text, "snap": snap})
}
fn ctx(system: &str, turns: &[Value]) -> Value {
    json!({"system": system, "turns": turns})
}
fn body_of(v: &Value) -> ContextBody {
    serde_json::from_value(v.clone()).unwrap()
}
fn len_of(v: &Value, through_turn: Option<usize>) -> usize {
    let segments = render_segments(&body_of(v)).unwrap();
    segments[..=through_turn.map_or(0, |t| t + 1)].iter().map(String::len).sum()
}

// ------------------------------------------------------------------------ put

#[tokio::test]
async fn a_put_holds_a_context_and_answers_with_the_contracts_result() {
    let h = harness(16);
    let c = ctx("Judge.", &[turn("one", true), turn("two", false)]);
    let (status, v, _) = put(&h.router, A, &c).await;
    assert_eq!(status, 200, "{v}");
    assert_valid("ContextPutResult", &v);
    assert_eq!(v["id"], A);
    assert_eq!((v["kept"].as_u64(), v["dry_run"].as_bool()), (Some(0), Some(false)));
    assert_eq!(v["fed"], v["tokens"], "a first build runs everything");
    let snaps = v["snapshots"].as_array().unwrap();
    assert_eq!(snaps.len(), 3, "the system, the snap turn and the head: {v}");
    assert_eq!((snaps[0]["layer"].as_str(), snaps[1]["layer"].as_str(), snaps[2]["layer"].as_str()), (Some("system"), Some("turn"), Some("turn")));
    assert_eq!((snaps[1]["turn"].as_u64(), snaps[2]["turn"].as_u64()), (Some(0), Some(1)));
    assert_eq!(v["head"], snaps[2]["id"], "the head is the last snapshot");
}

#[tokio::test]
async fn engine_ids_never_reach_the_wire() {
    let h = harness(16);
    let c = ctx("Judge.", &[turn("one", true), turn("two", false)]);
    let (_, v, text) = put(&h.router, A, &c).await;
    let store = h.fake.0.lock().unwrap();
    assert!(store.items.len() >= 3);
    for item in &store.items {
        assert!(!text.contains(&item.id), "the engine's id is internal: {v}");
    }
    drop(store);
    let (_, _, text) = get(&h.router, A).await;
    assert!(!h.fake.0.lock().unwrap().items.iter().any(|i| text.contains(&i.id)));
}

#[tokio::test]
async fn the_same_context_again_keeps_everything_and_feeds_nothing() {
    let h = harness(16);
    let c = ctx("Judge.", &[turn("one", true), turn("two", false)]);
    let (_, first, _) = put(&h.router, A, &c).await;
    let (status, again, _) = put(&h.router, A, &c).await;
    assert_eq!(status, 200);
    assert_valid("ContextPutResult", &again);
    assert_eq!(again["head"], first["head"]);
    assert_eq!((again["kept"].as_u64(), again["fed"].as_u64()), (first["tokens"].as_u64(), Some(0)));
}

#[tokio::test]
async fn an_update_extends_only_from_a_declared_boundary() {
    // Append-only with every turn marked: only the new turn is fed.
    let h = harness(16);
    let marked = |n: usize| ctx("Judge.", &(0..n).map(|i| turn(&format!("turn {i}"), true)).collect::<Vec<_>>());
    put(&h.router, A, &marked(3)).await;
    let (_, v, _) = put(&h.router, A, &marked(4)).await;
    assert_eq!(v["kept"].as_u64().unwrap() as usize, len_of(&marked(4), Some(2)), "everything but the new turn");
    assert_eq!(v["fed"].as_u64().unwrap() as usize, len_of(&marked(4), Some(3)) - len_of(&marked(4), Some(2)));

    // The same log with nothing marked: the held head is not a boundary, so
    // an appended turn re-feeds every turn since the system's.
    let unmarked = |n: usize| ctx("Judge.", &(0..n).map(|i| turn(&format!("turn {i}"), false)).collect::<Vec<_>>());
    // A fresh store: the same texts are the same content, so with A's marked log held,
    // B's would find A's exact head and keep it all.
    let h = harness(16);
    put(&h.router, B, &unmarked(3)).await;
    let (_, v, _) = put(&h.router, B, &unmarked(4)).await;
    assert_eq!(v["kept"].as_u64().unwrap() as usize, len_of(&unmarked(4), None), "only the system head");
}

#[tokio::test]
async fn what_an_update_keeps_agrees_with_the_contracts_diff_rule() {
    // The engine and `plan_put` (the contract's rule, as a pure function)
    // must agree wherever the content differs.
    let base = ctx("Judge.", &[turn("a", true), turn("b", false), turn("c", true), turn("d", false)]);
    let edits: Vec<(&str, Value)> = vec![
        ("an appended turn", ctx("Judge.", &[turn("a", true), turn("b", false), turn("c", true), turn("d", false), turn("e", false)])),
        ("an edit at b", ctx("Judge.", &[turn("a", true), turn("B", false), turn("c", true), turn("d", false)])),
        ("an edit at the first turn", ctx("Judge.", &[turn("A", true), turn("b", false), turn("c", true), turn("d", false)])),
        ("a changed system", ctx("Other.", &[turn("a", true), turn("b", false), turn("c", true), turn("d", false)])),
        ("a shorter log", ctx("Judge.", &[turn("a", true), turn("b", false)])),
    ];
    for (what, edit) in edits {
        let h = harness(64);
        put(&h.router, A, &base).await;
        let (_, v, _) = put(&h.router, A, &edit).await;
        let held = body_of(&base);
        let plan = plan_put(Some((&held.system, &held.turns)), &body_of(&edit));
        let segments = render_segments(&body_of(&edit)).unwrap();
        let want = if plan.kept_system { segments[..=plan.kept_turns].iter().map(String::len).sum::<usize>() } else { 0 };
        assert_eq!(v["kept"].as_u64().unwrap() as usize, want, "{what}: {v}");
    }
}

#[tokio::test]
async fn going_back_to_an_earlier_version_reuses_what_is_still_held() {
    let h = harness(64);
    let v1 = ctx("Judge.", &[turn("a", true), turn("b", true)]);
    let v2 = ctx("Judge.", &[turn("a", true), turn("B", true)]);
    let (_, first, _) = put(&h.router, A, &v1).await;
    put(&h.router, A, &v2).await;
    let (_, back, _) = put(&h.router, A, &v1).await;
    assert_eq!(back["head"], first["head"], "the same content is the same snapshot");
    assert_eq!((back["kept"].as_u64(), back["fed"].as_u64()), (first["tokens"].as_u64(), Some(0)));
}

#[tokio::test]
async fn a_dry_run_reports_the_cost_and_builds_nothing() {
    let h = harness(16);
    let first = ctx("Judge.", &[turn("a", true)]);
    let second = ctx("Judge.", &[turn("a", true), turn("b", true)]);
    put(&h.router, A, &first).await;
    let held = h.fake.0.lock().unwrap().items.len();

    let mut dry = second.clone();
    dry["dry_run"] = json!(true);
    let (status, d, _) = put(&h.router, A, &dry).await;
    assert_eq!(status, 200);
    assert_valid("ContextPutResult", &d);
    assert_eq!(d["dry_run"], true);
    assert_eq!(h.fake.0.lock().unwrap().items.len(), held, "nothing was built");
    let (_, now, _) = get(&h.router, A).await;
    assert_eq!(now["tokens"].as_u64().unwrap() as usize, len_of(&first, Some(0)), "the held context is unchanged");
    assert_ne!(d["head"], now["head"], "head is the id the build would have");

    let (_, real, _) = put(&h.router, A, &second).await;
    assert_eq!((real["kept"].clone(), real["fed"].clone(), real["head"].clone()), (d["kept"].clone(), d["fed"].clone(), d["head"].clone()), "the dry run told the truth");
    // And a dry run on a context never held creates no record.
    let (_, _, _) = put(&h.router, B, &{ let mut v = first.clone(); v["dry_run"] = json!(true); v }).await;
    assert_error(&get(&h.router, B).await, 404, "not_found");
}

#[tokio::test]
async fn a_get_is_the_contracts_state_and_matches_the_put() {
    let h = harness(16);
    let (_, p, _) = put(&h.router, A, &ctx("Judge.", &[turn("a", true), turn("b", false)])).await;
    let (status, g, _) = get(&h.router, A).await;
    assert_eq!(status, 200);
    assert_valid("ContextState", &g);
    for key in ["id", "head", "tokens", "snapshots"] {
        assert_eq!(g[key], p[key], "{key}");
    }
    assert!(g.get("kept").is_none() && g.get("fed").is_none(), "a GET carries no kept and fed");
}

#[tokio::test]
async fn unknown_and_malformed_ids() {
    let h = harness(16);
    for id in [A, "nope", "0199B3C4-6C1E-7A2B-9F00-3E5D1C2A7B11"] {
        assert_error(&get(&h.router, id).await, 404, "not_found");
        assert_error(&delete(&h.router, id).await, 404, "not_found");
    }
    // A client names a context it holds with a UUID; a PUT with anything else is its mistake.
    for id in ["nope", "0199B3C4-6C1E-7A2B-9F00-3E5D1C2A7B11", "0199b3c46c1e7a2b9f003e5d1c2a7b11"] {
        let reply = put(&h.router, id, &ctx("Judge.", &[])).await;
        assert_error(&reply, 400, "invalid_request");
        assert_eq!(reply.1["error"]["param"], "id");
    }
}

// ------------------------------------------------------------------- If-Match

#[tokio::test]
async fn if_match_makes_a_put_conditional_on_the_head() {
    let h = harness(16);
    let (_, first, _) = put(&h.router, A, &ctx("Judge.", &[turn("a", true)])).await;
    let head = first["head"].as_str().unwrap().to_owned();
    let next = ctx("Judge.", &[turn("a", true), turn("b", true)]);

    let ok = put_with(&h.router, A, &next, &[("if-match", &head)]).await;
    assert_eq!(ok.0, 200, "{}", ok.1);
    // The head moved: the old one no longer matches, and the answer names the current one.
    let stale = put_with(&h.router, A, &ctx("Judge.", &[turn("zzz", true)]), &[("if-match", &head)]).await;
    assert_error(&stale, 412, "head_mismatch");
    assert_eq!(stale.1["error"]["head"], ok.1["head"]);
    assert_eq!(get(&h.router, A).await.1["head"], ok.1["head"], "a refused PUT changed nothing");
    // A quoted value, as an HTTP client may send it.
    let quoted = put_with(&h.router, A, &next, &[("if-match", &format!("\"{}\"", ok.1["head"].as_str().unwrap()))]).await;
    assert_eq!(quoted.0, 200);
}

#[tokio::test]
async fn if_match_on_a_missing_context_and_a_malformed_value() {
    let h = harness(16);
    let missing = put_with(&h.router, A, &ctx("Judge.", &[]), &[("if-match", &format!("snap:{}", "a".repeat(64)))]).await;
    assert_error(&missing, 412, "head_mismatch");
    assert!(missing.1["error"].get("head").is_none(), "there is no current head to name");
    let bad = put_with(&h.router, A, &ctx("Judge.", &[]), &[("if-match", "not-a-snapshot")]).await;
    assert_error(&bad, 400, "invalid_request");
    assert_eq!(h.fake.0.lock().unwrap().puts, 0, "neither reached the engine");
}

// ----------------------------------------------------------------- delete, pin

#[tokio::test]
async fn a_delete_drops_the_record_and_leaves_the_snapshots_to_eviction() {
    let h = harness(16);
    put(&h.router, A, &ctx("Judge.", &[turn("a", true)])).await;
    let held = h.fake.0.lock().unwrap().items.len();
    let (status, body, text) = delete(&h.router, A).await;
    assert_eq!((status, text.as_str(), &body), (204, "", &Value::Null));
    assert_error(&get(&h.router, A).await, 404, "not_found");
    assert_error(&delete(&h.router, A).await, 404, "not_found");
    assert_eq!(h.fake.0.lock().unwrap().items.len(), held, "snapshots are eviction candidates, not deleted");
    // Another context with the same content finds them again.
    let (_, v, _) = put(&h.router, B, &ctx("Judge.", &[turn("a", true)])).await;
    assert_eq!(v["fed"], 0);
}

fn engine_pinned(h: &Harness) -> usize {
    h.fake.0.lock().unwrap().items.iter().filter(|i| i.pinned).count()
}

#[tokio::test]
async fn a_pin_exempts_the_head_from_eviction_and_follows_it() {
    let h = harness(4);
    let mut pinned = ctx("Judge.", &[turn("a", true)]);
    pinned["pin"] = json!(true);
    let (_, v, _) = put(&h.router, A, &pinned).await;
    assert_eq!(v["pinned"], true);
    assert_eq!(engine_pinned(&h), 1);
    // Churn through the store: the pinned head survives, the unpinned go.
    for (n, id) in [B, C].into_iter().enumerate() {
        put(&h.router, id, &ctx(&format!("Other {n}."), &[turn("x", true), turn("y", true)])).await;
    }
    assert_eq!(get(&h.router, A).await.0, 200, "the pinned head was not evicted");
    // `pin` absent leaves it as it was: an append moves the pin to the new head.
    let (_, moved, _) = put(&h.router, A, &ctx("Judge.", &[turn("a", true), turn("b", true)])).await;
    assert_eq!(moved["pinned"], true);
    assert_eq!(engine_pinned(&h), 1, "the new head is pinned and the old one released");
    assert_ne!(moved["head"], v["head"]);
}

#[tokio::test]
async fn a_pin_is_released_by_pin_false_and_by_delete() {
    let h = harness(8);
    let mut pinned = ctx("Judge.", &[turn("a", true)]);
    pinned["pin"] = json!(true);
    put(&h.router, A, &pinned).await;
    let mut release = pinned.clone();
    release["pin"] = json!(false);
    let (_, v, _) = put(&h.router, A, &release).await;
    assert_eq!((v["pinned"].as_bool(), engine_pinned(&h)), (Some(false), 0));

    put(&h.router, A, &pinned).await;
    assert_eq!(engine_pinned(&h), 1);
    delete(&h.router, A).await;
    assert_eq!(engine_pinned(&h), 0, "a deleted context holds no pin");
}

#[tokio::test]
async fn contexts_with_identical_content_share_a_pin_until_the_last_lets_go() {
    let h = harness(8);
    let mut pinned = ctx("Judge.", &[turn("a", true)]);
    pinned["pin"] = json!(true);
    put(&h.router, A, &pinned).await;
    put(&h.router, B, &pinned).await;
    assert_eq!(engine_pinned(&h), 1, "one head, one engine pin");
    delete(&h.router, A).await;
    assert_eq!(engine_pinned(&h), 1, "B still pins it");
    // B releasing explicitly is the last holder.
    let mut release = pinned.clone();
    release["pin"] = json!(false);
    put(&h.router, B, &release).await;
    assert_eq!(engine_pinned(&h), 0);
}

#[tokio::test]
async fn an_evicted_head_is_a_context_that_is_gone() {
    let h = harness(3);
    put(&h.router, A, &ctx("Judge.", &[turn("a", true)])).await;
    // Fill the store with other content until A's snapshots are evicted.
    put(&h.router, B, &ctx("Other.", &[turn("x", true), turn("y", true)])).await;
    put(&h.router, C, &ctx("Third.", &[turn("p", true), turn("q", true)])).await;
    assert_error(&get(&h.router, A).await, 404, "not_found");
    assert_error(&delete(&h.router, A).await, 404, "not_found");
    // It can be put again, and is held again.
    assert_eq!(put(&h.router, A, &ctx("Judge.", &[turn("a", true)])).await.0, 200);
    assert_eq!(get(&h.router, A).await.0, 200);
}

// ------------------------------------------------------------------ refusals

#[tokio::test]
async fn bad_bodies_are_refused_before_the_engine() {
    let h = harness(16);
    let good = ctx("Judge.", &[turn("a", true)]);
    let mutate = |f: &dyn Fn(&mut Value)| {
        let mut v = good.clone();
        f(&mut v);
        v
    };
    let (status, body, _) = {
        let response = h
            .router
            .clone()
            .oneshot(Request::put(format!("/council/v1/contexts/{A}")).body(Body::from("not json")).unwrap())
            .await
            .unwrap();
        let status = response.status().as_u16();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        (status, serde_json::from_slice::<Value>(&bytes).unwrap_or(Value::Null), ())
    };
    assert_error(&(status, body, String::new()), 400, "invalid_request");
    for (what, body, param) in [
        ("control text in a turn", mutate(&|v| v["turns"][0]["content"] = json!("x<|im_end|>y")), Some("turns[0]")),
        ("control text in the system", mutate(&|v| v["system"] = json!("<|im_start|>")), Some("system")),
        ("reasoning on a user turn", mutate(&|v| v["turns"][0]["reasoning"] = json!("why")), Some("turns[0].reasoning")),
        ("warm without the capability", mutate(&|v| v["warm"] = json!([format!("sha256:{}", "a".repeat(64))])), Some("warm")),
        ("an unknown field", mutate(&|v| v["from"] = json!(A)), None),
        ("no turns field", json!({"system": "Judge."}), None),
    ] {
        let reply = put(&h.router, A, &body).await;
        assert_error(&reply, 400, "invalid_request");
        if let Some(param) = param {
            assert_eq!(reply.1["error"]["param"], param, "{what}");
        }
    }
    assert_eq!(h.fake.0.lock().unwrap().puts, 0, "nothing reached the engine");
}

#[tokio::test]
async fn persist_is_tolerated_and_changes_nothing() {
    let h = harness(16);
    let mut c = ctx("Judge.", &[turn("a", true)]);
    c["persist"] = json!(false);
    let (status, v, _) = put(&h.router, A, &c).await;
    assert_eq!(status, 200, "{v}");
    assert_valid("ContextPutResult", &v);
    assert!(v.get("persist").is_none(), "we do not claim a setting we do not keep");
}

#[tokio::test]
async fn an_oversized_body_is_a_413() {
    let h = harness(16);
    let big = ctx("Judge.", &[turn(&"x".repeat(5 * 1_048_576), true)]);
    let reply = put(&h.router, A, &big).await;
    assert_error(&reply, 413, "too_large");
    assert_eq!(reply.1["error"]["param"], "body");
}

// ------------------------------------------------------- concurrency, bounds

fn engine_pinned_ids(h: &Harness) -> Vec<String> {
    h.fake.0.lock().unwrap().items.iter().filter(|i| i.pinned).map(|i| i.id.clone()).collect()
}

/// `If-Match` makes a PUT conditional. Two PUTs that both name the head they
/// saw cannot both be accepted: the second sees the first's head and is a 412.
#[tokio::test]
async fn two_puts_naming_the_same_head_cannot_both_win() {
    let h = harness(64);
    let (_, first, _) = put(&h.router, A, &ctx("Judge.", &[turn("a", true)])).await;
    let head = first["head"].as_str().unwrap().to_owned();
    h.fake.0.lock().unwrap().put_delay_ms = 80;
    let (bx, by) = (ctx("Judge.", &[turn("a", true), turn("x", true)]), ctx("Judge.", &[turn("a", true), turn("y", true)]));
    let guard = [("if-match", head.as_str())];
    let (x, y) = tokio::join!(put_with(&h.router, A, &bx, &guard), put_with(&h.router, A, &by, &guard));
    let mut statuses = [x.0, y.0];
    statuses.sort();
    assert_eq!(statuses, [200, 412], "one wins, the other sees the winner's head: {} / {}", x.1, y.1);
    let winner = if x.0 == 200 { &x } else { &y };
    assert_eq!(get(&h.router, A).await.1["head"], winner.1["head"]);
}

/// The route's pin counts and the engine's pins stay in step when a release and
/// a pin on the same head overlap.
#[tokio::test]
async fn a_release_and_a_pin_on_one_head_leave_the_engine_pinned_as_the_records_say() {
    for _ in 0..8 {
        let h = harness(64);
        let mut pinned = ctx("Judge.", &[turn("a", true)]);
        pinned["pin"] = json!(true);
        put(&h.router, A, &pinned).await;
        h.fake.0.lock().unwrap().put_delay_ms = 20;
        let mut release = ctx("Judge.", &[turn("a", true)]);
        release["pin"] = json!(false);
        // A lets go of H while B takes it: whichever order, B holds the pin.
        let (a, b) = tokio::join!(put(&h.router, A, &release), put(&h.router, B, &pinned));
        assert_eq!((a.0, b.0), (200, 200), "{} / {}", a.1, b.1);
        assert_eq!(engine_pinned_ids(&h).len(), 1, "B pins the head, so the engine does");
        assert_eq!(get(&h.router, B).await.1["pinned"], true);
        delete(&h.router, B).await;
        assert!(engine_pinned_ids(&h).is_empty(), "and releases it when B lets go");
    }
}

#[tokio::test]
async fn an_unpin_that_fails_is_retried_not_forgotten() {
    let h = harness(64);
    let mut pinned = ctx("Judge.", &[turn("a", true)]);
    pinned["pin"] = json!(true);
    put(&h.router, A, &pinned).await;
    assert_eq!(engine_pinned_ids(&h).len(), 1);
    // The release fails: the client is told it is released, and the engine still holds it.
    h.fake.0.lock().unwrap().unpin_failures = 1;
    let mut release = pinned.clone();
    release["pin"] = json!(false);
    let (status, v, _) = put(&h.router, A, &release).await;
    assert_eq!((status, v["pinned"].as_bool()), (200, Some(false)));
    assert_eq!(engine_pinned_ids(&h).len(), 1, "the engine was never told");
    // The next mutation of any context tells it.
    put(&h.router, B, &ctx("Other.", &[turn("b", true)])).await;
    assert!(engine_pinned_ids(&h).is_empty(), "the pending release was retried");
}

#[tokio::test]
async fn a_refused_pin_changes_nothing() {
    let h = harness(64);
    h.fake.0.lock().unwrap().pin_limit = Some(1);
    let mut first = ctx("Judge.", &[turn("a", true)]);
    first["pin"] = json!(true);
    assert_eq!(put(&h.router, A, &first).await.0, 200);
    let items = h.fake.0.lock().unwrap().items.len();
    let mut second = ctx("Other.", &[turn("b", true)]);
    second["pin"] = json!(true);
    let reply = put(&h.router, B, &second).await;
    assert_error(&reply, 507, "pin_budget");
    assert_error(&get(&h.router, B).await, 404, "not_found");
    assert_eq!(h.fake.0.lock().unwrap().items.len(), items, "what the refused build made is gone");
    assert_eq!(engine_pinned_ids(&h).len(), 1);
    assert_eq!(get(&h.router, A).await.1["pinned"], true, "the first pin is untouched");
}

#[tokio::test]
async fn held_contexts_are_bounded_and_the_evicted_are_reaped_to_make_room() {
    let mut limits = lfm2d::council::Limits::new(100_000);
    limits.contexts_held = 2;
    let h = harness_with(64, limits);
    assert_eq!(put(&h.router, A, &ctx("A.", &[turn("a", true)])).await.0, 200);
    assert_eq!(put(&h.router, B, &ctx("B.", &[turn("b", true)])).await.0, 200);
    // A third, new context is over the bound: a 429 that says what to do.
    let reply = put(&h.router, C, &ctx("C.", &[turn("c", true)])).await;
    assert_error(&reply, 429, "busy");
    assert!(reply.1["error"]["message"].as_str().unwrap().contains("DELETE"));
    // An update to a held one is not a new context.
    assert_eq!(put(&h.router, A, &ctx("A.", &[turn("a", true), turn("more", true)])).await.0, 200);
    // A dry run never creates a record.
    let mut dry = ctx("C.", &[turn("c", true)]);
    dry["dry_run"] = json!(true);
    assert_eq!(put(&h.router, C, &dry).await.0, 200);
    // A delete makes room.
    delete(&h.router, B).await;
    assert_eq!(put(&h.router, C, &ctx("C.", &[turn("c", true)])).await.0, 200);
}

#[tokio::test]
async fn at_the_bound_a_context_the_engine_evicted_makes_room() {
    let mut limits = lfm2d::council::Limits::new(100_000);
    limits.contexts_held = 2;
    // A store of four: B's updates push A's snapshots out while A's record lingers.
    let h = harness_with(4, limits);
    put(&h.router, A, &ctx("A.", &[turn("a", true)])).await;
    put(&h.router, B, &ctx("B.", &[turn("b1", true)])).await;
    put(&h.router, B, &ctx("B.", &[turn("b2", true)])).await;
    put(&h.router, B, &ctx("B.", &[turn("b3", true)])).await;
    // A is dead in the engine but still has a record; a new context reaps it.
    assert_eq!(put(&h.router, C, &ctx("C.", &[turn("c", true)])).await.0, 200);
    assert_error(&get(&h.router, A).await, 404, "not_found");
}
