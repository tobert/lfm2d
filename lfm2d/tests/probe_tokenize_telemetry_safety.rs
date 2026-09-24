//! `POST /v1/probe` and `POST /v1/tokenize` both exist specifically to let
//! a caller hand this daemon arbitrary text — which is exactly the shape
//! of endpoint `tests/spans_telemetry_safety.rs` exists to hold to a
//! stricter bar: this file proves a known secret string embedded in a
//! probe/tokenize request NEVER shows up anywhere in this crate's tracing
//! output, modelled directly on that file (same capture layer, same
//! "everything this crate exports over OTLP is built from exactly these
//! same spans/events" reasoning — see its module docs).
use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use tower::ServiceExt;
use tracing::field::{Field, Visit};
use tracing::span;
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::prelude::*;
use tracing_subscriber::Registry;

use lfm2d::adjudicator::{
    AdjudicateRequest, AdjudicateResponse, Failure, Generator, Handle, PrefixInfo,
};
use lfm2d::tokenize_api::TokenizerRegistry;

#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Vec<String>>>);

struct StringVisitor<'a>(&'a mut Vec<String>);

impl Visit for StringVisitor<'_> {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        self.0.push(format!("{}={value:?}", field.name()));
    }
    fn record_str(&mut self, field: &Field, value: &str) {
        self.0.push(format!("{}={value}", field.name()));
    }
}

impl<S: tracing::Subscriber> Layer<S> for Capture {
    fn on_new_span(&self, attrs: &span::Attributes<'_>, _id: &span::Id, _ctx: Context<'_, S>) {
        let mut buf = Vec::new();
        attrs.record(&mut StringVisitor(&mut buf));
        self.0.lock().unwrap().extend(buf);
    }
    fn on_record(&self, _id: &span::Id, values: &span::Record<'_>, _ctx: Context<'_, S>) {
        let mut buf = Vec::new();
        values.record(&mut StringVisitor(&mut buf));
        self.0.lock().unwrap().extend(buf);
    }
    fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        let mut buf = Vec::new();
        event.record(&mut StringVisitor(&mut buf));
        self.0.lock().unwrap().extend(buf);
    }
}

/// A secret distinctive enough that finding its literal bytes anywhere in
/// captured telemetry is unambiguous proof of a leak, not a coincidence.
const SECRET: &str = "sk-KNOWN_TEST_SECRET_do_not_leak_9f8e7d6c5b4a";

fn secret_bearing_text() -> String {
    format!("here is an api key: {SECRET} — please rotate it")
}

async fn post_json(router: axum::Router, path: &str, body: Value) -> (StatusCode, Value) {
    let resp = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(path)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, json)
}

// ------------------------------------------------------------ /v1/probe

fn info() -> PrefixInfo {
    PrefixInfo {
        model_id: "fixture".into(),
        weight_hash: "hash".into(),
        tokenizer_hash: "tok".into(),
        template_version: "test".into(),
        snapshot_id: "snapshot".into(),
        prefix_tokens: 7,
        input_cache_capacity: 1,
        context_limit: 128,
        backend: "cpu".into(),
        dtype: "f32".into(),
        sampling: "greedy".into(),
        weight_dtypes: vec!["F32".into()],
    }
}

/// A minimal `Generator` that answers `probe` with a canned response —
/// the leak this file checks for would have to come from the WIRING
/// around the generator (the worker's dispatch span/log line,
/// `Handle::submit`'s span, the shared `telemetry_middleware`), not from
/// this double, so it does not need to do anything with the request text
/// beyond receiving it.
struct Fake;
impl Generator for Fake {
    fn generate(
        &mut self,
        _: &AdjudicateRequest,
        _: &dyn Fn() -> Result<(), Failure>,
    ) -> Result<AdjudicateResponse, Failure> {
        Err(Failure::Internal("not exercised here".into()))
    }
    fn opine(
        &mut self,
        _: &lfm2d::opinion_api::OpinionRequest,
        _: &[lfm2d::opinion_api::ResolvedQuestion],
        _: &dyn Fn() -> Result<(), Failure>,
    ) -> Result<lfm2d::opinion_api::OpinionResponse, Failure> {
        Err(Failure::Internal("not exercised here".into()))
    }
    fn register(
        &mut self,
        _: String,
        _: lfm2d::adjudicator::PromptSpec,
        _: &dyn Fn() -> Result<(), Failure>,
    ) -> Result<lfm2d::adjudicator::RegisterOutcome, Failure> {
        Err(Failure::Internal("not exercised here".into()))
    }
    fn unregister(&mut self, _: &str) -> lfm2d::adjudicator::UnregisterOutcome {
        lfm2d::adjudicator::UnregisterOutcome::NotFound
    }
    fn probe(
        &mut self,
        request: &lfm2d::probe_api::ProbeRequest,
        _: &dyn Fn() -> Result<(), Failure>,
    ) -> Result<lfm2d::probe_api::ProbeResponse, Failure> {
        Ok(lfm2d::probe_api::ProbeResponse {
            identity: lfm2d::probe_api::ProbeIdentity {
                model_id: "fixture".into(),
                weight_hash: "hash".into(),
                tokenizer_hash: "tok".into(),
                backend: "cpu".into(),
                dtype: "f32".into(),
                sampling: "greedy".into(),
            },
            // Deliberately echoes the request text into the RESPONSE BODY
            // (never telemetry) so this test also proves the response
            // itself carrying the secret back (expected — the caller sent
            // it) is not confused with a telemetry leak.
            rendered: Some(request.text.clone().unwrap_or_default()),
            rendered_sha256: "0".repeat(64),
            input_tokens: 3,
            cache: lfm2d::probe_api::ProbeCache {
                used_cache: false,
                resumed_spec: None,
                cached_tokens: 0,
                prefill_tokens: 3,
                stepwise_tokens: 0,
            },
            top_logprobs: vec![],
            continuations: None,
            generated: vec![],
            queue_ms: 0.,
            prefill_ms: 1.,
            score_ms: 1.,
        })
    }
}

/// `POST /v1/probe`'s telemetry is emitted on the ADJUDICATOR'S OWN
/// worker OS thread (`Handle::spawn`'s dispatch loop —
/// `tracing::info!("probe complete", ...)`), not on the request-handling
/// async task. `tracing::subscriber::set_default` is thread-local — it
/// would only capture spans/events on the CALLING thread, so a spawned
/// worker thread's own `tracing::info!` calls are invisible to it
/// (confirmed empirically: a minimal repro with a span created on the
/// `set_default` thread, entered on a spawned thread, and an `info!`
/// logged there was NOT captured). This is exactly the gap `/v1/spans`'
/// sibling test (`spans_telemetry_safety.rs`) doesn't have to cross:
/// that worker's telemetry uses `span.record(...)` on a span object bound
/// to the REQUEST-side dispatch, which DOES route correctly cross-thread
/// — but the adjudicator's `Job::Probe` arm uses a fresh `tracing::info!`
/// event, which does not carry a dispatch of its own. So this one test
/// function needs a PROCESS-GLOBAL default (`set_global_default`, valid
/// for the whole process, any thread) rather than `set_default`. Safe in
/// this file specifically because it is the ONLY caller of
/// `set_global_default` here — `expect` on it is deliberate: if a second
/// test in this file ever also tried to install a global default, that
/// should fail loudly (a real test-isolation bug), not silently produce a
/// vacuous capture. The sibling `/v1/tokenize` test below stays on
/// `set_default`: it never crosses a thread, so the cheaper, properly
/// test-scoped mechanism is correct for it.
async fn run_probe_request() -> (StatusCode, Value, Vec<String>) {
    let capture = Capture::default();
    let subscriber = Registry::default().with(capture.clone());
    tracing::subscriber::set_global_default(subscriber)
        .expect("this must be the only test in this binary installing a global default");

    let handle = Handle::spawn(Fake, (&info()).into());
    let router = lfm2d::adjudicator::router(handle, true);
    let (status, body) = post_json(router, "/v1/probe", json!({"text": secret_bearing_text()})).await;

    let captured = capture.0.lock().unwrap().clone();
    (status, body, captured)
}

#[tokio::test]
async fn v1_probe_never_leaks_the_secret_string_into_telemetry() {
    let (status, body, captured) = run_probe_request().await;
    assert_eq!(status, StatusCode::OK, "{body}");
    // The response body legitimately echoes it back (this Fake's
    // `rendered` field) — that is the request/response contract, not a
    // telemetry leak, and is asserted here only to prove the setup is
    // exercising a real secret-bearing round trip.
    assert_eq!(body["rendered"], secret_bearing_text());

    let joined = captured.join("\n");
    assert!(!joined.contains(SECRET), "the secret leaked into telemetry:\n{joined}");
    assert!(
        !joined.contains(&secret_bearing_text()),
        "the full input text leaked into telemetry:\n{joined}"
    );
}

// --------------------------------------------------------- /v1/tokenize

/// Real LFM2.5-8B-A1B tokenizer, no GGUF/model weights — same "Tier 1b"
/// convention `adjudicator::suffix_is_stable_tests`/`tokenize_api::tests`
/// use, so this stays a plain `#[tokio::test]` rather than an
/// `#[ignore]`d real-model one.
fn real_tokenizer() -> tokenizers::Tokenizer {
    let models = std::env::var_os("LFM2_MODELS_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().join(".models")
        });
    let path = std::env::var_os("LFM2D_ADJUDICATOR_TOKENIZER")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| models.join("LFM2.5-8B-A1B/tokenizer.json"));
    assert!(
        path.is_file(),
        "missing tokenizer at {}; set LFM2_MODELS_DIR/LFM2D_ADJUDICATOR_TOKENIZER",
        path.display()
    );
    tokenizers::Tokenizer::from_file(&path).unwrap()
}

async fn run_tokenize_request() -> (StatusCode, Value, Vec<String>) {
    let capture = Capture::default();
    let subscriber = Registry::default().with(capture.clone());
    let _guard = tracing::subscriber::set_default(subscriber);

    let mut registry = TokenizerRegistry::new();
    registry.insert("test-model", real_tokenizer(), "deadbeef");
    let router = lfm2d::tokenize_api::router(Arc::new(registry));
    let (status, body) =
        post_json(router, "/v1/tokenize", json!({"model": "test-model", "text": secret_bearing_text()})).await;

    drop(_guard);
    let captured = capture.0.lock().unwrap().clone();
    (status, body, captured)
}

#[tokio::test]
async fn v1_tokenize_never_leaks_the_secret_string_into_telemetry() {
    let (status, body, captured) = run_tokenize_request().await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(!body["ids"].as_array().unwrap().is_empty(), "test setup must produce real tokens");

    let joined = captured.join("\n");
    assert!(!joined.contains(SECRET), "the secret leaked into telemetry:\n{joined}");
    assert!(
        !joined.contains(&secret_bearing_text()),
        "the full input text leaked into telemetry:\n{joined}"
    );
    // /v1/tokenize's response body ITSELF contains the text broken into
    // token pieces (that is the whole point of the endpoint) — that is
    // not telemetry, so this is not a contradiction of the assertion
    // above, only a note that this endpoint's OWN safety property is
    // narrower than /v1/spans': it never promises not to echo text, only
    // never to put it in a span or log.
    assert!(
        body["tokens"].as_array().unwrap().iter().any(|t| t["token"].as_str().is_some()),
        "sanity: the response does carry token pieces (in the body, where it belongs)"
    );
}
