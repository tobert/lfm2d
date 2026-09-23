//! `/v1/tokenize` at the wire: unknown-model handling, response shape, and
//! the load-bearing property the whole endpoint exists for — it answers
//! even while the adjudicator worker is stuck mid-job, because it shares no
//! queue with it. Uses the real LFM2.5-8B-A1B tokenizer (same "Tier 1b"
//! convention as `tests/constrained_decoding.rs`'s `real_vocabulary_*`
//! tests and `adjudicator::suffix_is_stable_tests` — no GGUF, no candle
//! model, so a plain `#[tokio::test]` rather than an `#[ignore]`d one).
use lfm2d::adjudicator::{AdjudicateRequest, AdjudicateResponse, Failure, Generator, Handle, PrefixInfo};
use lfm2d::tokenize_api::TokenizerRegistry;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

fn models_dir() -> PathBuf {
    std::env::var_os("LFM2_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().join(".models"))
}

fn real_tokenizer() -> tokenizers::Tokenizer {
    let path = std::env::var_os("LFM2D_ADJUDICATOR_TOKENIZER")
        .map(PathBuf::from)
        .unwrap_or_else(|| models_dir().join("LFM2.5-8B-A1B/tokenizer.json"));
    assert!(
        path.is_file(),
        "missing tokenizer at {}; set LFM2_MODELS_DIR/LFM2D_ADJUDICATOR_TOKENIZER",
        path.display()
    );
    tokenizers::Tokenizer::from_file(&path).unwrap()
}

fn registry() -> Arc<TokenizerRegistry> {
    let mut reg = TokenizerRegistry::new();
    reg.insert("test-model", real_tokenizer(), "deadbeef".repeat(8));
    Arc::new(reg)
}

async fn post(router: &axum::Router, path: &str, body: &str) -> (u16, serde_json::Value) {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;
    let response = router
        .clone()
        .oneshot(
            Request::post(path)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status().as_u16();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let value = if bytes.is_empty() {
        serde_json::Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap()
    };
    (status, value)
}

#[tokio::test]
async fn unknown_model_is_404() {
    let router = lfm2d::tokenize_api::router(registry());
    let (status, value) = post(&router, "/v1/tokenize", r#"{"model":"nope","text":"hi"}"#).await;
    assert_eq!(status, 404, "{value}");
    assert!(value["error"]["message"].as_str().unwrap().contains("nope"));
}

#[tokio::test]
async fn a_wellformed_request_returns_ids_tokens_and_the_tokenizer_hash() {
    let router = lfm2d::tokenize_api::router(registry());
    let (status, value) =
        post(&router, "/v1/tokenize", r#"{"model":"test-model","text":"hi"}"#).await;
    assert_eq!(status, 200, "{value}");
    assert_eq!(value["model"], "test-model");
    assert_eq!(value["tokenizer_hash"], "deadbeef".repeat(8));
    let ids = value["ids"].as_array().unwrap();
    let tokens = value["tokens"].as_array().unwrap();
    assert_eq!(ids.len(), tokens.len());
    assert!(!ids.is_empty());
    assert_eq!(tokens[0]["start"], 0);
    assert_eq!(tokens.last().unwrap()["end"], 2, "\"hi\" is 2 bytes");
    // `serde_json::Value`'s `[]` indexing returns `Value::Null` for BOTH a
    // genuinely-absent key and a key present with a JSON `null` value, so
    // `value["context"].is_null()` can never distinguish "absent" from
    // "null" and would pass either way — the actual promise
    // (`#[serde(skip_serializing_if = "Option::is_none")]` on
    // `TokenizeResponse::context`) is that the key is OMITTED, checked
    // here by asking the object directly whether the key exists at all.
    assert!(
        !value.as_object().unwrap().contains_key("context"),
        "context must be ABSENT (skip_serializing_if), not present-and-null, when unasked: {value}"
    );
}

#[tokio::test]
async fn a_context_request_adds_the_context_block() {
    let router = lfm2d::tokenize_api::router(registry());
    let (status, value) = post(
        &router,
        "/v1/tokenize",
        r#"{"model":"test-model","text":"world","context":"hello "}"#,
    )
    .await;
    assert_eq!(status, 200, "{value}");
    assert!(value["context"]["suffix_stable"].is_boolean());
    assert!(!value["context"]["ids"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn malformed_requests_are_400() {
    let router = lfm2d::tokenize_api::router(registry());
    let cases: &[(&str, &str)] = &[
        ("empty model", r#"{"model":"","text":"hi"}"#),
        ("empty text", r#"{"model":"test-model","text":""}"#),
        ("missing model field", r#"{"text":"hi"}"#),
        ("unknown field", r#"{"model":"test-model","text":"hi","bogus":1}"#),
    ];
    for (label, body) in cases {
        let (status, value) = post(&router, "/v1/tokenize", body).await;
        assert_eq!(status, 400, "{label}: {value}");
    }
}

// ------------------------------------------------------- doesn't queue

/// The adjudicator's own `Generator`, made to block indefinitely on any
/// call so a request against it occupies the ONE adjudicator worker thread
/// (its queue is bounded and single-threaded — `adjudicator.rs`'s module
/// docs) until cancelled or the process ends. Every method blocks the same
/// way: whichever the test submits, the worker is stuck behind it.
struct BlockingGenerator;
impl Generator for BlockingGenerator {
    fn generate(
        &mut self,
        _: &AdjudicateRequest,
        check: &dyn Fn() -> Result<(), Failure>,
    ) -> Result<AdjudicateResponse, Failure> {
        loop {
            check()?;
            std::thread::sleep(Duration::from_millis(1));
        }
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
        _: &lfm2d::probe_api::ProbeRequest,
        _: &dyn Fn() -> Result<(), Failure>,
    ) -> Result<lfm2d::probe_api::ProbeResponse, Failure> {
        Err(Failure::Internal("not exercised here".into()))
    }
}

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

/// THE mutation-round test: if `/v1/tokenize` were ever wired through the
/// adjudicator's worker queue instead of answered directly on the handler
/// over a cloned tokenizer, this would hang for the full request timeout
/// instead of returning in milliseconds — `BlockingGenerator` never
/// returns, so the ONE adjudicator worker thread is occupied for the rest
/// of the process.
#[tokio::test]
async fn tokenize_answers_while_the_adjudicator_worker_is_stuck_mid_job() {
    let handle = Handle::spawn(BlockingGenerator, info());
    let router = lfm2d::adjudicator::router(handle.clone(), true)
        .merge(lfm2d::tokenize_api::router(registry()));

    // Fire the blocking adjudicate call and don't wait for it: it will
    // never finish (BlockingGenerator loops until the process ends), which
    // is exactly the point — the worker thread is permanently occupied for
    // the rest of this test.
    let blocked_router = router.clone();
    tokio::spawn(async move {
        let _ = post(&blocked_router, "/v1/adjudicate", r#"{"input":"x"}"#).await;
    });
    // Give the spawned task a moment to actually submit and have the
    // worker thread pick it up (single worker, FIFO) before racing it.
    tokio::time::sleep(Duration::from_millis(50)).await;

    let start = Instant::now();
    let (status, value) =
        post(&router, "/v1/tokenize", r#"{"model":"test-model","text":"cargo clean"}"#).await;
    let elapsed = start.elapsed();
    assert_eq!(status, 200, "{value}");
    assert!(
        elapsed < Duration::from_secs(1),
        "tokenize took {elapsed:?} while the adjudicator worker was permanently stuck — it must \
         share no queue with it"
    );
}
