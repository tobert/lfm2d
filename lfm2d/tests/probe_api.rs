//! `/v1/probe` at the wire, over a fake generator: what the handler refuses
//! before anything reaches the worker (validation, and the whole route when
//! `--no-probe` is set), and what a call that DOES reach the generator
//! carries back. The real engine (warm-prefix resume, continuation scoring,
//! bit-identical parity with `/v1/opinion`) is certified against the
//! generative path by `probe_real.rs`.
use lfm2d::adjudicator::{AdjudicateRequest, AdjudicateResponse, Failure, Generator, Handle, PrefixInfo};
use lfm2d::probe_api::{
    ProbeCache, ProbeContinuationScore, ProbeContinuations, ProbeIdentity, ProbeRequest, ProbeResponse,
};
use std::sync::{Arc, Mutex};
use std::time::Duration;

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

#[derive(Default)]
struct Seen {
    probe_calls: usize,
    last_text: Option<String>,
}

struct Fake(Arc<Mutex<Seen>>);

impl Generator for Fake {
    fn generate(
        &mut self,
        _: &AdjudicateRequest,
        _: &dyn Fn() -> Result<(), Failure>,
    ) -> Result<AdjudicateResponse, Failure> {
        Err(Failure::Internal("this fake only probes".into()))
    }
    fn opine(
        &mut self,
        _: &lfm2d::opinion_api::OpinionRequest,
        _: &[lfm2d::opinion_api::ResolvedQuestion],
        _: &dyn Fn() -> Result<(), Failure>,
    ) -> Result<lfm2d::opinion_api::OpinionResponse, Failure> {
        Err(Failure::Internal("this fake only probes".into()))
    }
    fn register(
        &mut self,
        _: String,
        _: lfm2d::adjudicator::PromptSpec,
        _: &dyn Fn() -> Result<(), Failure>,
    ) -> Result<lfm2d::adjudicator::RegisterOutcome, Failure> {
        Err(Failure::Internal("this fake only probes".into()))
    }
    fn unregister(&mut self, _: &str) -> lfm2d::adjudicator::UnregisterOutcome {
        lfm2d::adjudicator::UnregisterOutcome::NotFound
    }
    fn probe(
        &mut self,
        request: &ProbeRequest,
        check: &dyn Fn() -> Result<(), Failure>,
    ) -> Result<ProbeResponse, Failure> {
        {
            let mut seen = self.0.lock().unwrap();
            seen.probe_calls += 1;
            seen.last_text = request.text.clone();
        }
        // A blocking Generator double: a request naming this exact text
        // never returns until `check()` reports cancellation/deadline —
        // used to prove `/v1/tokenize` shares no queue with this worker
        // (see `tests/tokenize_api.rs`) and, here, that validation refuses
        // a malformed request before ever reaching this point (if it
        // reached here with `text: Some("slow")`, the wire test that
        // expects an immediate 400 would hang instead of failing fast).
        if request.text.as_deref() == Some("slow") {
            loop {
                check()?;
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        Ok(ProbeResponse {
            identity: ProbeIdentity {
                model_id: "fixture".into(),
                weight_hash: "hash".into(),
                tokenizer_hash: "tok".into(),
                backend: "cpu".into(),
                dtype: "f32".into(),
                sampling: "greedy".into(),
            },
            rendered: (request.messages.is_some() || request.ids.is_some()).then(|| "fake rendered".into()),
            rendered_sha256: "0".repeat(64),
            input_tokens: 3,
            cache: ProbeCache {
                used_cache: false,
                resumed_spec: None,
                cached_tokens: 0,
                // Echoes decode_from/decode_from_token back as a fake "1
                // stepwise token" so a wire test can see the field
                // actually reached the generator, without this double
                // needing to tokenize anything for real.
                prefill_tokens: if request.decode_from.is_some() || request.decode_from_token.is_some() {
                    2
                } else {
                    3
                },
                stepwise_tokens: if request.decode_from.is_some() || request.decode_from_token.is_some() {
                    1
                } else {
                    0
                },
            },
            top_logprobs: vec![],
            continuations: (!request.continuations.is_empty()).then(|| ProbeContinuations {
                options: request
                    .continuations
                    .iter()
                    .map(|text| ProbeContinuationScore {
                        text: text.clone(),
                        tokens: vec![1, 2],
                        token_logprobs: vec![-0.1, -0.2],
                        sequence_logprob: -0.3,
                        first_logprob: -0.1,
                        canonical: true,
                    })
                    .collect(),
                sequence_mass: -0.1,
                prob: request.continuations.iter().map(|_| 1.0 / request.continuations.len() as f32).collect(),
            }),
            generated: vec![],
            queue_ms: 0.,
            prefill_ms: 1.,
            score_ms: 1.,
        })
    }
}

fn spawn() -> (Handle, Arc<Mutex<Seen>>) {
    let seen = Arc::new(Mutex::new(Seen::default()));
    let handle = Handle::spawn(Fake(seen.clone()), (&info()).into());
    (handle, seen)
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
async fn a_wellformed_probe_reaches_the_generator_and_its_response_round_trips() {
    let (handle, seen) = spawn();
    let router = lfm2d::adjudicator::router(handle, true);
    let (status, value) = post(&router, "/v1/probe", r#"{"text":"cargo clean"}"#).await;
    assert_eq!(status, 200, "{value}");
    assert_eq!(value["model_id"], "fixture");
    assert_eq!(value["cache"]["used_cache"], false);
    assert_eq!(seen.lock().unwrap().probe_calls, 1);
    assert_eq!(seen.lock().unwrap().last_text.as_deref(), Some("cargo clean"));
}

#[tokio::test]
async fn continuations_and_messages_pass_through_to_the_generator_and_back() {
    let (handle, _seen) = spawn();
    let router = lfm2d::adjudicator::router(handle, true);
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "hi"}],
        "continuations": ["allow", "deny"],
    })
    .to_string();
    let (status, value) = post(&router, "/v1/probe", &body).await;
    assert_eq!(status, 200, "{value}");
    assert_eq!(value["rendered"], "fake rendered");
    assert_eq!(value["continuations"]["options"].as_array().unwrap().len(), 2);
    assert_eq!(value["continuations"]["prob"].as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn decode_from_reaches_the_generator_and_the_schedule_comes_back() {
    let (handle, seen) = spawn();
    let router = lfm2d::adjudicator::router(handle, true);
    let (status, value) = post(&router, "/v1/probe", r#"{"text":"cargo clean","decode_from":6}"#).await;
    assert_eq!(status, 200, "{value}");
    assert_eq!(seen.lock().unwrap().probe_calls, 1);
    assert_eq!(value["cache"]["prefill_tokens"], 2);
    assert_eq!(value["cache"]["stepwise_tokens"], 1);
}

/// F3 (kaibo review, 2026-09-23): the exact-ids form, immune to the
/// text-form re-tokenization caveat — `ids`/`decode_from_token` reach the
/// generator and the response echoes the schedule and a decoded
/// `rendered` string, same as `messages` does.
#[tokio::test]
async fn ids_form_reaches_the_generator_with_its_own_schedule_field() {
    let (handle, seen) = spawn();
    let router = lfm2d::adjudicator::router(handle, true);
    let (status, value) =
        post(&router, "/v1/probe", r#"{"ids":[1,2,3],"decode_from_token":2}"#).await;
    assert_eq!(status, 200, "{value}");
    assert_eq!(seen.lock().unwrap().probe_calls, 1);
    assert_eq!(value["rendered"], "fake rendered", "ids form echoes a decoded rendered string too");
    assert_eq!(value["cache"]["prefill_tokens"], 2);
    assert_eq!(value["cache"]["stepwise_tokens"], 1);

    // Without decode_from_token, ids still work (all-bulk schedule).
    let (status, value) = post(&router, "/v1/probe", r#"{"ids":[1,2,3]}"#).await;
    assert_eq!(status, 200, "{value}");
}

#[tokio::test]
async fn malformed_probe_requests_never_reach_the_generator() {
    let (handle, seen) = spawn();
    let router = lfm2d::adjudicator::router(handle, true);
    let cases: &[(&str, &str)] = &[
        ("neither text nor messages", r#"{}"#),
        ("both text and messages", r#"{"text":"x","messages":[{"role":"user","content":"hi"}]}"#),
        ("empty text", r#"{"text":""}"#),
        ("prefill with text", r#"{"text":"x","assistant_prefill":"y"}"#),
        ("unknown message role", r#"{"messages":[{"role":"narrator","content":"hi"}]}"#),
        ("top_k over the cap", r#"{"text":"x","top_k":21}"#),
        (
            "too many continuations",
            &serde_json::json!({"text": "x", "continuations": vec!["a"; 33]}).to_string(),
        ),
        ("empty continuation", r#"{"text":"x","continuations":["a",""]}"#),
        ("generate over the cap", r#"{"text":"x","generate":257}"#),
        ("timeout_ms zero", r#"{"text":"x","timeout_ms":0}"#),
        ("text and ids together", r#"{"text":"x","ids":[1,2,3]}"#),
        ("empty ids", r#"{"ids":[]}"#),
        ("decode_from with ids", r#"{"ids":[1,2,3],"decode_from":1}"#),
        ("decode_from_token with text", r#"{"text":"x","decode_from_token":1}"#),
        ("decode_from_token past ids.len()", r#"{"ids":[1,2,3],"decode_from_token":4}"#),
        ("assistant_prefill with ids", r#"{"ids":[1,2,3],"assistant_prefill":"y"}"#),
    ];
    for (label, body) in cases {
        let (status, value) = post(&router, "/v1/probe", body).await;
        assert_eq!(status, 400, "{label}: {value}");
    }
    assert_eq!(
        seen.lock().unwrap().probe_calls,
        0,
        "not one malformed request above should have reached the generator (the blocking \
         'slow' double would hang this test if one did)"
    );
}

#[tokio::test]
async fn probe_disabled_makes_the_route_entirely_absent() {
    let (handle, seen) = spawn();
    let router = lfm2d::adjudicator::router(handle, false);
    let (status, _) = post(&router, "/v1/probe", r#"{"text":"cargo clean"}"#).await;
    assert_eq!(status, 404, "a disabled probe is an absent route, never a present-but-403 one");
    assert_eq!(seen.lock().unwrap().probe_calls, 0);
    // Every other adjudicator route must still be present and unaffected —
    // this flag turns off exactly one route, nothing else.
    let (status, _) = post(&router, "/v1/adjudicate", r#"{"input":"x"}"#).await;
    assert_ne!(status, 404, "disabling probe must not disable /v1/adjudicate");
}

#[tokio::test]
async fn probe_shares_the_adjudicator_worker_deadline() {
    let (handle, _seen) = spawn();
    let router = lfm2d::adjudicator::router(handle, true);
    let (status, value) = post(
        &router,
        "/v1/probe",
        r#"{"text":"slow","timeout_ms":50}"#,
    )
    .await;
    assert_eq!(status, 504, "{value}");
}

/// Guards against a request-shape hazard the handler never checks
/// explicitly, because [`ProbeMessage`] round-trips through JSON like any
/// other type — pinned here so a future refactor that loosens
/// `deny_unknown_fields` is caught at the wire, not just in a unit test.
#[tokio::test]
async fn an_unknown_field_on_a_message_is_refused() {
    let (handle, seen) = spawn();
    let router = lfm2d::adjudicator::router(handle, true);
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "hi", "extra": "nope"}]
    })
    .to_string();
    let (status, value) = post(&router, "/v1/probe", &body).await;
    assert_eq!(status, 400, "{value}");
    assert_eq!(seen.lock().unwrap().probe_calls, 0);
}
