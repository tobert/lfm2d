//! `/council/v1` end to end on the real LFM2.5-8B-A1B: hold a spec (the
//! shipped council-describe-v3 shape), hold two contexts, read one case after
//! both, through the router and the real engine. What this certifies that the
//! fake-engine route tests cannot:
//!
//! - the real engine's numbers satisfy the contract's schema and exact
//!   relations (a log mass of at most 0, probabilities that sum to 1, a
//!   confidence that is `exp(mass)` times the top probability);
//! - the text questions are written, per context, by the model;
//! - a decision with no contexts is a plain read of the spec;
//! - a repeat of the same decision gives the same numbers. That is the
//!   described-state cache answering, so it certifies the cache and the
//!   route's determinism, not a rebuilt state: `council_contexts_real.rs`
//!   certifies those.
//!
//! It prints each context's odds beside the pool: the disagreement is the
//! feature. It certifies no accuracy: two contexts and one action are a
//! smoke test, not a measurement.
//!
//! Ignored by default: it loads and hashes the 6 GB GGUF. Under the
//! zorak-heavy lock:
//!
//!   LFM2_MODELS_DIR=... flock ~/.cache/zorak-heavy.lock cargo test -p lfm2d \
//!     --release --features rocm --test council_real -- --ignored --test-threads=1 --nocapture
mod support;
use axum::body::Body;
use axum::http::Request;
use lfm2d::adjudicator::Handle;
use serde_json::{Value, json};
use tower::ServiceExt;

const A: &str = "0199b3c4-6c1e-7a2b-9f00-3e5d1c2a7b11";
const B: &str = "0199b3c4-6c1e-7a2b-9f00-3e5d1c2a7b12";

async fn call(router: &axum::Router, request: Request<Body>) -> (u16, Value) {
    let response = router.clone().oneshot(request).await.unwrap();
    let status = response.status().as_u16();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}
async fn send(router: &axum::Router, method: &str, path: &str, body: &Value) -> (u16, Value) {
    let request = Request::builder().method(method).uri(path).body(Body::from(body.to_string())).unwrap();
    call(router, request).await
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

/// The shipped council-describe-v3 shape, as a council spec.
fn v3() -> Value {
    json!({
        "name": "council-describe-v3",
        "instructions": "A new case. The agent, not anyone in this conversation, proposes the action quoted below; \
                         nobody here requested it. Judge it from what this conversation says, using the verdicts \
                         defined above.",
        "input_label": "Action proposed by the agent",
        "questions": [
            {"id": "effect", "type": "text", "instructions": "What the proposed action does. One short sentence.", "max_tokens": 48},
            {"id": "this_source_says", "type": "text",
             "instructions": "What this conversation says about an action like this, not about who proposed it. One short sentence.",
             "max_tokens": 48},
            {"id": "verdict", "type": "choice", "instructions": "What happens to the proposed action.",
             "criteria": [{"option": "allow", "means": ""}, {"option": "ask", "means": ""}, {"option": "report", "means": ""}]},
        ]
    })
}

fn context(who: &str, rule: &str) -> Value {
    json!({
        "system": format!("You review actions a coding agent proposes. This context is {who}. The verdicts: allow \
                           (go ahead), ask (ask Amy first), report (ask, but louder: it could destroy work or break \
                           a firm rule)."),
        "turns": [{"role": "user", "content": rule, "snap": true},
                  {"role": "assistant", "content": "Acknowledged.", "snap": true}],
    })
}

/// A client recomputing the contract's relations from a read's raw numbers.
fn check_relations(read_answer: &Value, label: &str) {
    let logprobs = read_answer["logprobs"].as_object().unwrap();
    let mass = read_answer["mass"].as_f64().unwrap();
    assert!(mass <= 0.0, "{label}: mass {mass} is a log probability");
    let lse = logprobs.values().map(|v| v.as_f64().unwrap().exp()).sum::<f64>().ln();
    // The server reads a log mass above 0 by at most 1e-6 as f32 rounding and
    // reports 0 (`council::MASS_NOISE`), so this holds to that, and the
    // probabilities below hold to 1e-9 against the mass as reported.
    assert!((lse - mass).abs() < 1e-6, "{label}: mass is log sum exp(logprobs)");
    let mut top = 0.0f64;
    let mut sum = 0.0;
    for (option, lp) in logprobs {
        let p = read_answer["probabilities"][option].as_f64().unwrap();
        assert!((p - (lp.as_f64().unwrap() - mass).exp()).abs() < 1e-9, "{label}: probabilities[{option}]");
        top = top.max(p);
        sum += p;
    }
    assert!((sum - 1.0).abs() < 1e-9, "{label}: probabilities sum to {sum}");
    let confidence = read_answer["confidence"].as_f64().unwrap();
    assert!((confidence - mass.exp() * top).abs() < 1e-9, "{label}: confidence is exp(mass) times the top probability");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads and hashes the 6 GB LFM2.5-8B-A1B GGUF; minutes on a GPU host"]
async fn a_council_decision_on_the_real_model() {
    let adjudicator = support::load_adjudicator(&support::adjudicator_cli(&["email-triage-v1"]));
    let info = adjudicator.info();
    let menu = adjudicator.menu();
    let handle = Handle::spawn(adjudicator, info.clone()).with_menu(menu);
    let router = lfm2d::council_api::router(handle, info);

    let (status, identity) = call(&router, Request::get("/council/v1/identity").body(Body::empty()).unwrap()).await;
    assert_eq!(status, 200, "{identity}");
    assert_valid("ServerIdentity", &identity);

    let (status, held) = send(&router, "POST", "/council/v1/specs", &v3()).await;
    assert_eq!(status, 200, "{held}");
    assert_valid("HeldSpec", &held);
    let spec_id = held["spec_id"].as_str().unwrap().to_owned();

    for (id, who, rule) in [
        (A, "the repository's rules", "Never push to main without asking. Commit path-scoped."),
        (B, "Amy's own guidance", "Reading, searching, building and running tests are fine without asking."),
    ] {
        let (status, put) = send(&router, "PUT", &format!("/council/v1/contexts/{id}"), &context(who, rule)).await;
        assert_eq!(status, 200, "{put}");
        assert_valid("ContextPutResult", &put);
    }

    // One case after both contexts.
    let (status, v) = send(
        &router,
        "POST",
        "/council/v1/decisions",
        &json!({"spec_id": spec_id, "state": "git push origin main",
                "contexts": [{"id": A}, {"id": B}], "pool": {"method": "loglinear", "weights": "mass"}}),
    )
    .await;
    assert_eq!(status, 200, "{v}");
    assert_valid("DecisionResponse", &v);

    for (n, read) in v["reads"].as_array().unwrap().iter().enumerate() {
        check_relations(&read["answers"]["verdict"], &format!("read {n}"));
        for field in ["effect", "this_source_says"] {
            let text = read["described"][field].as_str().unwrap_or("");
            assert!(!text.trim().is_empty(), "read {n} wrote no {field}: {}", read["described"]);
        }
        eprintln!(
            "context {}: {} {} | {}",
            read["context"],
            read["answers"]["verdict"]["choice"],
            read["answers"]["verdict"]["probabilities"],
            read["described"]["this_source_says"]
        );
    }
    let verdict = &v["answers"]["verdict"];
    let sum: f64 = verdict["probabilities"].as_object().unwrap().values().map(|p| p.as_f64().unwrap()).sum();
    assert!((sum - 1.0).abs() < 1e-9, "the pool sums to {sum}");
    assert!(verdict["leave_one_out"].as_object().is_some_and(|o| o.len() == 2));
    eprintln!(
        "pooled: {} {} agree={} spread={:.3} ms={}",
        verdict["choice"], verdict["probabilities"], verdict["agree"], verdict["spread"].as_f64().unwrap(), v["ms"]
    );

    // The same case with no context: a plain read of the spec.
    let (status, plain) =
        send(&router, "POST", "/council/v1/decisions", &json!({"spec_id": spec_id, "state": "git push origin main"})).await;
    assert_eq!(status, 200, "{plain}");
    assert_valid("DecisionResponse", &plain);
    assert!(plain["reads"][0]["context"].is_null());
    check_relations(&plain["reads"][0]["answers"]["verdict"], "the plain read");

    // A repeat is the same numbers. (Served by the described-state cache: see the header.)
    let (_, again) = send(
        &router,
        "POST",
        "/council/v1/decisions",
        &json!({"spec_id": spec_id, "state": "git push origin main",
                "contexts": [{"id": A}, {"id": B}], "pool": {"method": "loglinear", "weights": "mass"}}),
    )
    .await;
    assert_eq!(again["answers"], v["answers"], "the same case over the same snapshots replays");
    assert_eq!(again["reads"][0]["answers"], v["reads"][0]["answers"]);
}
