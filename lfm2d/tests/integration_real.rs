//! End-to-end smoke test against a REAL checkpoint — the one place in this
//! crate's test suite that loads actual weights through the actual HTTP
//! router. Everything else (`contract_serde.rs`, `router_stub.rs`,
//! `serve_transports.rs`) deliberately avoids candle entirely so the bulk
//! of the suite stays fast; this file is the one check that the real
//! `engine_real::RealEngine` wiring — `Lfm2SequenceClassifier::from_dir`,
//! weight hashing, the worker thread, the router — actually holds together
//! end to end.
//!
//! Gated the same way `tests/cascade.rs`/`tests/trunk_parity.rs` gate in
//! the parent crate: FAIL LOUDLY with a fetch/pointer command rather than
//! silently skip, via `assert!` on the checkpoint file's presence before
//! ever loading it. `LFM2_SEQ_CLF_DIR`/`LFM2_MODELS_DIR` env vars override
//! the defaults, matching `tests/cascade.rs`'s own convention exactly (this
//! is the same `kube_ordinal_v6` checkpoint that file uses).

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use tower::ServiceExt;

use lfm2d::config::Cli;
use lfm2d::engine_real::RealEngine;
use lfm2d::server::{build_router, AppState};
use lfm2d::worker::WorkerHandle;

/// `kube_ordinal_v6`'s directory — not under `.models/` (which holds
/// LiquidAI's own Hub checkpoints), same as `tests/cascade.rs`'s
/// `classifier_dir`.
fn classifier_dir() -> PathBuf {
    if let Ok(d) = std::env::var("LFM2_SEQ_CLF_DIR") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").expect("HOME must be set to locate the default checkpoint");
    PathBuf::from(home).join(".local/share/lfm2-training-data/runs/kube_ordinal_v6")
}

fn cli_with_classifier_only() -> Cli {
    let dir = classifier_dir();
    assert!(
        dir.join("model.safetensors").is_file(),
        "missing weights at {}\n\n  (point LFM2_SEQ_CLF_DIR at an \
         Lfm2BidirForSequenceClassification checkpoint dir — kube_ordinal_v6's shape, \
         produced by training/finetune_sequence_classifier.py; tests/cascade.rs in the \
         parent crate uses this same checkpoint)\n",
        dir.display(),
    );
    Cli {
        embedder_dir: None,
        classifier_dir: Some(dir),
        router_dir: None,
        candidate_classifier_dir: None,
        token_classifier_dir: Vec::new(),
        log_input_hash: false,
        cascade_routes: Vec::new(),
        cascade_severe_labels: vec!["mutating".to_string(), "destructive".to_string()],
        socket_path: None,
        bind_addr: None,
        dtype: lfm2d::config::DtypeArg::F32,
        device: lfm2d::device::DeviceArg::Cpu,
        device_index: 0,
        threads: None,
    }
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
    let json: Value = serde_json::from_slice(&bytes).unwrap();
    (status, json)
}

#[tokio::test]
async fn real_classifier_serves_readyz_models_and_a_summing_softmax() {
    let cli = cli_with_classifier_only();
    let engine = RealEngine::load(&cli).expect("loading the real kube_ordinal_v6 checkpoint");
    let worker = WorkerHandle::spawn(engine);
    let router = build_router(AppState { worker, ready: Arc::new(AtomicBool::new(true)) });

    // --- /readyz ---
    let resp = router
        .clone()
        .oneshot(Request::builder().uri("/readyz").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // --- /v1/models: one classifier, real 64-hex weight hash, real labels ---
    let resp = router
        .clone()
        .oneshot(Request::builder().uri("/v1/models").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let models: Value = serde_json::from_slice(&bytes).unwrap();
    let models = models.as_array().expect("array of models");
    assert_eq!(models.len(), 1);
    assert_eq!(models[0]["kind"], "classifier");
    assert_eq!(models[0]["hidden_size"], 1024, "every LFM2.5 encoder in this family is hidden=1024");

    let hash = models[0]["weight_hash"].as_str().expect("weight_hash is a string");
    assert_eq!(hash.len(), 64, "sha256 hex must be 64 chars, got {hash:?}");
    assert!(hash.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()), "{hash:?}");

    let labels: Vec<String> = models[0]["labels"]
        .as_array()
        .expect("classifier carries labels")
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    // Exact id order, not a sorted set: v6's id2label is alphabetical, so
    // `destructive` sits at index 0 — it VIOLATES the ascending-severity
    // ladder v8+ checkpoints follow (docs/integration.md, invariant 3).
    // An ordinal-position consumer pointed at v6 would read its most
    // destructive label as "allow"; this assertion keeps that hazard
    // visible against the real checkpoint, not just the committed fixture.
    assert_eq!(
        labels,
        vec!["destructive".to_string(), "informative".to_string(), "mutating".to_string()],
        "kube_ordinal_v6's real label set, in id (alphabetical, NOT severity) order"
    );

    // --- /v1/classify: a real forward pass, full 3-label softmax summing to ~1.0 ---
    let (status, body) = post_json(
        router.clone(),
        "/v1/classify",
        json!({"inputs": ["kubectl delete namespace staging --wait=false"]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let result = &body[0];
    let scores = result["scores"].as_object().expect("scores object");
    assert_eq!(scores.len(), 3, "must cover ALL labels, not just top-1");

    let sum: f64 = scores.values().map(|v| v.as_f64().unwrap()).sum();
    assert!((sum - 1.0).abs() < 1e-3, "softmax must sum to ~1.0, got {sum}");

    // top must be the argmax of scores (internally consistent).
    let top = result["top"].as_str().expect("top is a string");
    let (argmax_label, _) = scores
        .iter()
        .max_by(|a, b| a.1.as_f64().unwrap().total_cmp(&b.1.as_f64().unwrap()))
        .unwrap();
    assert_eq!(top, argmax_label);

    assert_eq!(result["model_id"], models[0]["id"]);
    assert_eq!(result["weight_hash"], hash);
}
