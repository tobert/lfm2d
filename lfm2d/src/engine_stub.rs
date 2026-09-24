//! [`crate::worker::InferenceEngine`] test double — no candle, no weights,
//! deterministic canned output. Exists so `tests/router_stub.rs` can drive
//! the real `axum` router (real [`crate::worker::WorkerHandle`], real
//! worker thread, real channel plumbing) and assert HTTP-level behavior
//! (status codes, JSON shape, header passthrough, error mapping) in
//! milliseconds, independent of whether any checkpoint is present on disk.
//!
//! Configuration mirrors what `--embedder-dir`/`--classifier-dir`/
//! `--router-dir` would produce in production: each head is independently
//! present-or-absent, so tests can exercise "endpoint called with that head
//! unconfigured" (→ 400) the same way a misconfigured real deployment
//! would.

use crate::types::{
    ClassifyResult, EmbedKind, LabelScore, ModelInfo, ModelKind, RouteResponse, RouteScore,
    SpanResult,
};
use crate::worker::{EmbedOutcome, InferenceEngine, PredictOutcome, SpansOutcome, WorkerError};

/// A fake loaded model's identity — same shape as what a real load would
/// record, without ever touching a file.
#[derive(Debug, Clone)]
pub struct StubModel {
    pub id: String,
    pub weight_hash: String,
    pub hidden_size: usize,
}

impl StubModel {
    pub fn new(id: impl Into<String>) -> Self {
        // A recognizable, obviously-fake but correctly-shaped (64 hex
        // chars) weight hash, distinct per model id so tests can tell
        // models apart by hash alone.
        let id = id.into();
        let weight_hash = format!("{:0<64}", format!("stub-{id}-hash"))
            .chars()
            .map(|c| if c.is_ascii_hexdigit() { c.to_ascii_lowercase() } else { '0' })
            .collect();
        Self { id, weight_hash, hidden_size: 8 }
    }
}

/// A fake loaded token-classification head — same shape as what
/// `RealEngine` records per `--token-classifier-dir` entry
/// ([`crate::engine_real::ModelMeta`] + its `entity_types()`), without ever
/// touching a checkpoint.
#[derive(Debug, Clone)]
pub struct StubTokenClassifier {
    pub model: StubModel,
    /// BIOES-prefix-stripped entity type names — what `GET /v1/models`
    /// reports as `labels` for this head, and what `/v1/spans`'
    /// fake-but-deterministic decode below picks entities from.
    pub entity_types: Vec<String>,
}

impl StubTokenClassifier {
    /// The PII detector's shape at a glance: a handful of representative
    /// entity types spanning both a `credential.*` type (so
    /// `/v1/spans/credentials` tests have something to filter down to) and
    /// a non-credential type (so filtering is actually exercised, not
    /// vacuously true).
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            model: StubModel::new(id),
            entity_types: vec![
                "credential.api_key".to_string(),
                "person.name".to_string(),
                "location.address".to_string(),
            ],
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct StubEngine {
    pub embedder: Option<StubModel>,
    pub classifier: Option<StubModel>,
    /// Order is the label id order — `predict`/`classify` output follows
    /// it, same contract as the real
    /// [`lfm2_encoder::Lfm2SequenceClassifier::labels`].
    pub classifier_labels: Vec<String>,
    pub router: Option<StubModel>,
    /// Zero, one, or many loaded token-classification heads — mirrors
    /// `--token-classifier-dir` being repeatable. Tests exercise both the
    /// "exactly 1, implicit" and "2+, `model` required" selection rules
    /// against this directly (see `router_stub.rs`'s `/v1/spans` tests).
    pub token_classifiers: Vec<StubTokenClassifier>,
    /// When set, every inference call fails with this message as a
    /// [`WorkerError::Internal`] — the test double's way of simulating "a
    /// loaded model's forward pass blew up," for asserting 500 mapping and
    /// the error JSON shape without needing a real model that can be made
    /// to fail on cue.
    pub fail_with: Option<String>,
    /// When set, every inference call panics with this message — simulating
    /// the worker thread itself dying (not a clean `Err`), so tests can
    /// assert both halves of the channel failure contract: the in-flight
    /// request's dropped reply channel and later requests' dead sender.
    pub panic_with: Option<String>,
    /// When set, every inference call blocks (`std::thread::sleep`, since
    /// this runs on the worker OS thread) for this long before doing
    /// anything else — including before `fail_with`/`panic_with` take
    /// effect. Exists so a test can hold a request "in flight" on purpose,
    /// e.g. `tests/graceful_shutdown.rs` proving an in-flight request still
    /// completes with 200 after shutdown has been requested.
    pub delay: Option<std::time::Duration>,
    /// When set, dropping the engine takes time and then leaves a marker —
    /// see [`DropProbe`]. Shared behind an `Arc` so the stub stays `Clone`;
    /// the probe fires when the last clone goes.
    pub drop_probe: Option<std::sync::Arc<DropProbe>>,
}

/// Stands in for an engine whose drop does real work. A GPU engine frees
/// device memory through the driver when it drops, and on ROCm that free
/// faults once `exit`'s atexit teardown has begun (2026-09-13 core dump:
/// `RocmAllocator::release_all` -> `hipFree` on the worker thread, racing
/// `main`'s `exit(0)`). This makes the drop slow and observable, so
/// `tests/shutdown_exit.rs` can prove the process waits for it — no GPU.
#[derive(Debug)]
pub struct DropProbe {
    pub delay: std::time::Duration,
    pub marker: std::path::PathBuf,
}

impl Drop for DropProbe {
    fn drop(&mut self) {
        std::thread::sleep(self.delay);
        std::fs::write(&self.marker, b"engine dropped\n").expect("DropProbe could not write its marker");
    }
}

impl StubEngine {
    /// The everything-loaded configuration most router tests want: one of
    /// each head, three classifier labels (`destructive`/`informative`/
    /// `mutating` — `kube_ordinal_v6`'s real label set).
    pub fn fully_configured() -> Self {
        Self {
            embedder: Some(StubModel::new("stub-embedder")),
            classifier: Some(StubModel::new("stub-classifier")),
            classifier_labels: vec!["destructive".into(), "informative".into(), "mutating".into()],
            router: Some(StubModel::new("stub-router")),
            // Deliberately empty: `v1_models_lists_every_configured_head_with_hash_and_kind`
            // asserts `models.len() == 3` against this exact configuration.
            // Tests that want a token classifier build a `StubEngine`
            // directly (same pattern `v1_models_reflects_a_partially_configured_server`
            // already uses for "only-router").
            token_classifiers: Vec::new(),
            fail_with: None,
            panic_with: None,
            delay: None,
            drop_probe: None,
        }
    }

    /// Resolve which loaded token-classification head answers a
    /// `/v1/spans`/`/v1/spans/credentials` call — mirrors
    /// `engine_real::RealEngine::resolve_token_classifier` exactly (same
    /// rule, same error wording shape) so a router test genuinely exercises
    /// the disambiguation logic, not a stub-only approximation of it.
    fn resolve_token_classifier(&self, model: Option<&str>) -> Result<&StubTokenClassifier, WorkerError> {
        if self.token_classifiers.is_empty() {
            return Err(WorkerError::BadRequest(
                "no token classifier configured on this server (--token-classifier-dir)".to_string(),
            ));
        }
        if let Some(name) = model {
            return self
                .token_classifiers
                .iter()
                .find(|c| c.model.id == name)
                .ok_or_else(|| {
                    let available: Vec<&str> = self.token_classifiers.iter().map(|c| c.model.id.as_str()).collect();
                    WorkerError::BadRequest(format!(
                        "no token classifier named {name:?} on this server; loaded: {available:?}"
                    ))
                });
        }
        if self.token_classifiers.len() == 1 {
            return Ok(&self.token_classifiers[0]);
        }
        let available: Vec<&str> = self.token_classifiers.iter().map(|c| c.model.id.as_str()).collect();
        Err(WorkerError::BadRequest(format!(
            "2+ token classifiers are loaded ({available:?}) — a \"model\" field naming one is required"
        )))
    }

    /// Deterministic, obviously-fake spans: one span per input, covering
    /// the whole (trimmed) text, whose entity type and score both vary with
    /// `text.len()` — enough spread for tests to assert something specific
    /// without needing real inference. NOT meant to resemble a real BIOES
    /// decode; `RealEngine`'s tests (`integration_real.rs`) cover that.
    fn fake_spans(&self, clf: &StubTokenClassifier, text: &str) -> Vec<SpanResult> {
        let trimmed_len = text.trim().len();
        if trimmed_len == 0 {
            return Vec::new();
        }
        let start = text.len() - text.trim_start().len();
        let n = clf.entity_types.len().max(1);
        let entity = clf.entity_types[text.len() % n].clone();
        // Deterministic pseudo-score in [0.5, 1.0) — enough range that a
        // test can assert two different inputs produce different scores
        // without claiming to model real confidence.
        let score = 0.5 + (text.len() % 50) as f32 / 100.0;
        vec![SpanResult { start, end: start + trimmed_len, entity, score }]
    }

    fn check_fail(&self) -> Result<(), WorkerError> {
        if let Some(d) = self.delay {
            std::thread::sleep(d);
        }
        if let Some(msg) = &self.panic_with {
            panic!("{msg}");
        }
        if let Some(msg) = &self.fail_with {
            return Err(WorkerError::Internal(msg.clone()));
        }
        Ok(())
    }

    /// Deterministic, obviously-fake "scores": longer text skews toward the
    /// last label, shorter toward the first — enough spread that tests can
    /// assert a specific label wins without needing real inference.
    fn fake_scores(&self, text: &str) -> Vec<f32> {
        let n = self.classifier_labels.len().max(1);
        let bucket = text.len() % n;
        let mut raw = vec![0.1f32; n];
        raw[bucket] = 1.0;
        let sum: f32 = raw.iter().sum();
        raw.iter().map(|v| v / sum).collect()
    }
}

impl InferenceEngine for StubEngine {
    fn list_models(&self) -> Vec<ModelInfo> {
        let mut out = Vec::new();
        if let Some(m) = &self.embedder {
            out.push(ModelInfo {
                id: m.id.clone(),
                kind: ModelKind::Embedder,
                weight_hash: m.weight_hash.clone(),
                labels: None,
                hidden_size: m.hidden_size,
            });
        }
        if let Some(m) = &self.classifier {
            out.push(ModelInfo {
                id: m.id.clone(),
                kind: ModelKind::Classifier,
                weight_hash: m.weight_hash.clone(),
                labels: Some(self.classifier_labels.clone()),
                hidden_size: m.hidden_size,
            });
        }
        if let Some(m) = &self.router {
            out.push(ModelInfo {
                id: m.id.clone(),
                kind: ModelKind::Router,
                weight_hash: m.weight_hash.clone(),
                labels: None,
                hidden_size: m.hidden_size,
            });
        }
        for clf in &self.token_classifiers {
            out.push(ModelInfo {
                id: clf.model.id.clone(),
                kind: ModelKind::TokenClassifier,
                weight_hash: clf.model.weight_hash.clone(),
                labels: Some(clf.entity_types.clone()),
                hidden_size: clf.model.hidden_size,
            });
        }
        out
    }

    fn embed(&self, inputs: &[String], kind: EmbedKind) -> Result<EmbedOutcome, WorkerError> {
        self.check_fail()?;
        let model = self
            .embedder
            .as_ref()
            .ok_or_else(|| WorkerError::BadRequest("no embedder configured on this server".to_string()))?;
        let kind_component = match kind {
            EmbedKind::Document => 0.0,
            EmbedKind::Query => 1.0,
        };
        let vectors = inputs
            .iter()
            .map(|s| {
                let mut v = vec![0f32; model.hidden_size];
                v[0] = s.len() as f32;
                v[1] = kind_component;
                v
            })
            .collect();
        Ok(EmbedOutcome { vectors, model_id: model.id.clone(), weight_hash: model.weight_hash.clone() })
    }

    fn predict(&self, inputs: &[String]) -> Result<PredictOutcome, WorkerError> {
        self.check_fail()?;
        let model = self.classifier.as_ref().ok_or_else(|| {
            WorkerError::BadRequest("no classifier configured on this server".to_string())
        })?;
        let per_input = inputs
            .iter()
            .map(|text| {
                let scores = self.fake_scores(text);
                let mut ranked: Vec<LabelScore> = self
                    .classifier_labels
                    .iter()
                    .zip(scores)
                    .map(|(label, score)| LabelScore { label: label.clone(), score })
                    .collect();
                ranked.sort_by(|a, b| b.score.total_cmp(&a.score));
                ranked
            })
            .collect();
        Ok(PredictOutcome { per_input, model_id: model.id.clone(), weight_hash: model.weight_hash.clone() })
    }

    fn classify(&self, inputs: &[String]) -> Result<Vec<ClassifyResult>, WorkerError> {
        self.check_fail()?;
        let model = self.classifier.as_ref().ok_or_else(|| {
            WorkerError::BadRequest("no classifier configured on this server".to_string())
        })?;
        inputs
            .iter()
            .map(|text| {
                let scores = self.fake_scores(text);
                let mut map = std::collections::BTreeMap::new();
                let mut top = (String::new(), f32::NEG_INFINITY);
                for (label, score) in self.classifier_labels.iter().zip(scores) {
                    if score > top.1 {
                        top = (label.clone(), score);
                    }
                    map.insert(label.clone(), score);
                }
                Ok(ClassifyResult {
                    scores: map,
                    top: top.0,
                    model_id: model.id.clone(),
                    weight_hash: model.weight_hash.clone(),
                })
            })
            .collect()
    }

    fn route(&self, input: &str, routes: &[String]) -> Result<RouteResponse, WorkerError> {
        self.check_fail()?;
        let model = self
            .router
            .as_ref()
            .ok_or_else(|| WorkerError::BadRequest("no router configured on this server".to_string()))?;
        let base = (input.len() % 7) as f32 / 7.0;
        let routes = routes
            .iter()
            .enumerate()
            .map(|(i, r)| RouteScore { route: r.clone(), cosine: (base - i as f32 * 0.2).clamp(-1.0, 1.0) })
            .collect();
        Ok(RouteResponse { model_id: model.id.clone(), weight_hash: model.weight_hash.clone(), routes })
    }

    fn spans(&self, inputs: &[String], model: Option<&str>) -> Result<SpansOutcome, WorkerError> {
        self.check_fail()?;
        let clf = self.resolve_token_classifier(model)?;
        let per_input = inputs.iter().map(|text| self.fake_spans(clf, text)).collect();
        Ok(SpansOutcome { per_input, model_id: clf.model.id.clone(), weight_hash: clf.model.weight_hash.clone() })
    }

    fn spans_credentials(&self, inputs: &[String], model: Option<&str>) -> Result<SpansOutcome, WorkerError> {
        self.check_fail()?;
        let clf = self.resolve_token_classifier(model)?;
        let per_input = inputs
            .iter()
            .map(|text| self.fake_spans(clf, text).into_iter().filter(|s| s.entity.starts_with("credential.")).collect())
            .collect();
        Ok(SpansOutcome { per_input, model_id: clf.model.id.clone(), weight_hash: clf.model.weight_hash.clone() })
    }
}
