//! [`crate::worker::InferenceEngine`] backed by real candle models —
//! what `main.rs` spawns in production. Loads each configured checkpoint
//! exactly once (via the library's own `from_dir` constructors — this
//! crate never touches a `VarBuilder` or a tensor directly), computes each
//! checkpoint's weight hash, and answers every [`crate::worker::WorkerCommand`]
//! by calling straight through to `candle-lfm2-encoder`'s public API.
//!
//! No trunk sharing: each head loads its own trunk via `from_dir`, even
//! though the library supports `Lfm2Trunk::load_shared` + `from_trunk` for
//! heads that share one checkpoint family. The three heads this daemon
//! serves (a generic embedder, an arbitrary sequence classifier, the
//! Prompt-Router) are independently-trained checkpoints in the general
//! case — `kube_ordinal_v6` and the Prompt-Router do NOT share a trunk (see
//! `src/cascade.rs`'s module docs) — so there is no trunk-sharing
//! opportunity to take here without assuming a same-trunk deployment this
//! service's config doesn't promise. Noted as a judgment call, not an
//! oversight: see `lfm2d/README.md`.

use std::path::Path;

use lfm2_encoder::{
    resolve_severe_labels, Cascade, Lfm2Embedding, Lfm2EncoderConfig,
    Lfm2SequenceClassifier, Lfm2SequenceRouter, Lfm2TokenClassifier, TextKind,
};

use crate::config::Cli;
use crate::hash::sha256_hex_file;
use crate::types::{
    CascadeClause, CascadeLane, CascadeModelRef, CascadeResponse, CascadeWinner, ClassifyResult,
    EmbedKind, LabelScore, ModelInfo, ModelKind, RouteResponse, RouteScore, SpanResult,
};
use crate::worker::{EmbedOutcome, InferenceEngine, PredictOutcome, SpansOutcome, WorkerError};

/// A loaded checkpoint's audit identity, computed once at load time.
#[derive(Debug, Clone)]
struct ModelMeta {
    id: String,
    weight_hash: String,
    /// sha256 of this head's own `tokenizer.json`, hex — feeds
    /// `POST /v1/tokenize`'s `tokenizer_hash` field
    /// (`crate::tokenize_api::TokenizerRegistry`), same convention as
    /// `weight_hash`.
    tokenizer_hash: String,
    hidden_size: usize,
}

/// Directory basename as the model id (`LFM2.5-Embedding-350M`,
/// `kube_ordinal_v6`, ...) — falls back to the full path if the directory
/// has no basename component (e.g. `/`), which should never happen for a
/// real checkpoint dir but must not panic if it somehow does.
fn model_id_from_dir(dir: &Path) -> String {
    dir.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| dir.display().to_string())
}

/// `hidden_size` is read directly from `config.json` rather than from any
/// loaded head object: none of [`Lfm2Embedding`]/[`Lfm2SequenceClassifier`]/
/// [`Lfm2SequenceRouter`]/[`lfm2_encoder::Lfm2Trunk`] exposes it
/// uniformly (`Lfm2Embedding::dim` exists; the others don't), and every
/// head's checkpoint carries this same `config.json` regardless of which
/// head it is, so reading it directly is both simpler and uniform across
/// all three head kinds.
fn read_hidden_size(dir: &Path) -> Result<usize, String> {
    let cfg_path = dir.join("config.json");
    let bytes = std::fs::read(&cfg_path).map_err(|e| format!("reading {}: {e}", cfg_path.display()))?;
    let cfg = Lfm2EncoderConfig::from_json(&bytes).map_err(|e| format!("parsing {}: {e}", cfg_path.display()))?;
    Ok(cfg.hidden_size)
}

fn load_meta(dir: &Path) -> Result<ModelMeta, String> {
    let weights = dir.join("model.safetensors");
    let weight_hash = sha256_hex_file(&weights)
        .map_err(|e| format!("hashing weights at {}: {e}", weights.display()))?;
    let tokenizer_path = dir.join("tokenizer.json");
    let tokenizer_hash = sha256_hex_file(&tokenizer_path)
        .map_err(|e| format!("hashing tokenizer at {}: {e}", tokenizer_path.display()))?;
    Ok(ModelMeta {
        id: model_id_from_dir(dir),
        weight_hash,
        tokenizer_hash,
        hidden_size: read_hidden_size(dir)?,
    })
}

pub struct RealEngine {
    adjudicator_meta: Option<ModelInfo>,
    execution: crate::device::ExecutionDevice,
    dtype: lfm2_encoder::DType,
    embedder: Option<(Lfm2Embedding, ModelMeta)>,
    classifier: Option<(Lfm2SequenceClassifier, ModelMeta)>,
    /// SHADOW second classifier, scored alongside `classifier` on every
    /// `classify`/`cascade` call and immediately discarded after recording
    /// an agreement counter (`telemetry::record_candidate_agreement`) —
    /// never returned to a caller, never listed in `list_models`. See
    /// `--candidate-classifier-dir` in `config.rs` for the full contract.
    candidate_classifier: Option<(Lfm2SequenceClassifier, ModelMeta)>,
    router: Option<(Lfm2SequenceRouter, ModelMeta)>,
    /// Zero, one, or many — mirrors `--token-classifier-dir` being
    /// repeatable, unlike every other head. Order is load order (the
    /// `--token-classifier-dir` flags' order), which is also the order
    /// `list_models` reports them in; `/v1/spans`'s `"model"` selection
    /// looks these up BY ID, never by position, so load order has no
    /// observable effect beyond that listing order.
    token_classifiers: Vec<(Lfm2TokenClassifier, ModelMeta)>,
    cascade_routes: Vec<String>,
    cascade_severe_labels: Vec<String>,
}

impl RealEngine {
    /// Load every head named in `cli`. Fails loudly and stops at the FIRST
    /// problem — a bad config, a missing file, an incompatible checkpoint —
    /// rather than starting a server that can only serve some of what it
    /// was told to. No lazy loading: every head is fully loaded (weights
    /// mmapped and read once for hashing) before this returns.
    ///
    /// If `--cascade-route` is configured, this ALSO validates now (not on
    /// first request) that both a classifier and a router are loaded and
    /// that every `--cascade-severe-label` names a real label on the loaded
    /// classifier — a typo'd severe-label name would otherwise silently
    /// zero out every clause's severity score on the first real request
    /// instead of failing at the moment the operator could still fix it.
    pub fn load(cli: &Cli) -> Result<Self, String> {
        // One dtype for every head, deliberately: a mixed-precision set
        // would make the audit story ("this hash, at this precision")
        // per-head, and nothing here needs that yet.
        let dtype = cli.dtype.to_dtype();
        let execution = crate::device::ExecutionDevice::select(cli.device, cli.device_index)?;
        for reason in &execution.selection_reasons {
            // Telemetry is initialized only after the model hashes are known.
            // Emit the device-selection explanation even if model loading fails.
            eprintln!("lfm2d: device selection: {reason}; selected {}", execution.backend.as_str());
        }
        let embedder = match &cli.embedder_dir {
            Some(dir) => {
                let model = Lfm2Embedding::from_dir_with(dir, dtype, &execution.device)
                    .map_err(|e| format!("loading embedder at {}: {e}", dir.display()))?;
                let meta = load_meta(dir)?;
                Some((model, meta))
            }
            None => None,
        };
        let classifier = match &cli.classifier_dir {
            Some(dir) => {
                let model = Lfm2SequenceClassifier::from_dir_with(dir, dtype, &execution.device)
                    .map_err(|e| format!("loading classifier at {}: {e}", dir.display()))?;
                let meta = load_meta(dir)?;
                Some((model, meta))
            }
            None => None,
        };
        let router = match &cli.router_dir {
            Some(dir) => {
                let model = Lfm2SequenceRouter::from_dir_with(dir, dtype, &execution.device)
                    .map_err(|e| format!("loading router at {}: {e}", dir.display()))?;
                let meta = load_meta(dir)?;
                Some((model, meta))
            }
            None => None,
        };
        // Shadow classifier: same loading discipline (fail loudly, fully
        // loaded before serving) as every other head, but never surfaced —
        // see the struct field doc and --candidate-classifier-dir.
        let candidate_classifier = match &cli.candidate_classifier_dir {
            Some(dir) => {
                let model = Lfm2SequenceClassifier::from_dir_with(dir, dtype, &execution.device)
                    .map_err(|e| format!("loading candidate classifier at {}: {e}", dir.display()))?;
                let meta = load_meta(dir)?;
                Some((model, meta))
            }
            None => None,
        };

        // Load every `--token-classifier-dir` in order. Same fail-first
        // discipline as every other head above — a bad checkpoint in the
        // Nth directory must not start a server that only serves the first
        // N-1.
        let mut token_classifiers = Vec::with_capacity(cli.token_classifier_dir.len());
        for dir in &cli.token_classifier_dir {
            let model = Lfm2TokenClassifier::from_dir_with(dir, dtype, &execution.device)
                .map_err(|e| format!("loading token classifier at {}: {e}", dir.display()))?;
            let meta = load_meta(dir)?;
            token_classifiers.push((model, meta));
        }
        // Two directories with the same basename would load fine
        // individually but make `/v1/spans`'s `"model"` selection silently
        // ambiguous — the SECOND one loaded would just never be reachable
        // by name (or worse, ordering-dependent behavior if the lookup were
        // ever changed to "first match"). Fail loudly at startup instead,
        // while the operator can still fix the deploy config, rather than
        // lazily on whichever request happens to name the shadowed id.
        {
            let mut seen: Vec<&str> = Vec::with_capacity(token_classifiers.len());
            for (_, meta) in &token_classifiers {
                if seen.contains(&meta.id.as_str()) {
                    return Err(format!(
                        "two --token-classifier-dir entries resolve to the same model id {:?} \
                         (directory basenames must be unique across all loaded token classifiers)",
                        meta.id
                    ));
                }
                seen.push(&meta.id);
            }
        }

        if !cli.cascade_routes.is_empty() {
            let (classifier_model, _) = classifier.as_ref().ok_or_else(|| {
                "--cascade-route was given but no --classifier-dir was configured: \
                 /v1/cascade needs both a classifier and a router"
                    .to_string()
            })?;
            if router.is_none() {
                return Err(
                    "--cascade-route was given but no --router-dir was configured: \
                     /v1/cascade needs both a classifier and a router"
                        .to_string(),
                );
            }
            resolve_severe_labels(classifier_model.labels(), &cli.cascade_severe_labels)
                .map_err(|e| format!("--cascade-severe-label: {e}"))?;
        }

        let engine = Self {
            adjudicator_meta: None,
            execution,
            dtype,
            embedder,
            classifier,
            candidate_classifier,
            router,
            token_classifiers,
            cascade_routes: cli.cascade_routes.clone(),
            cascade_severe_labels: cli.cascade_severe_labels.clone(),
        };
        engine.smoke_test(dtype)?;
        Ok(engine)
    }

    pub fn register_adjudicator(&mut self, meta: ModelInfo) -> Result<(), String> {
        if self.list_models().iter().any(|m| m.id == meta.id) {
            return Err(format!("duplicate model id {}", meta.id));
        }
        self.adjudicator_meta = Some(meta);
        Ok(())
    }

    pub fn execution_metadata(&self) -> crate::telemetry::ExecutionMetadata {
        self.execution.metadata(self.dtype)
    }

    /// Every loaded head's own tokenizer, cloned out for
    /// `POST /v1/tokenize` (`crate::tokenize_api::TokenizerRegistry`):
    /// `(id, tokenizer, tokenizer_hash)`, keyed the SAME way
    /// [`InferenceEngine::list_models`] keys its `ModelInfo::id` — so a
    /// consumer that read a model's id off `GET /v1/models` can use it here
    /// unchanged. `main.rs` calls this BEFORE handing `self` to
    /// [`crate::worker::WorkerHandle::spawn_crash_on_panic`], which moves
    /// it into the encoder worker thread — same reason
    /// `Adjudicator::tokenizer_clone` is called before `Handle::spawn`. The
    /// shadow `--candidate-classifier-dir` head is deliberately excluded —
    /// it is never listed in `/v1/models` either (see that field's doc
    /// comment), and this endpoint answers to the same id space.
    pub fn tokenizers(&self) -> Vec<(String, tokenizers::Tokenizer, String)> {
        let mut out = Vec::new();
        if let Some((model, meta)) = &self.embedder {
            out.push((meta.id.clone(), model.tokenizer().clone(), meta.tokenizer_hash.clone()));
        }
        if let Some((model, meta)) = &self.classifier {
            out.push((meta.id.clone(), model.tokenizer().clone(), meta.tokenizer_hash.clone()));
        }
        if let Some((model, meta)) = &self.router {
            out.push((meta.id.clone(), model.tokenizer().clone(), meta.tokenizer_hash.clone()));
        }
        for (model, meta) in &self.token_classifiers {
            out.push((meta.id.clone(), model.tokenizer().clone(), meta.tokenizer_hash.clone()));
        }
        out
    }

    pub fn device_selection_reasons(&self) -> &[String] {
        &self.execution.selection_reasons
    }

    /// Run one tiny forward through every loaded head, so a dtype the build
    /// cannot actually compute in fails HERE rather than on the first real
    /// request.
    ///
    /// # Why this exists
    ///
    /// Loading a checkpoint at a given dtype succeeds independently of
    /// whether the ops can run at it. Measured 2026-08-12: `--dtype bf16`
    /// loads cleanly, logs "loaded model", answers `/healthz` and
    /// `/v1/models`, and then returns 500
    /// `unsupported dtype BF16 for op matmul` on every single inference
    /// call. A daemon that reports healthy and fails every request is worse
    /// than one that refuses to start — it passes a rollout's checks and
    /// then breaks production silently.
    ///
    /// This is the same lesson as `--cascade-severe-label`: a configuration
    /// that is individually valid at every step can still compose into an
    /// endpoint that refuses everything. Deliberately a real forward rather
    /// than a dtype allowlist, so it stays correct when candle gains or
    /// loses an op — the build in front of us is the authority, not a list
    /// in this file.
    fn smoke_test(&self, dtype: lfm2_encoder::DType) -> Result<(), String> {
        const PROBE: &str = "ok";
        let fail = |what: &str, e: String| {
            format!(
                "the {what} head loaded at --dtype {dtype:?} but cannot run a forward pass at it: \
                 {e}. Refusing to start: this daemon would answer /healthz and then fail every \
                 inference request. Use --dtype f32 (or f16), or rebuild with support for {dtype:?}."
            )
        };

        if let Some((m, _)) = &self.embedder {
            m.embed(PROBE, TextKind::Document).map_err(|e| fail("embedder", e.to_string()))?;
        }
        if let Some((m, _)) = &self.classifier {
            m.predict(PROBE).map_err(|e| fail("classifier", e.to_string()))?;
        }
        if let Some((m, _)) = &self.candidate_classifier {
            m.predict(PROBE).map_err(|e| fail("candidate classifier", e.to_string()))?;
        }
        if let Some((m, _)) = &self.router {
            m.route_cosines(PROBE, &["a"]).map_err(|e| fail("router", e.to_string()))?;
        }
        for (m, meta) in &self.token_classifiers {
            m.predict(PROBE)
                .map_err(|e| fail(&format!("token classifier {}", meta.id), e.to_string()))?;
        }
        Ok(())
    }

    /// Resolve which loaded token-classification head answers a
    /// `/v1/spans`/`/v1/spans/credentials` call. `model: Some(name)` always
    /// looks up by id (a caller-named id that doesn't exist is a 400 naming
    /// every loaded id, even when only one head is loaded — silently
    /// falling back to "the one head" on a mistyped name would be exactly
    /// the kind of silent-fallback this crate's house rule forbids).
    /// `model: None` is only valid when exactly one head is loaded; with
    /// 2+, it's a 400 naming every loaded id — see the crate root docs' "N
    /// token heads" section.
    fn resolve_token_classifier(&self, model: Option<&str>) -> Result<&(Lfm2TokenClassifier, ModelMeta), WorkerError> {
        if self.token_classifiers.is_empty() {
            return Err(WorkerError::BadRequest(
                "no token classifier configured on this server (--token-classifier-dir)".to_string(),
            ));
        }
        if let Some(name) = model {
            return self.token_classifiers.iter().find(|(_, meta)| meta.id == name).ok_or_else(|| {
                let available: Vec<&str> = self.token_classifiers.iter().map(|(_, m)| m.id.as_str()).collect();
                WorkerError::BadRequest(format!(
                    "no token classifier named {name:?} on this server; loaded: {available:?}"
                ))
            });
        }
        if self.token_classifiers.len() == 1 {
            return Ok(&self.token_classifiers[0]);
        }
        let available: Vec<&str> = self.token_classifiers.iter().map(|(_, m)| m.id.as_str()).collect();
        Err(WorkerError::BadRequest(format!(
            "2+ token classifiers are loaded ({available:?}) — a \"model\" field naming one is required"
        )))
    }
}

/// [`lfm2_encoder::Span`] → the wire [`SpanResult`]: byte offsets and
/// label unchanged, and `score` straight through from the library.
///
/// That score is the MINIMUM softmax confidence over the span's tokens, not
/// a mean — see [`lfm2_encoder::Span::score`]. It reads lower than
/// the equivalent number from tools that average; that is deliberate and
/// must not be "corrected" here.
fn to_wire_span(span: lfm2_encoder::Span) -> SpanResult {
    SpanResult { start: span.start, end: span.end, entity: span.label, score: span.score }
}

/// Build the full per-label score map plus the argmax `(label, score)` in
/// one pass — shared by `predict`/`classify`/`cascade`'s per-clause
/// breakdown, all of which need exactly this from a `probs` vector plus its
/// label names.
fn scores_and_top(labels: &[String], probs: &[f32]) -> (std::collections::BTreeMap<String, f32>, String) {
    let mut map = std::collections::BTreeMap::new();
    let mut top = (String::new(), f32::NEG_INFINITY);
    for (label, &score) in labels.iter().zip(probs) {
        if score > top.1 {
            top = (label.clone(), score);
        }
        map.insert(label.clone(), score);
    }
    (map, top.0)
}

/// One shadow-classifier observation, derived from the candidate head's
/// forward-pass result.
///
/// Extracted from the shadow passes in `classify`/`cascade` so the
/// swallow-vs-record decision has a fast, model-free test rather than one
/// that needs a checkpoint engineered to fail — the same shape
/// `worker_thread_outcome_is_a_crash` uses for the crash-vs-clean-exit
/// decision. The `Err` arm must produce a [`ShadowObservation::Failure`],
/// never nothing: see `candidate_failure_is_observed_not_swallowed`.
#[derive(Debug, PartialEq, Eq)]
enum ShadowObservation {
    Agreement { candidate_top: String },
    Failure { error: String },
}

fn shadow_observation<E: std::fmt::Display>(labels: &[String], probs: Result<Vec<f32>, E>) -> ShadowObservation {
    match probs {
        Ok(probs) => ShadowObservation::Agreement { candidate_top: scores_and_top(labels, &probs).1 },
        Err(e) => ShadowObservation::Failure { error: e.to_string() },
    }
}

impl InferenceEngine for RealEngine {
    fn list_models(&self) -> Vec<ModelInfo> {
        let mut out = Vec::new();
        if let Some((_, meta)) = &self.embedder {
            out.push(ModelInfo {
                id: meta.id.clone(),
                kind: ModelKind::Embedder,
                weight_hash: meta.weight_hash.clone(),
                labels: None,
                hidden_size: meta.hidden_size,
            });
        }
        if let Some((model, meta)) = &self.classifier {
            out.push(ModelInfo {
                id: meta.id.clone(),
                kind: ModelKind::Classifier,
                weight_hash: meta.weight_hash.clone(),
                labels: Some(model.labels().to_vec()),
                hidden_size: meta.hidden_size,
            });
        }
        if let Some((_, meta)) = &self.router {
            out.push(ModelInfo {
                id: meta.id.clone(),
                kind: ModelKind::Router,
                weight_hash: meta.weight_hash.clone(),
                labels: None,
                hidden_size: meta.hidden_size,
            });
        }
        for (model, meta) in &self.token_classifiers {
            out.push(ModelInfo {
                id: meta.id.clone(),
                kind: ModelKind::TokenClassifier,
                weight_hash: meta.weight_hash.clone(),
                labels: Some(model.entity_types().into_iter().map(str::to_string).collect()),
                hidden_size: meta.hidden_size,
            });
        }
        out.extend(self.adjudicator_meta.iter().cloned());
        out
    }

    fn embed(&self, inputs: &[String], kind: EmbedKind) -> Result<EmbedOutcome, WorkerError> {
        let (model, meta) = self
            .embedder
            .as_ref()
            .ok_or_else(|| WorkerError::BadRequest("no embedder configured on this server".to_string()))?;
        let text_kind = match kind {
            EmbedKind::Document => TextKind::Document,
            EmbedKind::Query => TextKind::Query,
        };
        let vectors = inputs
            .iter()
            .map(|text| model.embed(text, text_kind).map_err(WorkerError::from))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(EmbedOutcome { vectors, model_id: meta.id.clone(), weight_hash: meta.weight_hash.clone() })
    }

    fn predict(&self, inputs: &[String]) -> Result<PredictOutcome, WorkerError> {
        let (model, meta) = self.classifier.as_ref().ok_or_else(|| {
            WorkerError::BadRequest("no classifier configured on this server".to_string())
        })?;
        let per_input = inputs
            .iter()
            .map(|text| {
                let probs = model.predict(text).map_err(WorkerError::from)?;
                let mut ranked: Vec<LabelScore> = model
                    .labels()
                    .iter()
                    .zip(probs)
                    .map(|(label, score)| LabelScore { label: label.clone(), score })
                    .collect();
                ranked.sort_by(|a, b| b.score.total_cmp(&a.score));
                Ok(ranked)
            })
            .collect::<Result<Vec<_>, WorkerError>>()?;
        Ok(PredictOutcome { per_input, model_id: meta.id.clone(), weight_hash: meta.weight_hash.clone() })
    }

    fn classify(&self, inputs: &[String]) -> Result<Vec<ClassifyResult>, WorkerError> {
        let (model, meta) = self.classifier.as_ref().ok_or_else(|| {
            WorkerError::BadRequest("no classifier configured on this server".to_string())
        })?;
        let out: Result<Vec<ClassifyResult>, WorkerError> = inputs
            .iter()
            .map(|text| {
                let probs = model.predict(text).map_err(WorkerError::from)?;
                let (scores, top) = scores_and_top(model.labels(), &probs);
                Ok(ClassifyResult { scores, top, model_id: meta.id.clone(), weight_hash: meta.weight_hash.clone() })
            })
            .collect();
        // SHADOW pass: scores the same inputs against candidate_classifier
        // (if configured) purely for local counters, never touching `out`.
        // A candidate forward-pass failure is not propagated -- an
        // experimental second head must never turn into a caller-visible
        // error the primary classifier didn't have -- but it IS counted,
        // because agreement without its denominator is not a measurement.
        if let (Ok(primary), Some((cand_model, cand_meta))) = (&out, &self.candidate_classifier) {
            for (text, result) in inputs.iter().zip(primary) {
                match shadow_observation(cand_model.labels(), cand_model.predict(text)) {
                    ShadowObservation::Agreement { candidate_top } => {
                        crate::telemetry::record_candidate_agreement(&meta.id, &cand_meta.id, &result.top, &candidate_top)
                    }
                    ShadowObservation::Failure { error } => {
                        crate::telemetry::record_candidate_failure(&meta.id, &cand_meta.id, &error)
                    }
                }
            }
        }
        out
    }

    fn route(&self, input: &str, routes: &[String]) -> Result<RouteResponse, WorkerError> {
        let (model, meta) = self
            .router
            .as_ref()
            .ok_or_else(|| WorkerError::BadRequest("no router configured on this server".to_string()))?;
        let cosines = model.route_cosines(input, routes).map_err(WorkerError::from)?;
        let routes = routes
            .iter()
            .zip(cosines)
            .map(|(route, cosine)| RouteScore { route: route.clone(), cosine })
            .collect();
        Ok(RouteResponse { model_id: meta.id.clone(), weight_hash: meta.weight_hash.clone(), routes })
    }

    fn cascade(&self, clauses: &[String]) -> Result<CascadeResponse, WorkerError> {
        let (classifier, classifier_meta) = self.classifier.as_ref().ok_or_else(|| {
            WorkerError::BadRequest("cascade requires a classifier configured on this server".to_string())
        })?;
        let (router, router_meta) = self.router.as_ref().ok_or_else(|| {
            WorkerError::BadRequest("cascade requires a router configured on this server".to_string())
        })?;
        if self.cascade_routes.is_empty() {
            return Err(WorkerError::BadRequest(
                "cascade has no routes configured on this server (--cascade-route)".to_string(),
            ));
        }

        // Rank-then-route: entirely the library's own aggregation, called
        // through unmodified — this engine only reshapes the result into
        // the wire contract below.
        let verdict = Cascade::new(classifier, router)
            .run(clauses, &self.cascade_routes, &self.cascade_severe_labels)
            .map_err(WorkerError::from)?;

        let labels = classifier.labels();
        let clause_rows: Vec<CascadeClause> = verdict
            .clauses
            .iter()
            .enumerate()
            .map(|(index, c)| {
                let (severity_scores, top_severity) = scores_and_top(labels, &c.severity_probs);
                CascadeClause { index, clause: c.clause.clone(), severity_scores, top_severity }
            })
            .collect();

        let winner_row = &verdict.clauses[verdict.winner];
        let (winner_scores, winner_top) = scores_and_top(labels, &winner_row.severity_probs);

        // SHADOW pass on the winning clause only -- that's the text
        // cascade's decision actually hinges on. Same discipline as
        // `classify`: a failure is counted, not propagated, and never
        // touches the response.
        if let Some((cand_model, cand_meta)) = &self.candidate_classifier {
            match shadow_observation(cand_model.labels(), cand_model.predict(&winner_row.clause)) {
                ShadowObservation::Agreement { candidate_top } => {
                    crate::telemetry::record_candidate_agreement(&classifier_meta.id, &cand_meta.id, &winner_top, &candidate_top)
                }
                ShadowObservation::Failure { error } => {
                    crate::telemetry::record_candidate_failure(&classifier_meta.id, &cand_meta.id, &error)
                }
            }
        }

        Ok(CascadeResponse {
            winner: CascadeWinner { index: verdict.winner, clause: winner_row.clause.clone(), severity_scores: winner_scores },
            lane: CascadeLane {
                route: self.cascade_routes[verdict.winner_lane].clone(),
                cosine: winner_row.lane_cosines[verdict.winner_lane],
            },
            clauses: clause_rows,
            models: vec![
                CascadeModelRef { model_id: classifier_meta.id.clone(), weight_hash: classifier_meta.weight_hash.clone() },
                CascadeModelRef { model_id: router_meta.id.clone(), weight_hash: router_meta.weight_hash.clone() },
            ],
        })
    }

    fn spans(&self, inputs: &[String], model: Option<&str>) -> Result<SpansOutcome, WorkerError> {
        let (clf, meta) = self.resolve_token_classifier(model)?;
        let per_input = inputs
            .iter()
            .map(|text| Ok(clf.predict(text).map_err(WorkerError::from)?.into_iter().map(to_wire_span).collect()))
            .collect::<Result<Vec<_>, WorkerError>>()?;
        Ok(SpansOutcome { per_input, model_id: meta.id.clone(), weight_hash: meta.weight_hash.clone() })
    }

    fn spans_credentials(&self, inputs: &[String], model: Option<&str>) -> Result<SpansOutcome, WorkerError> {
        let (clf, meta) = self.resolve_token_classifier(model)?;
        let per_input = inputs
            .iter()
            .map(|text| Ok(clf.credentials(text).map_err(WorkerError::from)?.into_iter().map(to_wire_span).collect()))
            .collect::<Result<Vec<_>, WorkerError>>()?;
        Ok(SpansOutcome { per_input, model_id: meta.id.clone(), weight_hash: meta.weight_hash.clone() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels() -> Vec<String> {
        ["informative", "situation-normal", "data-critical"].iter().map(|s| s.to_string()).collect()
    }

    /// The regression this guards: the shadow pass was
    /// `if let Ok(probs) = cand_model.predict(text)`, so a candidate
    /// forward-pass failure produced NO record at all. The agreement
    /// counter then measures agreement over the candidate's SUCCESSES, and
    /// a candidate failing systematically reads as high agreement over a
    /// silently shrinking denominator — which is exactly the number a
    /// checkpoint-promotion decision would rest on. An `Err` must produce
    /// an observation.
    #[test]
    fn candidate_failure_is_observed_not_swallowed() {
        let obs = shadow_observation::<String>(&labels(), Err("tensor shape mismatch".to_string()));
        assert_eq!(
            obs,
            ShadowObservation::Failure { error: "tensor shape mismatch".to_string() },
            "a failed candidate forward pass must be recorded, never dropped"
        );
    }

    #[test]
    fn candidate_success_names_the_top_label() {
        let obs = shadow_observation::<String>(&labels(), Ok(vec![0.1, 0.2, 0.7]));
        assert_eq!(obs, ShadowObservation::Agreement { candidate_top: "data-critical".to_string() });
    }
}
