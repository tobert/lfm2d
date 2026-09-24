//! The axum `Router` and its handlers. Handlers do exactly three things:
//! validate the request shape (empty-input 400s that don't need the worker
//! to answer), forward to [`WorkerHandle`], and translate the result into
//! the wire types in [`crate::types`]. No model logic, no aggregation
//! logic — the library owns both.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use axum::extract::{FromRequest, MatchedPath, Request, State};
use axum::http::{HeaderValue, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use tracing::Instrument as _;
use tracing_opentelemetry::OpenTelemetrySpanExt as _;

use crate::shutdown::{ShutdownHandle, ShutdownSignal};
use crate::types::{
    ApiError, ClassifyRequest, ClassifyResult, EmbedRequest, ModelInfo, PredictRequest, RouteRequest, RouteResponse, SpansRequest,
};
use crate::worker::{WorkerError, WorkerHandle};

/// Shared handler state. `ready` is a separate flag from "the worker
/// exists" so a test can construct a not-yet-ready server and assert
/// `/readyz`'s 503 — in production this flips true once, at process start,
/// before the socket is ever bound (loading is synchronous, no lazy
/// loading), but the k8s-probe distinction from `/healthz` (process alive)
/// is kept because that's the correct probe semantics regardless of how
/// fast this particular process happens to become ready.
pub struct AppState {
    pub worker: WorkerHandle,
    pub ready: Arc<AtomicBool>,
}

/// Build the full API router over `state`. Called once by `main.rs` for
/// production (serving both a Unix socket and TCP off the SAME router
/// instance — see [`serve`]) and once per test in `tests/router_stub.rs`
/// (served in-process via `tower::ServiceExt::oneshot`, no socket at all).
pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        .route("/v1/models", get(list_models))
        .route("/embed", post(embed))
        .route("/predict", post(predict))
        .route("/v1/classify", post(classify))
        .route("/v1/route", post(route))
        .route("/v1/spans", post(spans))
        .route("/v1/spans/credentials", post(spans_credentials))
        .with_state(Arc::new(state))
        .layer(axum::middleware::from_fn(telemetry_middleware))
}

/// Wraps every request (including an unmatched-route 404 — this is a
/// `Router`-wide `.layer()`, not a per-route one) in one tracing span
/// carrying `method`/`route`/`status`, and records the
/// `lfm2d.requests` counter + `lfm2d.request.duration` histogram by
/// route+status. Unknown paths use a bounded label; arbitrary request paths
/// may carry private data and must not become trace or metric attributes.
pub(crate) async fn telemetry_middleware(req: Request, next: Next) -> Response {
    let method = req.method().as_str().to_string();
    let route = req
        .extensions()
        .get::<MatchedPath>()
        .map(|path| path.as_str())
        .unwrap_or("<unmatched>")
        .to_string();
    let span = tracing::info_span!(
        parent: None,
        "http_request",
        otel.kind = "server",
        http.request.method = %method,
        http.route = %route,
        http.response.status_code = tracing::field::Empty,
        otel.status_code = tracing::field::Empty,
        method = %method,
        route = %route,
        status = tracing::field::Empty,
    );
    // Explicit W3C extraction works independently of global propagator
    // installation. An empty base prevents unrelated task-local context from
    // parenting requests that lack a valid traceparent. Without an OTEL layer,
    // set_parent is a documented no-op error (stderr-only mode).
    let _ = span.set_parent(crate::telemetry::extract_parent(req.headers()));
    let start = Instant::now();
    let resp = next.run(req).instrument(span.clone()).await;
    let elapsed = start.elapsed();
    let status = resp.status().as_u16();
    span.record("status", status);
    span.record("http.response.status_code", i64::from(status));
    if resp.status().is_server_error() {
        span.record("otel.status_code", "ERROR");
    }
    crate::telemetry::record_request(&route, &method, status, elapsed);
    resp
}

/// Flip `/readyz` to 503 and request [`serve`]'s graceful shutdown, in that
/// order — a caller (SIGTERM/SIGINT in `main.rs`, or a test driving this
/// directly) must never trigger the listener shutdown before readiness
/// flips, or a load balancer could still route a fresh request in during
/// the small window between the two. Takes `ready` directly (rather than
/// `&AppState`) so a caller that has already moved its `AppState` into
/// [`build_router`] — which is every caller, `main.rs` included — can still
/// invoke this using the `Arc<AtomicBool>` clone it kept beforehand.
pub fn begin_shutdown(ready: &AtomicBool, shutdown: &ShutdownHandle) {
    ready.store(false, Ordering::SeqCst);
    shutdown.trigger();
}

// ------------------------------------------------------------- extractor

/// `axum::Json<T>`, but a parse failure produces THIS crate's
/// `{"error": {"message", "type"}}` shape (400) instead of axum's default
/// plain-text rejection body. Every handler below uses this for request
/// bodies so "malformed input is 400" holds for EVERY failure mode
/// (missing field, wrong type, invalid UTF-8, not JSON at all), not just
/// the ones each handler happens to check explicitly.
pub struct ValidJson<T>(pub T);

impl<S, T> FromRequest<S> for ValidJson<T>
where
    T: serde::de::DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = ApiErrorResponse;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        match Json::<T>::from_request(req, state).await {
            Ok(Json(value)) => Ok(ValidJson(value)),
            Err(rejection) => Err(bad_request(rejection.body_text())),
        }
    }
}

// ------------------------------------------------------------ error shape

/// An HTTP status paired with the `{"error": {...}}` body — every error
/// path in this module produces one of these, so the shape is uniform
/// regardless of which handler or which failure raised it.
pub struct ApiErrorResponse(StatusCode, ApiError);

impl IntoResponse for ApiErrorResponse {
    fn into_response(self) -> Response {
        (self.0, Json(self.1)).into_response()
    }
}

fn bad_request(message: impl Into<String>) -> ApiErrorResponse {
    ApiErrorResponse(StatusCode::BAD_REQUEST, ApiError::bad_request(message))
}

/// [`WorkerError::BadRequest`] → 400 (a caller mistake, or a call against a
/// head this server instance never loaded). [`WorkerError::Internal`] → 500
/// (a loaded model's forward pass failed, or the worker thread itself is
/// gone) — never a silently-wrong 200, per this crate's house rule.
fn map_worker_error(err: WorkerError) -> ApiErrorResponse {
    match err {
        WorkerError::BadRequest(msg) => bad_request(msg),
        WorkerError::Internal(msg) => ApiErrorResponse(StatusCode::INTERNAL_SERVER_ERROR, ApiError::internal(msg)),
    }
}

/// Stamp the audit pair onto a response as headers — see the crate root
/// docs' "two response conventions on purpose" section for why `/embed`
/// and `/predict` carry `model_id`/`weight_hash` here instead of in the
/// JSON body.
fn attach_audit_headers(resp: &mut Response, model_id: &str, weight_hash: &str) {
    let headers = resp.headers_mut();
    headers.insert(
        "x-model-id",
        HeaderValue::from_str(model_id).unwrap_or_else(|_| HeaderValue::from_static("(unrepresentable-model-id)")),
    );
    headers.insert(
        "x-model-weight-hash",
        HeaderValue::from_str(weight_hash)
            .unwrap_or_else(|_| HeaderValue::from_static("(unrepresentable-weight-hash)")),
    );
}

// -------------------------------------------------------------- handlers

/// `GET /healthz` — process alive. Never touches the worker: a healthz
/// check must answer even if the worker thread is wedged mid-forward-pass.
async fn healthz() -> &'static str {
    "ok"
}

/// `GET /readyz` — 200 once every configured model has loaded, 503 before.
/// See [`AppState::ready`]'s doc comment for why this stays a distinct flag
/// from `/healthz` even though, in production, loading finishes
/// synchronously before the socket is ever bound.
async fn readyz(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if state.ready.load(Ordering::SeqCst) {
        (StatusCode::OK, "ready")
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "loading")
    }
}

/// `GET /v1/models` — every loaded model's identity, kind, weight hash,
/// and (classifier only) label set.
async fn list_models(State(state): State<Arc<AppState>>) -> Result<Json<Vec<ModelInfo>>, ApiErrorResponse> {
    state.worker.list_models().await.map(Json).map_err(map_worker_error)
}

/// `POST /embed` — TEI-compatible-ish: `{"inputs": "text"}` or
/// `{"inputs": ["a","b"]}` → `[[f32,...]]`, one pooled vector per input, in
/// input order. `model_id`/`weight_hash` travel as `X-Model-Id`/
/// `X-Model-Weight-Hash` response headers, not in the body — see the crate
/// root docs.
async fn embed(
    State(state): State<Arc<AppState>>,
    ValidJson(req): ValidJson<EmbedRequest>,
) -> Result<Response, ApiErrorResponse> {
    let inputs = req.inputs.into_vec();
    if inputs.is_empty() {
        return Err(bad_request("inputs must not be empty"));
    }
    let outcome = state.worker.embed(inputs, req.kind).await.map_err(map_worker_error)?;
    let mut resp = Json(outcome.vectors).into_response();
    attach_audit_headers(&mut resp, &outcome.model_id, &outcome.weight_hash);
    Ok(resp)
}

/// `POST /predict` — TEI-ish sequence classification: `{"inputs": ...}` →
/// per input, a list of `{"label", "score"}` covering ALL labels (full
/// softmax, never just top-1), sorted by score descending. Audit pair in
/// headers, same as `/embed`.
async fn predict(
    State(state): State<Arc<AppState>>,
    ValidJson(req): ValidJson<PredictRequest>,
) -> Result<Response, ApiErrorResponse> {
    let inputs = req.inputs.into_vec();
    if inputs.is_empty() {
        return Err(bad_request("inputs must not be empty"));
    }
    let outcome = state.worker.predict(inputs).await.map_err(map_worker_error)?;
    let mut resp = Json(outcome.per_input).into_response();
    attach_audit_headers(&mut resp, &outcome.model_id, &outcome.weight_hash);
    Ok(resp)
}

/// `POST /v1/classify` — our full contract: `{"inputs": [...]}` → per input
/// `{scores: {label: prob}, top, model_id, weight_hash}`. Not TEI-compat,
/// so the audit pair lives directly in the body here (see the crate root
/// docs).
async fn classify(
    State(state): State<Arc<AppState>>,
    ValidJson(req): ValidJson<ClassifyRequest>,
) -> Result<Json<Vec<ClassifyResult>>, ApiErrorResponse> {
    // `Inputs` normalizes the string-or-array forms; a bare string is
    // always one input, so only the explicit-empty-array case can be empty.
    let inputs = req.inputs.into_vec();
    if inputs.is_empty() {
        return Err(bad_request("inputs must not be empty"));
    }
    state.worker.classify(inputs).await.map(Json).map_err(map_worker_error)
}

/// `POST /v1/route` — `{"input": str, "routes": [str,...]}` → per route
/// `{"route", "cosine"}`. RAW COSINE ONLY: this router's softmax is
/// route-count arithmetic and carries no confidence information (see
/// `lfm2_encoder::routing`'s module docs, "the softmax is
/// saturated") — never exposed here.
async fn route(
    State(state): State<Arc<AppState>>,
    ValidJson(req): ValidJson<RouteRequest>,
) -> Result<Json<RouteResponse>, ApiErrorResponse> {
    if req.input.is_empty() {
        return Err(bad_request("input must not be empty"));
    }
    if req.routes.is_empty() {
        return Err(bad_request("routes must not be empty"));
    }
    state.worker.route(req.input, req.routes).await.map(Json).map_err(map_worker_error)
}

/// `POST /v1/spans` — TEI-style `{"inputs": ...}`, optional `"model"` to
/// pick a loaded token-classification head → `[[{"start","end","entity",
/// "score"}, ...], ...]`, one span list per input, BARE array-of-arrays
/// (no wrapper object) same as `/embed`/`/predict`. Audit pair travels as
/// `X-Model-Id`/`X-Model-Weight-Hash` headers, same convention, same
/// reason (see the crate root docs' "two response conventions").
///
/// Model selection: with 2+ `--token-classifier-dir` heads loaded and no
/// `"model"` field, [`WorkerHandle::spans`] → [`crate::engine_real::RealEngine`]
/// returns a 400 that NAMES every loaded id — never a silent "pick the
/// first one" default. With exactly one head loaded, `"model"` is
/// optional.
///
/// **Never returns the matched text** — see [`crate::types::SpanResult`]'s
/// doc comment for why that's a hard rule on this endpoint, not a missing
/// feature.
async fn spans(
    State(state): State<Arc<AppState>>,
    ValidJson(req): ValidJson<SpansRequest>,
) -> Result<Response, ApiErrorResponse> {
    let inputs = req.inputs.into_vec();
    if inputs.is_empty() {
        return Err(bad_request("inputs must not be empty"));
    }
    let outcome = state.worker.spans(inputs, req.model).await.map_err(map_worker_error)?;
    let mut resp = Json(outcome.per_input).into_response();
    attach_audit_headers(&mut resp, &outcome.model_id, &outcome.weight_hash);
    Ok(resp)
}

/// `POST /v1/spans/credentials` — as [`spans`], filtered server-side (via
/// `lfm2_encoder::Lfm2TokenClassifier::credentials`) to ONLY the
/// `credential.*` entity family. A separate endpoint rather than a
/// `/v1/spans?credentials_only=true`-style flag — AWS's precedent of
/// shipping "contains PII" as its own API, not a parameter on the general
/// one, per the crate root docs.
async fn spans_credentials(
    State(state): State<Arc<AppState>>,
    ValidJson(req): ValidJson<SpansRequest>,
) -> Result<Response, ApiErrorResponse> {
    let inputs = req.inputs.into_vec();
    if inputs.is_empty() {
        return Err(bad_request("inputs must not be empty"));
    }
    let outcome = state.worker.spans_credentials(inputs, req.model).await.map_err(map_worker_error)?;
    let mut resp = Json(outcome.per_input).into_response();
    attach_audit_headers(&mut resp, &outcome.model_id, &outcome.weight_hash);
    Ok(resp)
}

// ----------------------------------------------------------------- serve

/// Serve `router` over whichever of `socket_path`/`bind_addr` is `Some`,
/// concurrently if both are given. At least one must be `Some` — enforced
/// by [`crate::config::Cli::validate`] before this is ever called, so this
/// function itself has nothing left to validate.
///
/// An existing file at `socket_path` is removed first: a stale socket from
/// a previous run (crash, kill -9) would otherwise make `UnixListener::bind`
/// fail with "address in use" on a perfectly good path — a daemon restart
/// should not require an operator to manually `rm` a socket file first.
///
/// `shutdown` is handed to EVERY listener's `with_graceful_shutdown` (one
/// clone each): once triggered, each stops accepting new connections and
/// waits for its in-flight requests to finish. `drain_timeout` bounds that
/// wait — Rust runs as PID 1 in the container (see `lfm2d/Containerfile`),
/// so a wedged drain must not hang the process forever; production
/// (`main.rs`) passes [`crate::shutdown::DEFAULT_DRAIN_TIMEOUT`], tests
/// pass a short one to keep the suite fast. Hitting the cap is not treated
/// as an error — this function returns `Ok(())` either way, matching "exit
/// 0 on SIGTERM" from `main.rs`'s perspective.
pub async fn serve(
    router: Router,
    socket_path: Option<PathBuf>,
    bind_addr: Option<String>,
    shutdown: ShutdownSignal,
    drain_timeout: std::time::Duration,
) -> std::io::Result<()> {
    let tcp = match &bind_addr {
        Some(addr) => Some(tokio::net::TcpListener::bind(addr).await?),
        None => None,
    };
    let uds = match &socket_path {
        Some(path) => {
            // Unconditional remove, ignoring NotFound: a check-then-remove
            // pair races with anything else deleting the file in between.
            match std::fs::remove_file(path) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(e),
            }
            Some(tokio::net::UnixListener::bind(path)?)
        }
        None => None,
    };

    let serving = {
        let shutdown = shutdown.clone();
        async move {
            match (tcp, uds) {
                (Some(tcp), Some(uds)) => {
                    let tcp_router = router.clone();
                    let tcp_shutdown = shutdown.clone();
                    let (tcp_res, uds_res) = tokio::join!(
                        axum::serve(tcp, tcp_router).with_graceful_shutdown(tcp_shutdown.wait()),
                        axum::serve(uds, router).with_graceful_shutdown(shutdown.wait()),
                    );
                    tcp_res?;
                    uds_res?;
                    Ok(())
                }
                (Some(tcp), None) => axum::serve(tcp, router).with_graceful_shutdown(shutdown.wait()).await,
                (None, Some(uds)) => axum::serve(uds, router).with_graceful_shutdown(shutdown.wait()).await,
                (None, None) => unreachable!("Cli::validate requires at least one transport"),
            }
        }
    };

    // Hard cap on the drain: once shutdown fires, give in-flight requests
    // `drain_timeout` to finish; if they haven't by then, stop waiting
    // regardless. `main.rs` calls `std::process::exit(0)` right after this
    // resolves either way, so a request still draining past the cap simply
    // doesn't get to finish — an explicit, bounded worst case rather than
    // an unbounded hang.
    let cap = async move {
        shutdown.wait().await;
        tokio::time::sleep(drain_timeout).await;
    };

    tokio::select! {
        res = serving => res,
        () = cap => Ok(()),
    }
}
