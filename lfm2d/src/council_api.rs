//! `/council/v1`: the council contract's HTTP surface.
//!
//! The wire is the contract's (`tests/fixtures/council/`, validated in
//! `tests/council_contract.rs`): client-chosen names, the contract's error
//! bodies and statuses. The engine underneath keeps its own content ids
//! (`SpecMenuEntry::id`, checkpoint ids) and they never cross this module's
//! edge.
//!
//! Specs, here: `POST` compiles a council spec onto the engine's prompt spec
//! ([`crate::council_compile`]), registers that with the engine, and holds
//! the council spec under its own id (the sha256 of its canonical JSON, so a
//! re-submission is the same spec). The engine owns the prompt spec's
//! lifetime and may evict it; this module checks the engine's menu on every
//! read and says a spec is gone when the engine no longer holds it, rather
//! than serving a spec it could not read.
use std::collections::{BTreeMap, HashSet};
use std::sync::{Arc, Mutex};

use axum::Router;
use axum::body::Body;
use axum::extract::{DefaultBodyLimit, Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use serde_json::{Value, json};

use crate::adjudicator::{AdjudicatorInfo, Handle};
use crate::council::{CouncilError, CouncilSpec, Limits, is_spec_id, spec_id};
use crate::council_compile::{compile, template};
use crate::council_wire::{Capability, HeldSpec, ServerIdentity};

/// A spec body larger than this is a `413`. The engine's own spec upload
/// holds the same bound.
const MAX_SPEC_BYTES: usize = 1_048_576;

impl IntoResponse for CouncilError {
    fn into_response(self) -> Response {
        (StatusCode::from_u16(self.status).expect("a council status is a valid status"), axum::Json(self.to_body()))
            .into_response()
    }
}

/// What this module holds for one council spec: the spec as submitted, how it
/// was compiled, and the engine's id for the compiled prompt spec.
struct Held {
    spec: CouncilSpec,
    template: String,
    /// The engine's content id of the compiled prompt spec. Two council
    /// specs that differ only in `name` compile to the same one, so it is
    /// shared and reference-counted by [`Api::held`] itself.
    engine_id: String,
}

struct Api {
    handle: Handle,
    limits: Limits,
    identity: ServerIdentity,
    held: Mutex<BTreeMap<String, Held>>,
}

/// What this server is, as `GET /council/v1/identity` reports it. The
/// capabilities are the ones it has: `describe` (a text question is the
/// engine's describe step) and `leave_one_out` (the pool computes it).
/// `engine` is information only (Amy, 2026-10-07: nothing gates on it) and
/// names the daemon's version, never a commit.
fn identity(info: &AdjudicatorInfo, limits: Limits) -> ServerIdentity {
    ServerIdentity {
        model: info.model_id.clone(),
        aliases: Vec::new(),
        weight_hash: info.weight_hash.clone(),
        tokenizer_hash: info.tokenizer_hash.clone(),
        template: template(),
        engine: format!("lfm2d/{}", env!("CARGO_PKG_VERSION")),
        device: Some(info.device.clone()),
        text_stop: Some("the closing quote of the field's JSON string".into()),
        limits,
        capabilities: vec![Capability::Describe, Capability::LeaveOneOut],
    }
}

/// The `/council/v1` routes, over the engine behind `handle`. `info` is that
/// engine's own, so the context limit the identity reports is the one it
/// enforces.
pub fn router(handle: Handle, info: AdjudicatorInfo) -> Router {
    let limits = Limits::new(info.context_limit);
    let api = Arc::new(Api { handle, limits, identity: identity(&info, limits), held: Mutex::new(BTreeMap::new()) });
    Router::new()
        .route("/council/v1/identity", get(get_identity))
        .route(
            "/council/v1/specs",
            get(list_specs)
                .post(post_spec)
                // The body is read with its own limit below, so an oversized
                // one is a council `413` and not axum's plain-text one.
                .layer(DefaultBodyLimit::disable()),
        )
        .route("/council/v1/specs/{spec_id}", get(get_spec).delete(delete_spec))
        .with_state(api)
        .layer(axum::middleware::from_fn(crate::server::telemetry_middleware))
}

/// An engine refusal, as the council error it means. Statuses the engine
/// does not use for this are not guessed at: they are `internal`, with the
/// status in the message.
async fn from_engine(response: Response) -> CouncilError {
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024).await.unwrap_or_default();
    let message = serde_json::from_slice::<Value>(&bytes)
        .ok()
        .and_then(|v| v["error"]["message"].as_str().map(str::to_owned))
        .unwrap_or_else(|| String::from_utf8_lossy(&bytes).into_owned());
    match status.as_u16() {
        400 | 422 => CouncilError::bad_request(message),
        404 => CouncilError::not_found(message),
        413 => CouncilError::too_large(message, "body"),
        429 => CouncilError::busy(message),
        408 | 503 => CouncilError::unavailable(message),
        504 => CouncilError::timeout(message),
        507 => CouncilError::pin_budget(message),
        500 => CouncilError::internal(message),
        other => CouncilError::internal(format!("the engine answered {other}: {message}")),
    }
}

impl Api {
    fn held_spec(&self, id: &str, held: &Held) -> HeldSpec {
        HeldSpec { spec_id: id.to_owned(), spec: held.spec.clone(), template: held.template.clone() }
    }

    /// Forget the specs whose compiled prompt the engine no longer holds
    /// (evicted, or deleted through the engine's own route).
    fn prune(&self, held: &mut BTreeMap<String, Held>) {
        let live: HashSet<String> = self.handle.menu().into_iter().map(|e| e.id).collect();
        held.retain(|_, h| live.contains(&h.engine_id));
    }

    fn gone(id: &str) -> CouncilError {
        CouncilError::not_found(format!("no spec {id:?} is held: POST it again")).param("spec_id")
    }
}

#[allow(clippy::result_large_err)]
async fn post_spec(State(api): State<Arc<Api>>, body: Body) -> Result<Response, CouncilError> {
    let bytes = axum::body::to_bytes(body, MAX_SPEC_BYTES).await.map_err(|_| {
        CouncilError::too_large(format!("a spec is at most {MAX_SPEC_BYTES} bytes"), "body")
    })?;
    let submitted: Value = serde_json::from_slice(&bytes)
        .map_err(|e| CouncilError::bad_request(format!("the body is not JSON: {e}")))?;
    let spec = CouncilSpec::parse(&submitted, &api.limits)?;
    let id = spec_id(&submitted);
    let compiled = compile(&spec)?;
    let prompt = serde_json::to_vec(&compiled.prompt)
        .map_err(|e| CouncilError::internal(format!("the compiled spec does not serialize: {e}")))?;
    let (_, entry) = match api.handle.register(prompt).await {
        Ok(registered) => registered,
        Err(refusal) => return Err(from_engine(refusal).await),
    };
    let held = Held { spec, template: compiled.template, engine_id: entry.id };
    let body = api.held_spec(&id, &held);
    api.held.lock().expect("held lock poisoned").insert(id, held);
    Ok(axum::Json(body).into_response())
}

async fn get_identity(State(api): State<Arc<Api>>) -> Response {
    axum::Json(api.identity.clone()).into_response()
}

async fn list_specs(State(api): State<Arc<Api>>) -> Response {
    let mut held = api.held.lock().expect("held lock poisoned");
    api.prune(&mut held);
    let specs: Vec<HeldSpec> = held.iter().map(|(id, h)| api.held_spec(id, h)).collect();
    axum::Json(json!({ "specs": specs })).into_response()
}

#[allow(clippy::result_large_err)]
async fn get_spec(State(api): State<Arc<Api>>, Path(id): Path<String>) -> Result<Response, CouncilError> {
    let mut held = api.held.lock().expect("held lock poisoned");
    api.prune(&mut held);
    // An id that could name nothing is a 404 like an unknown one.
    let found = is_spec_id(&id).then(|| held.get(&id)).flatten().ok_or_else(|| Api::gone(&id))?;
    Ok(axum::Json(api.held_spec(&id, found)).into_response())
}

#[allow(clippy::result_large_err)]
async fn delete_spec(State(api): State<Arc<Api>>, Path(id): Path<String>) -> Result<Response, CouncilError> {
    let (engine_id, shared) = {
        let mut held = api.held.lock().expect("held lock poisoned");
        api.prune(&mut held);
        let dropped = is_spec_id(&id).then(|| held.remove(&id)).flatten().ok_or_else(|| Api::gone(&id))?;
        let shared = held.values().any(|h| h.engine_id == dropped.engine_id);
        (dropped.engine_id, shared)
    };
    // Another council spec compiled to the same prompt: the engine keeps it.
    if shared {
        return Ok(StatusCode::NO_CONTENT.into_response());
    }
    match api.handle.unregister(engine_id).await {
        Ok(_) => Ok(StatusCode::NO_CONTENT.into_response()),
        // A boot-time prompt spec the engine will not delete: the council
        // spec is dropped, and the engine's own spec stays as it was.
        Err(r) if r.status() == StatusCode::FORBIDDEN => Ok(StatusCode::NO_CONTENT.into_response()),
        Err(r) => Err(from_engine(r).await),
    }
}
