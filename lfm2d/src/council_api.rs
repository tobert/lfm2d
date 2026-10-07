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
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, Mutex};

use axum::Router;
use axum::body::Body;
use axum::extract::{DefaultBodyLimit, Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use serde_json::{Value, json};

use crate::adjudicator::{AdjudicatorInfo, Handle};
use crate::council::{
    ContextBody, CouncilError, CouncilSpec, Limits, boundaries, is_spec_id, parse_context_id, render_segments, spec_id,
};
use crate::council_compile::{compile, template};
use crate::council_context::BuildRequest;
use crate::council_wire::{
    Capability, ContextPutResult, ContextState, HeldSpec, Layer, ServerIdentity, SnapshotId, SnapshotInfo,
};

/// A context body larger than this is a `413`.
const MAX_CONTEXT_BYTES: usize = 4 * 1_048_576;

/// What a `PUT` builds when the client names no deadline: the contract has
/// none for a context.
const CONTEXT_TIMEOUT_MS: u64 = 120_000;

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

/// One context the client holds, by the UUID it chose: what it sent, and the
/// engine's ids for the snapshots that build made. The engine ids are
/// internal; the wire's snapshot ids are derived from them ([`wire_id`]).
struct Record {
    head: String,
    snapshots: Vec<RecordSnapshot>,
    pinned: bool,
}

struct RecordSnapshot {
    engine_id: String,
    layer: Layer,
    turn: Option<usize>,
}

/// The records, and how many of them pin each engine id: the engine holds one
/// pin per checkpoint, and two contexts with identical content share a head,
/// so a pin is released only when the last context holding it lets go.
#[derive(Default)]
struct Contexts {
    records: HashMap<String, Record>,
    pins: HashMap<String, usize>,
}

impl Contexts {
    /// Count a pin on `id`.
    fn pin(&mut self, id: &str) {
        *self.pins.entry(id.to_owned()).or_default() += 1;
    }
    /// Drop a pin on `id`; `true` when it was the last, and the engine may
    /// release its own.
    fn unpin(&mut self, id: &str) -> bool {
        match self.pins.get_mut(id) {
            Some(n) if *n > 1 => {
                *n -= 1;
                false
            }
            Some(_) => {
                self.pins.remove(id);
                true
            }
            None => false,
        }
    }
}

/// The client-facing id of a snapshot: a domain-separated hash of the
/// engine's content id, so the engine's own ids stay inside and a snapshot
/// never shares a name with a `/v1/contexts` id.
fn wire_id(engine_id: &str) -> SnapshotId {
    let mut bytes = b"lfm2d-council-snapshot\0".to_vec();
    bytes.extend_from_slice(engine_id.as_bytes());
    SnapshotId::from_digest(&crate::hash::sha256_hex_bytes(&bytes)).expect("a sha256 is a digest")
}

struct Api {
    handle: Handle,
    limits: Limits,
    identity: ServerIdentity,
    held: Mutex<BTreeMap<String, Held>>,
    contexts: Mutex<Contexts>,
}

/// What this server is, as `GET /council/v1/identity` reports it. The
/// capabilities are the ones it has: `dry_run` (a `PUT` reports what it would
/// feed and builds nothing), `describe` (a text question is the engine's
/// describe step) and `leave_one_out` (the pool computes it). Not listed:
/// `warm` and `park` are not built, and `persist` is tolerated, not kept.
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
        capabilities: vec![Capability::DryRun, Capability::Describe, Capability::LeaveOneOut],
    }
}

/// The `/council/v1` routes, over the engine behind `handle`. `info` is that
/// engine's own, so the context limit the identity reports is the one it
/// enforces.
pub fn router(handle: Handle, info: AdjudicatorInfo) -> Router {
    let limits = Limits::new(info.context_limit);
    let api = Arc::new(Api { handle, limits, identity: identity(&info, limits), held: Mutex::new(BTreeMap::new()),
        contexts: Mutex::new(Contexts::default()),
    });
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
        .route(
            "/council/v1/contexts/{id}",
            get(get_context).put(put_context).delete(delete_context).layer(DefaultBodyLimit::disable()),
        )
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

// ---------------------------------------------------------------- contexts

/// A snapshot as the wire lists it.
fn snapshot_info(
    record: &RecordSnapshot,
    engine: &crate::council_context::HeldInfo,
) -> SnapshotInfo {
    SnapshotInfo {
        id: wire_id(&record.engine_id),
        tokens: engine.tokens,
        layer: record.layer,
        turn: record.turn,
        spec_id: None,
        bytes: Some(engine.bytes),
        parked: None,
        pinned: Some(engine.pinned),
    }
}

/// The wire layer and turn index of the snapshot after segment `b`.
fn layer_of(b: usize) -> (Layer, Option<usize>) {
    if b == 0 { (Layer::System, None) } else { (Layer::Turn, Some(b - 1)) }
}

#[allow(clippy::result_large_err)]
async fn put_context(
    State(api): State<Arc<Api>>,
    Path(id): Path<String>,
    headers: HeaderMap,
    body: Body,
) -> Result<Response, CouncilError> {
    parse_context_id(&id)?;
    let bytes = axum::body::to_bytes(body, MAX_CONTEXT_BYTES).await.map_err(|_| {
        CouncilError::too_large(format!("a context is at most {MAX_CONTEXT_BYTES} bytes"), "body")
    })?;
    let put: ContextBody = serde_json::from_slice(&bytes)
        .map_err(|e| CouncilError::bad_request(format!("the body is not a context: {e}")))?;
    put.validate(&api.limits)?;
    if !put.warm.is_empty() {
        return Err(CouncilError::bad_request("warm needs the warm capability, which this server does not list")
            .param("warm"));
    }
    // `If-Match`: the head the client believes the context has.
    let (current, was_pinned) = {
        let contexts = api.contexts.lock().expect("contexts lock poisoned");
        match contexts.records.get(&id) {
            Some(r) => (Some(wire_id(&r.head)), r.pinned),
            None => (None, false),
        }
    };
    if let Some(value) = headers.get("if-match") {
        let text = value.to_str().map_err(|_| CouncilError::bad_request("If-Match is not text").param("If-Match"))?;
        let wanted = SnapshotId::parse(text.trim().trim_matches('"')).map_err(|e| e.param("If-Match"))?;
        match &current {
            Some(head) if *head == wanted => {}
            Some(head) => return Err(CouncilError::precondition_failed(head.as_str())),
            None => return Err(CouncilError::precondition_failed_missing()),
        }
    }
    let segments = render_segments(&put)?;
    let hold_after = boundaries(&put);
    // The engine is only ever told to pin: a release is decided below by the
    // reference counts, because another context with identical content shares
    // this head and its pin. `pin` absent leaves the context as it was, so a
    // pinned one keeps its pin on the new head.
    let pin = (put.pin == Some(true) || (put.pin.is_none() && was_pinned)).then_some(true);
    let request = BuildRequest { segments, hold_after: hold_after.clone(), pin, dry_run: put.dry_run, timeout_ms: CONTEXT_TIMEOUT_MS };
    let outcome = match api.handle.council_put(request).await {
        Ok(outcome) => outcome,
        Err(refusal) => return Err(from_engine(refusal).await),
    };
    let head = outcome.head().clone();
    let pinned = put.pin.unwrap_or(was_pinned);
    let mut releases: Vec<String> = Vec::new();
    if !put.dry_run {
        let mut contexts = api.contexts.lock().expect("contexts lock poisoned");
        let old = contexts.records.insert(
            id.clone(),
            Record {
                head: head.engine_id.clone(),
                snapshots: outcome
                    .snapshots
                    .iter()
                    .map(|s| {
                        let (layer, turn) = layer_of(s.after_segment);
                        RecordSnapshot { engine_id: s.engine_id.clone(), layer, turn }
                    })
                    .collect(),
                pinned,
            },
        );
        if pinned {
            contexts.pin(&head.engine_id);
        }
        // The pin this record held on its old head, now unheld by anyone.
        if let Some(old) = old.filter(|o| o.pinned)
            && contexts.unpin(&old.head)
        {
            releases.push(old.head);
        }
        // An explicit `pin: false` on a head nobody else pins.
        if put.pin == Some(false) && !contexts.pins.contains_key(&head.engine_id) {
            releases.push(head.engine_id.clone());
        }
    }
    releases.sort();
    releases.dedup();
    for engine_id in releases {
        let _ = api.handle.council_unpin(engine_id).await;
    }
    let snapshots = outcome
        .snapshots
        .iter()
        .filter(|s| s.held)
        .map(|s| {
            let (layer, turn) = layer_of(s.after_segment);
            snapshot_info(
                &RecordSnapshot { engine_id: s.engine_id.clone(), layer, turn },
                &crate::council_context::HeldInfo { tokens: s.tokens, bytes: s.bytes, pinned: s.pinned },
            )
        })
        .collect();
    let result = ContextPutResult {
        state: ContextState {
            id,
            head: wire_id(&head.engine_id),
            tokens: outcome.tokens,
            pinned: Some(pinned),
            persist: None,
            snapshots,
        },
        kept: outcome.kept,
        fed: outcome.fed,
        dry_run: put.dry_run,
    };
    Ok(axum::Json(result).into_response())
}

fn context_gone(id: &str) -> CouncilError {
    CouncilError::not_found(format!("no context {id:?} is held: PUT it again")).param("id")
}

#[allow(clippy::result_large_err)]
async fn get_context(State(api): State<Arc<Api>>, Path(id): Path<String>) -> Result<Response, CouncilError> {
    // An id that could name nothing is a 404 like an unknown one.
    if parse_context_id(&id).is_err() {
        return Err(context_gone(&id));
    }
    let (head, listed) = {
        let contexts = api.contexts.lock().expect("contexts lock poisoned");
        let record = contexts.records.get(&id).ok_or_else(|| context_gone(&id))?;
        let listed: Vec<(String, Layer, Option<usize>)> =
            record.snapshots.iter().map(|s| (s.engine_id.clone(), s.layer, s.turn)).collect();
        (record.head.clone(), listed)
    };
    let held = match api.handle.council_inspect(listed.iter().map(|(id, ..)| id.clone()).collect()).await {
        Ok(held) => held,
        Err(refusal) => return Err(from_engine(refusal).await),
    };
    let head_held = listed.iter().zip(&held).any(|((engine_id, ..), h)| *engine_id == head && h.is_some());
    if !head_held {
        // The engine evicted the head: nothing here can be read any more.
        let mut contexts = api.contexts.lock().expect("contexts lock poisoned");
        if let Some(record) = contexts.records.remove(&id)
            && record.pinned
        {
            contexts.unpin(&record.head);
        }
        return Err(context_gone(&id));
    }
    let mut snapshots = Vec::new();
    let mut tokens = 0;
    let mut pinned = false;
    for ((engine_id, layer, turn), info) in listed.into_iter().zip(held) {
        let Some(info) = info else { continue };
        if engine_id == head {
            tokens = info.tokens;
            pinned = info.pinned;
        }
        snapshots.push(snapshot_info(&RecordSnapshot { engine_id, layer, turn }, &info));
    }
    let state = ContextState { id, head: wire_id(&head), tokens, pinned: Some(pinned), persist: None, snapshots };
    Ok(axum::Json(state).into_response())
}

#[allow(clippy::result_large_err)]
async fn delete_context(State(api): State<Arc<Api>>, Path(id): Path<String>) -> Result<Response, CouncilError> {
    if parse_context_id(&id).is_err() {
        return Err(context_gone(&id));
    }
    let release = {
        let mut contexts = api.contexts.lock().expect("contexts lock poisoned");
        let record = contexts.records.remove(&id).ok_or_else(|| context_gone(&id))?;
        (record.pinned && contexts.unpin(&record.head)).then_some(record.head)
    };
    // The record and its pins go; the snapshots stay for the eviction policy,
    // so one another context shares is not lost.
    if let Some(head) = release {
        let _ = api.handle.council_unpin(head).await;
    }
    Ok(StatusCode::NO_CONTENT.into_response())
}
