//! The single inference worker: one OS thread owns every loaded model.
//! Handlers never touch a model directly — they send a [`WorkerCommand`]
//! down an unbounded [`std::sync::mpsc`] channel and `.await` a
//! [`tokio::sync::oneshot`] reply. This is deliberate serialization, not an
//! accidental bottleneck: a single forward pass already saturates ~13.6 of
//! this box's cores (measured), so request-level concurrency has no
//! headroom to spend.
//!
//! [`InferenceEngine`] is the seam that lets router/handler tests run
//! without candle: [`crate::engine_real::RealEngine`] backs production,
//! [`crate::engine_stub::StubEngine`] backs the HTTP-contract test suite.
//! Both are driven through the exact same channel/thread plumbing here, so
//! a router test genuinely exercises the same code path production traffic
//! does, up to the engine implementation itself.
//!
//! # Observability
//!
//! Every command carries a `tracing::Span` (created by the async method
//! that enqueues it, so it nests under whatever HTTP-request span is
//! current) and an enqueue timestamp. The worker thread — NOT the async
//! task awaiting the reply — records `queue_wait_ms` (time spent sitting in
//! the channel) and `inference_ms` (time spent inside [`InferenceEngine`])
//! onto that span, and reports `inference_ms` to the
//! `lfm2d.inference.duration` histogram: `tracing::Span` is `Send`/
//! `Clone` and not thread-affine, so recording from the worker OS thread is
//! the intended use, not a hack. A separate `AtomicUsize` (`queue_depth`)
//! backs the `lfm2d.worker.queue_depth` observable gauge — incremented in
//! [`WorkerHandle::send`], decremented the moment a command is dequeued.
//!
//! # Crash-only on worker death
//!
//! [`WorkerHandle::spawn`] never exits the process on its own — every
//! router/HTTP test in this crate relies on being able to make the worker
//! thread panic (simulating "a loaded model's forward pass corrupted
//! something") and observe a clean 500 afterward, in the SAME test
//! process. [`WorkerHandle::spawn_crash_on_panic`] is the production-only
//! variant `main.rs` calls instead: identical worker thread, PLUS a monitor
//! thread that calls `std::process::exit(1)` if (and only if) the worker
//! thread terminates via panic — never on the clean channel-closure exit a
//! graceful shutdown produces. See [`worker_thread_outcome_is_a_crash`] for
//! the (separately unit-tested) decision logic, and
//! `tests/worker_crash_monitor.rs` for the integration test that exercises
//! this through the real compiled binary.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use tokio::sync::oneshot;
use tracing::Span;

use crate::types::{EmbedKind, ModelInfo, RouteResponse, SpanResult};

/// A worker-side failure, already classified into the two HTTP status
/// families the API contract promises: a caller-supplied problem (400) or
/// an internal/model failure (500). Mapped to [`crate::types::ApiError`] at
/// the handler boundary — see `server.rs`.
#[derive(Debug, Clone)]
pub enum WorkerError {
    /// The request itself, or the server's current configuration relative
    /// to it (e.g. `/embed` called with no embedder loaded), makes this
    /// call impossible to satisfy. Never the model's fault.
    BadRequest(String),
    /// A loaded model failed during inference, or the worker thread itself
    /// is gone. Loud, not swallowed: this crate's house rule is that a
    /// worker failure is a 500, never a silently-wrong 200.
    Internal(String),
}

impl WorkerError {
    pub fn message(&self) -> &str {
        match self {
            WorkerError::BadRequest(m) | WorkerError::Internal(m) => m,
        }
    }
}

/// Map the library's own error type onto the two-bucket [`WorkerError`]
/// taxonomy. [`lfm2_encoder::Error::InvalidInput`] is, by the
/// library's own doc comment, "a caller-supplied argument that is
/// structurally invalid" — exactly a 400. Everything else (I/O, config
/// parse, tokenizer, candle numerics) is this SERVICE's problem to explain,
/// not the caller's mistake to fix by rephrasing the request — a 500.
impl From<lfm2_encoder::Error> for WorkerError {
    fn from(err: lfm2_encoder::Error) -> Self {
        match err {
            lfm2_encoder::Error::InvalidInput(msg) => WorkerError::BadRequest(msg),
            other => WorkerError::Internal(other.to_string()),
        }
    }
}

/// Everything [`InferenceEngine::embed`] hands back: the pooled vectors
/// plus the audit pair that travels out-of-band as response headers (see
/// the crate root docs — `/embed`'s JSON body stays a bare TEI-compatible
/// array, so `model_id`/`weight_hash` cannot live in it).
#[derive(Debug, Clone)]
pub struct EmbedOutcome {
    pub vectors: Vec<Vec<f32>>,
    pub model_id: String,
    pub weight_hash: String,
}

/// As [`EmbedOutcome`], for `/v1/spans` and `/v1/spans/credentials` — the
/// same outcome shape backs both, since "credentials" is just `spans`
/// filtered server-side to the `credential.*` family (see
/// [`InferenceEngine::spans_credentials`]).
#[derive(Debug, Clone)]
pub struct SpansOutcome {
    pub per_input: Vec<Vec<SpanResult>>,
    pub model_id: String,
    pub weight_hash: String,
}

/// The engine seam. Every method is synchronous (candle inference is
/// blocking work on either device — there is nothing to `.await` inside it) and
/// runs entirely on the worker thread; `&self` because a loaded model is
/// never mutated after construction.
pub trait InferenceEngine: Send + 'static {
    fn list_models(&self) -> Vec<ModelInfo>;
    fn embed(&self, inputs: &[String], kind: EmbedKind) -> Result<EmbedOutcome, WorkerError>;
    fn route(&self, input: &str, routes: &[String]) -> Result<RouteResponse, WorkerError>;
    /// `model` selects which loaded `--token-classifier-dir` head answers
    /// this call — required (as a 400 naming the loaded ids) when 2+ are
    /// loaded, optional (falls back to "the one loaded head") when exactly
    /// 1 is loaded, and always a 400 if it names an id nothing loaded.
    fn spans(&self, inputs: &[String], model: Option<&str>) -> Result<SpansOutcome, WorkerError>;
    /// As [`Self::spans`], filtered to the `credential.*` entity family —
    /// see `lfm2_encoder::Lfm2TokenClassifier::credentials`.
    fn spans_credentials(&self, inputs: &[String], model: Option<&str>) -> Result<SpansOutcome, WorkerError>;
}

/// Metadata every [`WorkerCommand`] variant carries alongside its
/// request-specific fields — enqueue time (for `queue_wait_ms`) and the
/// span it should record onto (for both `queue_wait_ms` and
/// `inference_ms`; see the module docs' "Observability" section).
pub(crate) struct Enqueued {
    queued_at: Instant,
    span: Span,
}

/// One request to the worker thread, paired with the `oneshot` it must
/// reply on. `pub(crate)` — handlers go through [`WorkerHandle`]'s typed
/// async methods, never build this directly.
pub(crate) enum WorkerCommand {
    ListModels {
        reply: oneshot::Sender<Vec<ModelInfo>>,
        meta: Enqueued,
    },
    Embed {
        inputs: Vec<String>,
        kind: EmbedKind,
        reply: oneshot::Sender<Result<EmbedOutcome, WorkerError>>,
        meta: Enqueued,
    },
    Route {
        input: String,
        routes: Vec<String>,
        reply: oneshot::Sender<Result<RouteResponse, WorkerError>>,
        meta: Enqueued,
    },
    Spans {
        inputs: Vec<String>,
        model: Option<String>,
        reply: oneshot::Sender<Result<SpansOutcome, WorkerError>>,
        meta: Enqueued,
    },
    SpansCredentials {
        inputs: Vec<String>,
        model: Option<String>,
        reply: oneshot::Sender<Result<SpansOutcome, WorkerError>>,
        meta: Enqueued,
    },
}

/// A cheap-to-clone handle to the worker thread. Every axum handler holds
/// one (via `Arc` in [`crate::server::AppState`]); sending a command never
/// blocks the async runtime — the underlying channel is unbounded, and
/// awaiting the `oneshot` reply yields to the executor like any other
/// future.
#[derive(Clone)]
pub struct WorkerHandle {
    tx: std::sync::mpsc::Sender<WorkerCommand>,
    /// Fires once the worker thread has dropped its engine — see [`WorkerExit`].
    exit: WorkerExit,
    /// Commands sent but not yet dequeued by the worker thread — the
    /// `lfm2d.worker.queue_depth` observable gauge reads this (see
    /// `telemetry::register_queue_depth_gauge`). Incremented in
    /// [`Self::send`], decremented at the top of the worker loop's match.
    queue_depth: Arc<AtomicUsize>,
    /// `--log-input-hash` (default `false`): attach a hash of the request's
    /// input TEXT (never the text itself) to the `worker_call` span for
    /// [`Self::spans`]/[`Self::spans_credentials`] ONLY — see
    /// [`Self::with_log_input_hash`] and the module docs' "Observability"
    /// section. Never threaded into any OTLP METRIC attribute (unbounded
    /// per-input cardinality would wreck VictoriaMetrics) — trace/log
    /// attribute only.
    log_input_hash: bool,
}

/// A single dispatch point every `WorkerCommand::Foo { .. }` variant's
/// worker-thread handling goes through: enter the span, record
/// `queue_wait_ms`, decrement the queue-depth gauge, time the actual
/// engine call, record `inference_ms` + the histogram, reply. Written once
/// as a closure-taking helper so the six near-identical match arms below
/// don't each re-implement this sequence (and can't drift out of sync).
fn run_timed<T>(
    meta: Enqueued,
    queue_depth: &AtomicUsize,
    operation: &'static str,
    f: impl FnOnce() -> T,
) -> T {
    // Entering a Span alone does not install its subscriber on this OS
    // thread. Carry the request's dispatch too, so inference events/child
    // spans and parent-span closure use the same subscriber as the caller.
    // Declare this guard before `span`: span closure must run before reset.
    let dispatch = meta.span.with_subscriber(|(_, dispatch)| dispatch.clone());
    let _dispatch_guard = dispatch.as_ref().map(tracing::dispatcher::set_default);
    let Enqueued { queued_at, span } = meta;
    let _guard = span.enter();
    let queue_wait = queued_at.elapsed();
    span.record("queue_wait_ms", queue_wait.as_secs_f64() * 1000.0);
    queue_depth.fetch_sub(1, Ordering::SeqCst);

    let t0 = Instant::now();
    let result = f();
    let inference = t0.elapsed();
    span.record("inference_ms", inference.as_secs_f64() * 1000.0);
    crate::telemetry::record_inference_duration(operation, inference);
    result
}

impl WorkerHandle {
    /// Spawn the worker thread over `engine` and return a handle to it. The
    /// thread runs until every [`WorkerHandle`] clone (and thus every
    /// sender) is dropped, at which point `rx` iteration ends and the
    /// thread exits cleanly. Production wants crash-only semantics on a
    /// worker panic — see [`Self::spawn_crash_on_panic`] — but THIS
    /// function deliberately does not install that behavior: several tests
    /// in `tests/router_stub.rs` intentionally panic the worker thread
    /// in-process to assert the 500 mapping, and must not take the whole
    /// test binary down with it.
    pub fn spawn<E: InferenceEngine>(engine: E) -> Self {
        Self::spawn_inner(engine, false)
    }

    /// As [`Self::spawn`], but additionally installs a monitor thread that
    /// calls `std::process::exit(1)` if the worker thread terminates via
    /// panic. Production-only (`main.rs`'s job, not this module's default):
    /// a worker-thread panic must not leave the HTTP server up answering
    /// 500s forever while `/healthz` still says 200 — kubelet only restarts
    /// a pod whose process actually exits. A clean exit (every
    /// [`WorkerHandle`] clone dropped, e.g. at the end of a graceful
    /// shutdown) is NOT treated as a crash — see
    /// [`worker_thread_outcome_is_a_crash`].
    pub fn spawn_crash_on_panic<E: InferenceEngine>(engine: E) -> Self {
        Self::spawn_inner(engine, true)
    }

    fn spawn_inner<E: InferenceEngine>(engine: E, exit_on_panic: bool) -> Self {
        let (tx, rx) = std::sync::mpsc::channel::<WorkerCommand>();
        let queue_depth = Arc::new(AtomicUsize::new(0));
        let worker_queue_depth = queue_depth.clone();
        let exit = WorkerExit::default();
        let thread_exit = exit.clone();
        let join = std::thread::Builder::new()
            .name("lfm2d-worker".to_string())
            .spawn(move || {
                for cmd in rx {
                    // Every branch replies exactly once. A dropped receiver
                    // (the HTTP request that asked was itself dropped,
                    // e.g. client disconnect) makes `.send()` return `Err`,
                    // which we discard — there is nothing to escalate to,
                    // the asker is gone.
                    match cmd {
                        WorkerCommand::ListModels { reply, meta } => {
                            let _ = reply.send(run_timed(meta, &worker_queue_depth, "list_models", || {
                                engine.list_models()
                            }));
                        }
                        WorkerCommand::Embed { inputs, kind, reply, meta } => {
                            let _ = reply.send(run_timed(meta, &worker_queue_depth, "embed", || {
                                engine.embed(&inputs, kind)
                            }));
                        }
                        WorkerCommand::Route { input, routes, reply, meta } => {
                            let _ = reply.send(run_timed(meta, &worker_queue_depth, "route", || {
                                engine.route(&input, &routes)
                            }));
                        }
                        WorkerCommand::Spans { inputs, model, reply, meta } => {
                            let _ = reply.send(run_timed(meta, &worker_queue_depth, "spans", || {
                                engine.spans(&inputs, model.as_deref())
                            }));
                        }
                        WorkerCommand::SpansCredentials { inputs, model, reply, meta } => {
                            let _ = reply.send(run_timed(meta, &worker_queue_depth, "spans_credentials", || {
                                engine.spans_credentials(&inputs, model.as_deref())
                            }));
                        }
                    }
                }
                // Every sender is gone. Drop the engine HERE, explicitly, and
                // only then report finished: a GPU engine's drop frees device
                // memory through the driver, and `main` must not start
                // `exit`'s teardown until it is done (see `WorkerExit`).
                drop(engine);
                thread_exit.mark();
            })
            .expect("failed to spawn the lfm2d inference worker thread");

        if exit_on_panic {
            std::thread::Builder::new()
                .name("lfm2d-worker-monitor".to_string())
                .spawn(move || {
                    let outcome = join.join();
                    if worker_thread_outcome_is_a_crash(&outcome) {
                        eprintln!(
                            "lfm2d: the inference worker thread panicked — exiting so the \
                             supervisor restarts this pod. An alive-but-broken worker must \
                             never keep answering 500s while /healthz still says 200."
                        );
                        std::process::exit(1);
                    }
                    // Ok(()) means the worker's `for cmd in rx` loop ended
                    // because every WorkerHandle (and thus every sender)
                    // was dropped — the expected end of a graceful
                    // shutdown, already underway on `main`'s task. Nothing
                    // to do here.
                })
                .expect("failed to spawn the lfm2d worker crash monitor thread");
        }

        Self { tx, exit, queue_depth, log_input_hash: false }
    }

    /// Opt into attaching an input-text hash to `/v1/spans`/
    /// `/v1/spans/credentials` trace spans (`--log-input-hash`, default
    /// off) — see [`Self::log_input_hash`]'s doc comment. Consumed builder-
    /// style so the common case (every existing call site: `WorkerHandle::
    /// spawn(engine)` with no chained call) needs no change at all.
    pub fn with_log_input_hash(mut self, enabled: bool) -> Self {
        self.log_input_hash = enabled;
        self
    }

    /// Clone of the queue-depth counter — `main.rs` passes this to
    /// [`crate::telemetry::register_queue_depth_gauge`].
    pub fn queue_depth_handle(&self) -> Arc<AtomicUsize> {
        self.queue_depth.clone()
    }

    /// A waiter for the worker thread's clean finish. Take it BEFORE handing
    /// the handle to the router: the waiter holds no sender, so it never
    /// keeps the worker loop alive itself.
    pub fn exit_signal(&self) -> WorkerExit {
        self.exit.clone()
    }

    /// `Err` only if the worker thread is gone (panicked or otherwise
    /// exited) — this is always a 500, never a caller mistake, so callers
    /// convert directly to [`WorkerError::Internal`].
    fn send(&self, cmd: WorkerCommand) -> Result<(), WorkerError> {
        self.queue_depth.fetch_add(1, Ordering::SeqCst);
        self.tx.send(cmd).map_err(|_| {
            // The send failed, so the worker will never dequeue this one —
            // undo the optimistic increment above rather than leaking a
            // phantom entry into the gauge.
            self.queue_depth.fetch_sub(1, Ordering::SeqCst);
            WorkerError::Internal("the inference worker thread is gone".to_string())
        })
    }

    async fn recv<T>(rx: oneshot::Receiver<T>) -> Result<T, WorkerError> {
        rx.await
            .map_err(|_| WorkerError::Internal("the inference worker dropped the reply channel without answering".to_string()))
    }

    pub async fn list_models(&self) -> Result<Vec<ModelInfo>, WorkerError> {
        let span = tracing::info_span!("worker_call", operation = "list_models", queue_wait_ms = tracing::field::Empty, inference_ms = tracing::field::Empty);
        let (tx, rx) = oneshot::channel();
        self.send(WorkerCommand::ListModels { reply: tx, meta: Enqueued { queued_at: Instant::now(), span } })?;
        Self::recv(rx).await
    }

    pub async fn embed(&self, inputs: Vec<String>, kind: EmbedKind) -> Result<EmbedOutcome, WorkerError> {
        let span = tracing::info_span!("worker_call", operation = "embed", queue_wait_ms = tracing::field::Empty, inference_ms = tracing::field::Empty);
        let (tx, rx) = oneshot::channel();
        self.send(WorkerCommand::Embed { inputs, kind, reply: tx, meta: Enqueued { queued_at: Instant::now(), span } })?;
        Self::recv(rx).await?
    }

    pub async fn route(&self, input: String, routes: Vec<String>) -> Result<RouteResponse, WorkerError> {
        let span = tracing::info_span!("worker_call", operation = "route", queue_wait_ms = tracing::field::Empty, inference_ms = tracing::field::Empty);
        let (tx, rx) = oneshot::channel();
        self.send(WorkerCommand::Route { input, routes, reply: tx, meta: Enqueued { queued_at: Instant::now(), span } })?;
        Self::recv(rx).await?
    }

    /// Hash `inputs` for the `input_hash` span field, iff `--log-input-hash`
    /// is enabled — computed here (request-side, before the command is even
    /// built) rather than on the worker thread, since it needs the caller's
    /// text and NOTHING downstream of this point is allowed to see that
    /// text in any form that could be logged (see [`Self::spans`]/
    /// [`Self::spans_credentials`]). Joins the batch with a control byte no
    /// legal input text can itself contain, so `["ab", "c"]` and `["a",
    /// "bc"]` hash differently — correctness the caller may care about even
    /// though this value's only sanctioned use is human correlation, never
    /// a lookup key.
    fn input_hash(&self, inputs: &[String]) -> Option<String> {
        self.log_input_hash.then(|| {
            let joined = inputs.join("\u{1}");
            crate::hash::sha256_hex_bytes(joined.as_bytes())
        })
    }

    /// `POST /v1/spans` — see [`InferenceEngine::spans`]. The
    /// `worker_call` span here NEVER carries the input text, any span
    /// offset, or any matched substring — see the module docs'
    /// "Observability" section and `crate::types::SpanResult`'s doc
    /// comment for why that's a hard rule on this endpoint specifically:
    /// request text contains live credentials by definition. `input_hash`
    /// is the ONE opt-in exception, gated on `--log-input-hash`
    /// (default off) and attached as a trace/log field only — never a
    /// metric label.
    pub async fn spans(&self, inputs: Vec<String>, model: Option<String>) -> Result<SpansOutcome, WorkerError> {
        let span = tracing::info_span!(
            "worker_call",
            operation = "spans",
            queue_wait_ms = tracing::field::Empty,
            inference_ms = tracing::field::Empty,
            input_hash = tracing::field::Empty,
        );
        if let Some(hash) = self.input_hash(&inputs) {
            span.record("input_hash", hash.as_str());
        }
        let (tx, rx) = oneshot::channel();
        self.send(WorkerCommand::Spans { inputs, model, reply: tx, meta: Enqueued { queued_at: Instant::now(), span } })?;
        Self::recv(rx).await?
    }

    /// `POST /v1/spans/credentials` — as [`Self::spans`], with the same
    /// no-text/no-offsets telemetry rule.
    pub async fn spans_credentials(&self, inputs: Vec<String>, model: Option<String>) -> Result<SpansOutcome, WorkerError> {
        let span = tracing::info_span!(
            "worker_call",
            operation = "spans_credentials",
            queue_wait_ms = tracing::field::Empty,
            inference_ms = tracing::field::Empty,
            input_hash = tracing::field::Empty,
        );
        if let Some(hash) = self.input_hash(&inputs) {
            span.record("input_hash", hash.as_str());
        }
        let (tx, rx) = oneshot::channel();
        self.send(WorkerCommand::SpansCredentials { inputs, model, reply: tx, meta: Enqueued { queued_at: Instant::now(), span } })?;
        Self::recv(rx).await?
    }
}

/// Pure decision logic for the crash monitor spawned by
/// [`WorkerHandle::spawn_crash_on_panic`], pulled out specifically so it's
/// unit-testable without ever calling `std::process::exit` from inside
/// `cargo test` (which would kill the test runner, not just fail one
/// test). `true` (a crash) iff the worker thread's closure unwound via
/// panic rather than returning normally.
fn worker_thread_outcome_is_a_crash(outcome: &std::thread::Result<()>) -> bool {
    outcome.is_err()
}

/// Fires once the worker thread has finished cleanly: its command loop
/// ended (every [`WorkerHandle`] clone was dropped) AND it has dropped the
/// engine. `main.rs` waits on this before `exit(0)`. A GPU engine's drop
/// frees device memory through the driver, and on ROCm a free issued after
/// `exit` has begun libamdhip64's atexit teardown segfaults (2026-09-13
/// core dump; `tests/shutdown_exit.rs`). Never fires for a worker that
/// panicked; that exit belongs to the crash monitor.
#[derive(Clone, Default)]
pub struct WorkerExit(Arc<(Mutex<bool>, Condvar)>);

impl WorkerExit {
    pub(crate) fn mark(&self) {
        let (finished, cv) = &*self.0;
        *finished.lock().expect("worker exit lock poisoned") = true;
        cv.notify_all();
    }

    /// Block until the worker has dropped its engine or `timeout` passes;
    /// `true` iff it finished.
    pub fn wait_timeout(&self, timeout: Duration) -> bool {
        let (finished, cv) = &*self.0;
        let guard = finished.lock().expect("worker exit lock poisoned");
        let (guard, _) = cv
            .wait_timeout_while(guard, timeout, |finished| !*finished)
            .expect("worker exit lock poisoned");
        *guard
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_panicked_worker_thread_is_a_crash() {
        let outcome: std::thread::Result<()> = Err(Box::new("simulated panic payload"));
        assert!(worker_thread_outcome_is_a_crash(&outcome));
    }

    #[test]
    fn a_cleanly_returned_worker_thread_is_not_a_crash() {
        let outcome: std::thread::Result<()> = Ok(());
        assert!(!worker_thread_outcome_is_a_crash(&outcome));
    }

    fn slow_drop_engine(marker: &std::path::Path) -> crate::engine_stub::StubEngine {
        crate::engine_stub::StubEngine {
            drop_probe: Some(Arc::new(crate::engine_stub::DropProbe {
                delay: Duration::from_millis(200),
                marker: marker.to_path_buf(),
            })),
            ..crate::engine_stub::StubEngine::fully_configured()
        }
    }

    #[test]
    fn exit_signal_fires_only_after_the_engine_has_dropped() {
        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join("dropped");
        let handle = WorkerHandle::spawn(slow_drop_engine(&marker));
        let exit = handle.exit_signal();
        drop(handle);
        assert!(exit.wait_timeout(Duration::from_secs(5)), "worker never reported a clean finish");
        assert!(marker.exists(), "exit signal fired before the engine finished dropping");
    }

    #[test]
    fn exit_signal_waits_while_any_handle_clone_is_alive() {
        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join("dropped");
        let handle = WorkerHandle::spawn(slow_drop_engine(&marker));
        let exit = handle.exit_signal();
        let clone = handle.clone();
        drop(handle);
        assert!(!exit.wait_timeout(Duration::from_millis(400)), "a live clone must keep the worker, and its engine, alive");
        assert!(!marker.exists());
        drop(clone);
        assert!(exit.wait_timeout(Duration::from_secs(5)), "dropping the last clone must finish the worker");
    }
}
