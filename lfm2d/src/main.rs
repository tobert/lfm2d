//! `lfm2d` — the LFM2.5 encoder sidecar/daemon binary.
//!
//! Parses config, sizes rayon's global thread pool (`--threads`), probes
//! the container runtime, loads every configured checkpoint SYNCHRONOUSLY
//! (no lazy loading — a bad config or a failed load exits the process
//! before any socket is ever bound), wires up OTLP telemetry now that each
//! model's weight hash is known, spawns the one inference worker thread
//! (crash-only: a worker-thread panic exits the process so the k8s
//! supervisor restarts it — see [`lfm2d::worker::WorkerHandle::spawn_crash_on_panic`]),
//! installs SIGTERM/SIGINT handling (flip `/readyz` to 503, drain in-flight
//! requests, exit 0), and serves the API router on whichever of a Unix
//! domain socket / TCP address was configured. See `lfm2d/README.md` for
//! the endpoint list and k8s/container deployment guidance, and
//! `src/lib.rs`'s module docs for the architecture this implements.
//!
//! ```text
//! lfm2d --classifier-dir .models/kube_ordinal_v6 \
//!       --router-dir .models/LFM2.5-Encoder-350M-Prompt-Router \
//!       --cascade-route shell --cascade-route k8s \
//!       --socket-path /run/lfm2d/lfm2d.sock \
//!       --bind-addr 127.0.0.1:8088
//! ```

use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;

use clap::Parser;

use lfm2d::config::Cli;
use lfm2d::engine_real::RealEngine;
use lfm2d::server::{begin_shutdown, build_router, serve, AppState};
use lfm2d::shutdown::{self, ShutdownHandle};
use lfm2d::worker::{InferenceEngine, WorkerHandle};

/// Test-only escape hatch for `tests/worker_crash_monitor.rs`: that test
/// needs to exercise the crash-only behavior of the REAL compiled binary
/// (`WorkerHandle::spawn_crash_on_panic`, installed below in the normal
/// path) without requiring a model checkpoint on disk to reach that code.
/// Gated on an env var that is never set outside that one test — see the
/// test file for the full rationale. `main`'s normal path is completely
/// unaffected when this var is unset, which is every real deployment.
const TEST_CRASH_ENV: &str = "LFM2D_TEST_CRASH_ON_WORKER_PANIC";

#[tokio::main]
async fn main() {
    if std::env::var_os(TEST_CRASH_ENV).is_some() {
        return run_crash_monitor_test_harness().await;
    }

    let cli = Cli::parse();
    if let Err(msg) = cli.validate() {
        eprintln!("lfm2d: {msg}");
        std::process::exit(2);
    }

    // --threads sizes rayon's GLOBAL pool, which candle's matmul (the
    // `gemm` crate) runs on transitively — this MUST happen before any
    // model load, or candle's own lazy default wins the race and this flag
    // silently does nothing. `rayon::ThreadPoolBuilder::build_global` can
    // only be called once per process, so there is no second chance either.
    let available_parallelism = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let threads = cli.threads.unwrap_or(available_parallelism);
    if let Err(e) = rayon::ThreadPoolBuilder::new().num_threads(threads).build_global() {
        eprintln!("lfm2d: failed to size the rayon global thread pool to {threads} threads: {e}");
        std::process::exit(1);
    }

    let container_probe = lfm2d::probe::detect_real();

    // Pre-telemetry bootstrap: the ONLY two lines in this file that stay
    // bare eprintln! — everything from telemetry::init() onward goes
    // through tracing, but the OTLP resource attributes need each loaded
    // model's weight hash (see telemetry.rs), which isn't known until
    // AFTER this load completes. A load failure here means the process
    // exits before any subscriber ever existed, so there is nothing to
    // route through tracing anyway.
    eprintln!("lfm2d: loading configured checkpoints (no lazy loading — this may take a few seconds)...");
    let load_start = std::time::Instant::now();
    let engine = match RealEngine::load(&cli) {
        Ok(engine) => engine,
        Err(msg) => {
            eprintln!("lfm2d: failed to load models: {msg}");
            std::process::exit(1);
        }
    };
    let load_elapsed = load_start.elapsed();

    let models = engine.list_models();
    let service_name = std::env::var("OTEL_SERVICE_NAME").unwrap_or_else(|_| "lfm2d".to_string());
    let _telemetry = lfm2d::telemetry::init(&service_name, &models);

    tracing::info!(
        embedder_dir = ?cli.embedder_dir,
        classifier_dir = ?cli.classifier_dir,
        candidate_classifier_dir = ?cli.candidate_classifier_dir,
        router_dir = ?cli.router_dir,
        token_classifier_dir = ?cli.token_classifier_dir,
        cascade_routes = ?cli.cascade_routes,
        cascade_severe_labels = ?cli.cascade_severe_labels,
        log_input_hash = cli.log_input_hash,
        socket_path = ?cli.socket_path,
        bind_addr = ?cli.bind_addr,
        threads,
        load_elapsed_ms = load_elapsed.as_secs_f64() * 1000.0,
        "lfm2d: effective configuration"
    );
    for m in &models {
        tracing::info!(
            id = %m.id,
            kind = ?m.kind,
            weight_hash = %m.weight_hash,
            hidden_size = m.hidden_size,
            "lfm2d: loaded model"
        );
    }

    // The severe-label ORDER is the ordinal severity scale, and reversing it
    // silently inverts every /v1/cascade ranking — both names are valid, so
    // `RealEngine::load`'s resolve step passes and the endpoint just starts
    // naming the least severe clause as the winner. Echoing the raw flag (as
    // the config line above does) cannot expose that: a reversed list reads
    // like an ordinary list. Rendering the RESOLVED ranking can, because
    // `1=data-critical 2=situation-normal` is wrong on sight.
    //
    // Deliberately its own line rather than a field on the config line: this
    // is the one startup line an operator is asked to re-read after touching
    // --cascade-severe-label, and burying it among twelve other fields is how
    // it stops being read. Only emitted when a cascade is actually configured,
    // matching `RealEngine::load` — with no --cascade-route there is no
    // ranking to get wrong.
    //
    // Unreachable-by-construction: load() already resolved these same labels
    // against these same classifier labels and exited non-zero on failure, so
    // an Err here would mean the two disagreed. Log it loudly rather than
    // unwrap — a startup diagnostic must never be the thing that kills the
    // process it is diagnosing.
    if !cli.cascade_routes.is_empty() {
        let classifier_labels = models
            .iter()
            .find(|m| m.kind == lfm2d::types::ModelKind::Classifier)
            .and_then(|m| m.labels.clone())
            .unwrap_or_default();
        match lfm2d::config::resolved_severity_ranking(&classifier_labels, &cli.cascade_severe_labels)
        {
            Ok(ranked) => tracing::info!(
                ranking = %lfm2d::config::render_severity_ranking(&ranked),
                classifier_labels = ?classifier_labels,
                "lfm2d: cascade severity ranking (ascending, least severe first) — \
                 re-read this after any --cascade-severe-label change; a reversed \
                 flag inverts /v1/cascade's winner and raises no error"
            ),
            Err(e) => tracing::error!(
                error = %e,
                "lfm2d: could not render the cascade severity ranking — startup \
                 validation passed, so this disagreeing is a bug, not a misconfiguration"
            ),
        }
    }
    tracing::info!(
        available_parallelism,
        configured_threads = threads,
        container_runtime = %container_probe.runtime,
        cgroup_cpu_max = ?container_probe.cgroup_cpu_max,
        "lfm2d: startup observability"
    );

    let worker = WorkerHandle::spawn_crash_on_panic(engine).with_log_input_hash(cli.log_input_hash);
    let _queue_depth_gauge = lfm2d::telemetry::register_queue_depth_gauge(worker.queue_depth_handle());

    // Loading (above) is synchronous and already complete by the time we
    // get here, so this starts `true` — see `AppState::ready`'s doc comment
    // for why the flag exists as a distinct concept from "the worker is
    // running" regardless. Kept here (not just inside AppState) so the
    // SIGTERM handler installed below can flip it without reaching back
    // into the router/state that `build_router` has already consumed.
    let ready = Arc::new(AtomicBool::new(true));
    let router = build_router(AppState { worker, ready: ready.clone() });

    let (shutdown_handle, shutdown_signal) = shutdown::channel();
    install_signal_handlers(ready, shutdown_handle);

    if let Some(path) = &cli.socket_path {
        tracing::info!(path = %path.display(), "lfm2d: will serve on unix socket");
    }
    if let Some(addr) = &cli.bind_addr {
        tracing::info!(addr = %addr, "lfm2d: will serve on tcp");
    }

    let result = serve(
        router,
        cli.socket_path.clone(),
        cli.bind_addr.clone(),
        shutdown_signal,
        shutdown::DEFAULT_DRAIN_TIMEOUT,
    )
    .await;

    match result {
        Ok(()) => {
            tracing::info!("lfm2d: shutdown complete, exiting 0");
            std::process::exit(0);
        }
        Err(e) => {
            tracing::error!(error = %e, "lfm2d: server error");
            std::process::exit(1);
        }
    }
}

/// Install SIGTERM + SIGINT handling: on whichever arrives first, run the
/// exact sequence [`begin_shutdown`] documents — flip `/readyz` to 503,
/// then trigger `serve()`'s graceful shutdown on every listener.
/// `tests/graceful_shutdown.rs` drives `begin_shutdown` directly
/// (programmatically, not via a real signal) to prove the drain sequence
/// itself; this function is the thin real-signal wiring around it. Rust
/// runs as PID 1 in the container (see `lfm2d/Containerfile`), so this
/// handler must be installed explicitly — there is no init process to
/// translate the signal into anything for us, and an unhandled SIGTERM to
/// PID 1 in most container runtimes does nothing at all.
fn install_signal_handlers(ready: Arc<AtomicBool>, shutdown_handle: ShutdownHandle) {
    tokio::spawn(async move {
        let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install a SIGTERM handler");
        tokio::select! {
            _ = sigterm.recv() => tracing::info!("lfm2d: received SIGTERM, draining in-flight requests..."),
            _ = tokio::signal::ctrl_c() => tracing::info!("lfm2d: received SIGINT, draining in-flight requests..."),
        }
        begin_shutdown(&ready, &shutdown_handle);
    });
}

/// Test-only harness for `tests/worker_crash_monitor.rs` — see
/// [`TEST_CRASH_ENV`]. Binds `LFM2D_TEST_BIND_ADDR` (the test picks a free
/// port itself and passes it explicitly, the same way `tests/serve_transports.rs`
/// does) with a [`lfm2d::engine_stub::StubEngine`] configured to panic on
/// every call, spawned via the PRODUCTION `WorkerHandle::spawn_crash_on_panic`
/// path — this exercises the real mechanism `main`'s normal path installs,
/// just without needing an actual checkpoint on disk to get there.
async fn run_crash_monitor_test_harness() {
    let bind_addr = std::env::var("LFM2D_TEST_BIND_ADDR")
        .expect("LFM2D_TEST_BIND_ADDR must be set alongside LFM2D_TEST_CRASH_ON_WORKER_PANIC");
    let engine = lfm2d::engine_stub::StubEngine {
        panic_with: Some("simulated worker death for crash-monitor test".to_string()),
        ..lfm2d::engine_stub::StubEngine::fully_configured()
    };
    let worker = WorkerHandle::spawn_crash_on_panic(engine);
    let ready = Arc::new(AtomicBool::new(true));
    let router = build_router(AppState { worker, ready });
    let (_shutdown_handle, shutdown_signal) = shutdown::channel();
    eprintln!("lfm2d-test-harness: listening on {bind_addr}");
    let _ = serve(router, None, Some(bind_addr), shutdown_signal, Duration::from_secs(60)).await;
}
