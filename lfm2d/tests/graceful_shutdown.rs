//! Proves the SIGTERM/SIGINT drain sequence end to end, WITHOUT sending a
//! real OS signal — [`lfm2d::server::begin_shutdown`] is exactly what
//! `main.rs`'s signal handler calls, so driving it directly here exercises
//! the same code path a real SIGTERM would, just triggered programmatically
//! (see `lfm2d/src/main.rs` for the signal-handling wiring itself, which
//! this crate has no clean way to integration-test without sending a real
//! signal to the test process).
//!
//! Asserts the three things the task calls out explicitly:
//! 1. `/readyz` flips to 503 the moment shutdown is requested.
//! 2. An in-flight request (held open via [`StubEngine::delay`]) still
//!    completes with 200 — graceful shutdown drains, it doesn't cut off.
//! 3. [`serve`]'s future then resolves (the listener stopped accepting new
//!    connections and every in-flight request finished).

use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tower::ServiceExt;

use lfm2d::engine_stub::StubEngine;
use lfm2d::server::{begin_shutdown, build_router, serve, AppState};
use lfm2d::worker::WorkerHandle;

async fn raw_post_embed(mut stream: TcpStream) -> String {
    let body = br#"{"inputs": ["the quarterly report is attached"]}"#;
    let request = format!(
        "POST /embed HTTP/1.1\r\n\
         Host: localhost\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(request.as_bytes()).await.expect("write headers");
    stream.write_all(body).await.expect("write body");
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).await.expect("read response");
    String::from_utf8(buf).expect("response is valid utf8")
}

#[tokio::test]
async fn sigterm_sequence_drains_in_flight_requests_then_resolves() {
    // The stub sleeps 300ms inside the (blocking) worker thread before
    // answering — long enough to reliably trigger shutdown mid-request
    // without the request racing ahead and finishing first.
    let engine = StubEngine { delay: Some(Duration::from_millis(300)), ..StubEngine::fully_configured() };
    let worker = WorkerHandle::spawn(engine);
    let ready = Arc::new(AtomicBool::new(true));
    let router = build_router(AppState { worker, ready: ready.clone() });

    let (shutdown_handle, shutdown_signal) = lfm2d::shutdown::channel();

    let probe = std::net::TcpListener::bind("127.0.0.1:0").expect("find a free port");
    let port = probe.local_addr().unwrap().port();
    drop(probe);
    let bind_addr = format!("127.0.0.1:{port}");

    let serve_router = router.clone();
    let serve_handle = tokio::spawn(async move {
        serve(serve_router, None, Some(bind_addr.clone()), shutdown_signal, Duration::from_secs(5)).await
    });

    // Wait for the TCP listener to actually be up before firing the
    // in-flight request.
    let mut connected = None;
    for _ in 0..50 {
        match TcpStream::connect(("127.0.0.1", port)).await {
            Ok(s) => {
                connected = Some(s);
                break;
            }
            Err(_) => tokio::time::sleep(Duration::from_millis(10)).await,
        }
    }
    let stream = connected.expect("server never came up");

    // Fire the slow request in the background — it will be mid-`sleep`
    // inside the worker thread by the time we trigger shutdown below.
    let inflight = tokio::spawn(raw_post_embed(stream));
    tokio::time::sleep(Duration::from_millis(50)).await;

    // --- trigger the same sequence main.rs's SIGTERM handler runs ---
    begin_shutdown(&ready, &shutdown_handle);

    // 1. /readyz flips to 503 immediately — checked via `oneshot` against
    //    the SAME router/state `serve` is using (not through the listener,
    //    which is now refusing new connections by design), so this proves
    //    the flag itself flipped rather than anything about the listener.
    let resp = router
        .clone()
        .oneshot(axum::http::Request::builder().uri("/readyz").body(axum::body::Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), axum::http::StatusCode::SERVICE_UNAVAILABLE, "readyz must flip to 503 on shutdown");

    // 2. The in-flight request still completes with 200 — drained, not cut off.
    let response = tokio::time::timeout(Duration::from_secs(3), inflight)
        .await
        .expect("in-flight request must complete within the drain cap")
        .expect("request task must not panic");
    assert!(response.starts_with("HTTP/1.1 200"), "in-flight request must complete with 200, got: {response}");

    // 3. serve()'s future then resolves.
    let result = tokio::time::timeout(Duration::from_secs(3), serve_handle)
        .await
        .expect("serve() must resolve after the drain completes")
        .expect("serve task must not panic");
    result.expect("serve() must resolve Ok after a graceful shutdown");
}

#[tokio::test]
async fn drain_hard_cap_gives_up_on_a_request_that_outlives_it() {
    // A request slower than the drain cap: shutdown must still resolve
    // `serve()` within roughly `drain_timeout`, not hang waiting forever.
    let engine = StubEngine { delay: Some(Duration::from_secs(10)), ..StubEngine::fully_configured() };
    let worker = WorkerHandle::spawn(engine);
    let ready = Arc::new(AtomicBool::new(true));
    let router = build_router(AppState { worker, ready: ready.clone() });

    let (shutdown_handle, shutdown_signal) = lfm2d::shutdown::channel();

    let probe = std::net::TcpListener::bind("127.0.0.1:0").expect("find a free port");
    let port = probe.local_addr().unwrap().port();
    drop(probe);
    let bind_addr = format!("127.0.0.1:{port}");

    let drain_timeout = Duration::from_millis(300);
    let serve_handle = tokio::spawn(async move {
        serve(router, None, Some(bind_addr), shutdown_signal, drain_timeout).await
    });

    let mut connected = None;
    for _ in 0..50 {
        match TcpStream::connect(("127.0.0.1", port)).await {
            Ok(s) => {
                connected = Some(s);
                break;
            }
            Err(_) => tokio::time::sleep(Duration::from_millis(10)).await,
        }
    }
    let stream = connected.expect("server never came up");
    let _inflight = tokio::spawn(raw_post_embed(stream));
    tokio::time::sleep(Duration::from_millis(50)).await;

    begin_shutdown(&ready, &shutdown_handle);

    // Must resolve close to drain_timeout, well before the 10s request
    // would ever finish on its own.
    let result = tokio::time::timeout(Duration::from_secs(2), serve_handle)
        .await
        .expect("serve() must give up at the drain cap, not hang for the full 10s request")
        .expect("serve task must not panic");
    result.expect("serve() must still resolve Ok even after hitting the drain cap");
}
