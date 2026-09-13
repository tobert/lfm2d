//! A graceful SIGTERM must exit 0 only AFTER the worker has dropped its
//! engine, proven through the REAL compiled `lfm2d` binary.
//!
//! On 2026-09-13 a ROCm build logged "shutdown complete, exiting 0" and then
//! died of SIGSEGV (1 of 11 paired shutdowns; the 2026-09-12 bench saw the
//! same death reported as exit 1). The core dump put the fault on the
//! `lfm2d-worker` thread: `serve()` returned, the router dropped the last
//! `WorkerHandle`, the worker loop ended and dropped `RealEngine`, and its
//! last tensor freed the ROCm allocator (`release_all` -> `hipFree`) while
//! `main`'s `exit(0)` was already running libamdhip64's atexit teardown. A
//! GPU pod would report every graceful stop as a crash.
//!
//! The race needs a GPU to fault, but the ORDERING that causes it does not:
//! `main.rs`'s `LFM2D_TEST_SHUTDOWN_DROP_MARKER` harness runs the production
//! shutdown tail with a stub engine whose drop sleeps and then writes a
//! marker (`engine_stub::DropProbe`). If the process exits before that drop
//! finishes, the marker is missing — the same window a GPU engine faults in.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

fn wait_for_listening(addr: &str, timeout: Duration) {
    let start = Instant::now();
    while TcpStream::connect(addr).is_err() {
        if start.elapsed() > timeout {
            panic!("lfm2d test harness never started listening on {addr} within {timeout:?}");
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// One real request, so the worker has served before shutdown as it would
/// in production; the response must be a 200.
fn classify_once(addr: &str) {
    let body = br#"{"inputs": ["kubectl get pods"]}"#;
    let request = format!(
        "POST /v1/classify HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let mut stream = TcpStream::connect(addr).expect("connect to the harness");
    stream.write_all(request.as_bytes()).unwrap();
    stream.write_all(body).unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).unwrap();
    assert!(response.starts_with("HTTP/1.1 200"), "classify before shutdown failed: {response}");
}

fn wait_for_exit(child: &mut std::process::Child, timeout: Duration) -> std::process::ExitStatus {
    let start = Instant::now();
    loop {
        if let Some(status) = child.try_wait().expect("try_wait") {
            return status;
        }
        if start.elapsed() > timeout {
            let _ = child.kill();
            panic!("lfm2d did not exit within {timeout:?} of SIGTERM");
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn sigterm_exits_zero_only_after_the_worker_drops_its_engine() {
    let dir = tempfile::tempdir().expect("tempdir");
    let marker = dir.path().join("engine-dropped");
    let probe = std::net::TcpListener::bind("127.0.0.1:0").expect("find a free port");
    let addr = format!("127.0.0.1:{}", probe.local_addr().unwrap().port());
    drop(probe);

    let mut child = Command::new(env!("CARGO_BIN_EXE_lfm2d"))
        .env("LFM2D_TEST_SHUTDOWN_DROP_MARKER", &marker)
        .env("LFM2D_TEST_BIND_ADDR", &addr)
        // main.rs checks the crash harness first; an inherited one would shadow this test
        .env_remove("LFM2D_TEST_CRASH_ON_WORKER_PANIC")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn the lfm2d binary");

    wait_for_listening(&addr, Duration::from_secs(10));
    classify_once(&addr);
    assert!(!marker.exists(), "the engine must still be alive while serving");

    let kill = Command::new("kill").args(["-TERM", &child.id().to_string()]).status().expect("run kill");
    assert!(kill.success(), "sending SIGTERM failed");
    let status = wait_for_exit(&mut child, Duration::from_secs(20));

    assert!(
        marker.exists(),
        "the process exited before the worker finished dropping its engine — on ROCm that drop \
         frees device memory and faults once exit's teardown has begun (exit status {status})"
    );
    assert_eq!(status.code(), Some(0), "a graceful SIGTERM must exit 0, got {status}");

    // `std::process::exit` never unwinds `main`, so `TelemetryGuard`'s drop
    // (its final OTLP flush) silently never ran. The exit tail must flush
    // explicitly, and after the engine drop so the worker's last spans count.
    let flushed = std::fs::read_to_string(dir.path().join("engine-dropped.flushed"))
        .expect("the exit tail never flushed telemetry");
    assert_eq!(flushed, "after-engine-drop", "telemetry flushed before the worker finished");
}
