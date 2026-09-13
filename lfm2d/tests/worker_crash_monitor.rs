//! Proves crash-only-on-panic through the REAL compiled `lfm2d` binary, not
//! just the in-process mechanism (`lfm2d::worker::worker_thread_outcome_is_a_crash`
//! is separately unit-tested in `src/worker.rs` — this file is the other
//! half: does `main.rs` actually WIRE that mechanism up in production).
//!
//! Today, a worker-thread panic leaves the HTTP server answering 500s
//! forever while `/healthz` still says 200 — kubelet never restarts an
//! alive-but-useless pod. The fix is crash-only: the worker's panic must
//! bring the whole PROCESS down nonzero so the supervisor restarts it.
//!
//! Exercising this against a real checkpoint isn't cheap (a model load
//! takes hundreds of ms and needs weights on disk), so `main.rs` has a
//! narrow, explicitly-named test-only branch (`LFM2D_TEST_CRASH_ON_WORKER_PANIC`)
//! that swaps in a panic-on-every-call `StubEngine` instead of loading real
//! models, but still goes through the PRODUCTION `WorkerHandle::spawn_crash_on_panic`
//! path — see `src/main.rs`'s `run_crash_monitor_test_harness`. This test
//! spawns that real binary as a child process, triggers the panic over a
//! real HTTP connection, and asserts the child process itself exits
//! nonzero — the actual observable behavior a k8s supervisor depends on.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Poll until `addr` accepts a TCP connection or `timeout` elapses.
fn wait_for_listening(addr: &str, timeout: Duration) {
    let start = Instant::now();
    loop {
        if TcpStream::connect(addr).is_ok() {
            return;
        }
        if start.elapsed() > timeout {
            panic!("lfm2d test harness never started listening on {addr} within {timeout:?}");
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// Send one raw HTTP POST to `/v1/classify` and discard the response —
/// the stub panics on every call, so all this needs to do is get the
/// request onto the wire; the response (if any) doesn't matter here.
fn fire_one_request(addr: &str) {
    let body = br#"{"inputs": ["trigger the panic"]}"#;
    let request = format!(
        "POST /v1/classify HTTP/1.1\r\n\
         Host: localhost\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\r\n",
        body.len()
    );
    if let Ok(mut stream) = TcpStream::connect(addr) {
        let _ = stream.write_all(request.as_bytes());
        let _ = stream.write_all(body);
        let mut buf = [0u8; 256];
        // Best-effort read: the connection may reset mid-response if the
        // worker thread dies before finishing — that's expected, not a
        // test failure.
        let _ = stream.read(&mut buf);
    }
}

#[test]
fn a_worker_panic_exits_the_real_process_nonzero() {
    let probe = std::net::TcpListener::bind("127.0.0.1:0").expect("find a free port");
    let addr = format!("127.0.0.1:{}", probe.local_addr().unwrap().port());
    drop(probe);

    let mut child = Command::new(env!("CARGO_BIN_EXE_lfm2d"))
        .env("LFM2D_TEST_CRASH_ON_WORKER_PANIC", "1")
        .env("LFM2D_TEST_BIND_ADDR", &addr)
        // the sibling harness in tests/shutdown_exit.rs; never let it leak in
        .env_remove("LFM2D_TEST_SHUTDOWN_DROP_MARKER")
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn the lfm2d binary");

    wait_for_listening(&addr, Duration::from_secs(10));

    // Sanity check: the process is alive and serving BEFORE the panic —
    // otherwise a nonzero exit later could just mean "never started."
    assert!(child.try_wait().expect("try_wait").is_none(), "process must still be running before the panic");

    fire_one_request(&addr);

    let status = wait_for_exit(&mut child, Duration::from_secs(10));
    assert!(!status.success(), "a worker-thread panic must exit the process nonzero, got {status}");
}

fn wait_for_exit(child: &mut std::process::Child, timeout: Duration) -> std::process::ExitStatus {
    let start = Instant::now();
    loop {
        if let Some(status) = child.try_wait().expect("try_wait") {
            return status;
        }
        if start.elapsed() > timeout {
            let _ = child.kill();
            panic!(
                "lfm2d process did not exit within {timeout:?} of the worker panic — \
                 crash-only behavior is not wired up (the process must exit nonzero, \
                 not keep serving 500s while /healthz still says 200)"
            );
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}
