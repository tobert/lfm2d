//! Graceful-shutdown signaling, shared between `main.rs` (SIGTERM/SIGINT →
//! drain → exit 0) and [`crate::server::serve`] (stop accepting new
//! connections on every configured listener, let in-flight requests
//! finish). Split into its own small module so [`ShutdownSignal`] can be
//! constructed and triggered programmatically from a test without going
//! through an actual OS signal — see `tests/graceful_shutdown.rs`.
//!
//! Built on [`tokio::sync::watch`] rather than a one-shot: `serve()` needs
//! to hand the SAME "has shutdown been requested" future to every listener
//! it's serving concurrently (TCP and/or a Unix socket), and `watch`'s
//! receiver is cheaply `Clone`, one per listener.

use std::time::Duration;

use tokio::sync::watch;

/// A cheap-to-clone future source: `.wait()` resolves once
/// [`ShutdownHandle::trigger`] has been called. If every [`ShutdownHandle`]
/// clone is dropped without ever calling `trigger` (a bug, since production
/// code keeps one alive for the process lifetime), `.wait()` still resolves
/// rather than hanging forever — a lost shutdown signal should fail toward
/// "stop serving," never toward "hang indefinitely."
#[derive(Clone)]
pub struct ShutdownSignal(watch::Receiver<bool>);

impl ShutdownSignal {
    /// Resolves once shutdown has been requested (or can never be, because
    /// every handle was dropped — see the struct docs).
    pub async fn wait(mut self) {
        loop {
            if *self.0.borrow() {
                return;
            }
            if self.0.changed().await.is_err() {
                // Every ShutdownHandle dropped without triggering: treat
                // as "shutdown requested" rather than waiting forever.
                return;
            }
        }
    }
}

/// The trigger side. `Clone`s share the same underlying signal — cloning is
/// how both the SIGTERM handler task and the SIGINT handler task in
/// `main.rs` can race to be the one that fires it.
#[derive(Clone)]
pub struct ShutdownHandle(watch::Sender<bool>);

impl ShutdownHandle {
    /// Request shutdown. Idempotent — calling it more than once (e.g. a
    /// second SIGTERM arriving mid-drain) is a harmless no-op after the
    /// first call.
    pub fn trigger(&self) {
        // `send` only errors if every receiver was dropped, which just
        // means nothing is listening any more — nothing to escalate to.
        let _ = self.0.send(true);
    }
}

/// A fresh, untriggered shutdown signal and its handle.
pub fn channel() -> (ShutdownHandle, ShutdownSignal) {
    let (tx, rx) = watch::channel(false);
    (ShutdownHandle(tx), ShutdownSignal(rx))
}

/// Hard cap on how long [`crate::server::serve`] waits for in-flight
/// requests to finish draining after shutdown is requested, before giving
/// up and returning anyway. A forward pass through these models is
/// sub-second (see `lfm2d/README.md`), so 10s is generous slack, not a
/// tuned budget — the point is bounding an operator's worst case, not
/// modeling real request latency.
pub const DEFAULT_DRAIN_TIMEOUT: Duration = Duration::from_secs(10);

/// How long a graceful exit waits, after [`crate::server::serve`] returns,
/// for the inference worker thread to finish dropping its engine (see
/// `main.rs`'s `exit_after_serve` and [`crate::worker::WorkerExit`]). The
/// drop itself takes milliseconds; the bound only matters when a request
/// outlived [`DEFAULT_DRAIN_TIMEOUT`] and still holds the worker. Drain plus
/// this stays under Kubernetes' default 30s termination grace period.
pub const WORKER_EXIT_TIMEOUT: Duration = Duration::from_secs(5);

/// How long a graceful exit waits for the final telemetry flush (dropping
/// `telemetry::TelemetryGuard`, which shuts each OTLP provider down) before
/// exiting without it. A dead collector must not eat the grace period:
/// drain + worker wait + this = 20s, still under Kubernetes' default 30s.
pub const TELEMETRY_FLUSH_TIMEOUT: Duration = Duration::from_secs(5);

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn wait_resolves_once_triggered() {
        let (handle, signal) = channel();
        let waiter = tokio::spawn(signal.wait());
        // Give the spawned task a chance to start polling before we
        // trigger — not required for correctness (trigger-then-wait must
        // also work, exercised below) but proves the "already waiting"
        // path specifically.
        tokio::task::yield_now().await;
        handle.trigger();
        tokio::time::timeout(Duration::from_secs(1), waiter)
            .await
            .expect("wait() must resolve promptly after trigger()")
            .expect("task must not panic");
    }

    #[tokio::test]
    async fn wait_resolves_immediately_if_already_triggered() {
        let (handle, signal) = channel();
        handle.trigger();
        tokio::time::timeout(Duration::from_millis(200), signal.wait())
            .await
            .expect("wait() must not block once already triggered");
    }

    #[tokio::test]
    async fn wait_resolves_if_every_handle_is_dropped_without_triggering() {
        let (handle, signal) = channel();
        drop(handle);
        tokio::time::timeout(Duration::from_millis(200), signal.wait())
            .await
            .expect("a lost signal must fail toward shutdown, not hang forever");
    }

    #[tokio::test]
    async fn wait_does_not_resolve_before_trigger() {
        let (_handle, signal) = channel();
        let result = tokio::time::timeout(Duration::from_millis(100), signal.wait()).await;
        assert!(result.is_err(), "wait() resolved without ever being triggered");
    }

    #[tokio::test]
    async fn trigger_is_idempotent() {
        let (handle, signal) = channel();
        handle.trigger();
        handle.trigger();
        tokio::time::timeout(Duration::from_millis(200), signal.wait())
            .await
            .expect("double-trigger must not break wait()");
    }

    #[tokio::test]
    async fn clones_of_handle_share_one_signal() {
        let (handle, signal) = channel();
        let handle2 = handle.clone();
        handle2.trigger();
        tokio::time::timeout(Duration::from_millis(200), signal.wait())
            .await
            .expect("triggering a clone must be visible to the original signal");
    }
}
