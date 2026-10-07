//! The engine-facing half of `/council/v1/contexts`: what the route layer
//! asks the worker to build, hold and inspect, in the engine's own terms.
//!
//! The route layer ([`crate::council_api`]) owns everything the client sees
//! (UUIDs, `snap` flags, wire snapshot ids, the records). This module is the
//! seam: rendered segments in, content ids out. Nothing here names a client
//! id, and nothing in it is a wire shape.
//!
//! **A boundary is a segment index** after which a snapshot is held: the
//! system head (segment 0), each turn the client marked `snap`, and the last
//! segment (the head). The engine forwards every segment in
//! [`crate::adjudicator::CHUNK`]-sized chunks from its own first token, so a
//! checkpoint's state is a function of its segments alone: resuming from any
//! held boundary gives the same bits as building from nothing. That is what
//! lets the contract's "same content, same snapshots, however it arrived"
//! hold here.
use serde::Serialize;

/// Build (or find) one context's snapshots.
#[derive(Clone, Debug)]
pub struct BuildRequest {
    /// The system head, then one segment per turn, each rendered as the
    /// template writes it. Never empty.
    pub segments: Vec<String>,
    /// Segment indexes to hold a snapshot after: ascending, distinct, and
    /// ending at the last segment (the head).
    pub hold_after: Vec<usize>,
    /// `Some` sets the head's pin; `None` leaves it as it was.
    pub pin: Option<bool>,
    /// Report what the build would keep and feed; build and hold nothing.
    pub dry_run: bool,
    pub timeout_ms: u64,
}

impl BuildRequest {
    pub fn validate(&self) -> Result<(), String> {
        let last = self.segments.len().checked_sub(1).ok_or("a context needs at least its system head")?;
        if self.segments.iter().any(String::is_empty) {
            return Err("a segment is empty".into());
        }
        if self.hold_after.last() != Some(&last) {
            return Err("the last segment is always a boundary: it is the head".into());
        }
        if self.hold_after.windows(2).any(|w| w[0] >= w[1]) {
            return Err("boundaries are ascending and distinct".into());
        }
        if self.timeout_ms == 0 || self.timeout_ms > 600_000 {
            return Err("timeout_ms must be 1..=600000".into());
        }
        Ok(())
    }
}

/// One boundary of a build, as the engine holds (or would hold) it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct BuiltSnapshot {
    /// The engine's content id of the ids through this boundary. Internal:
    /// the route layer derives the client-facing snapshot id from it.
    pub engine_id: String,
    pub after_segment: usize,
    pub tokens: usize,
    /// Held now. After a build every boundary the request named is held; in
    /// a dry run, or if the engine evicted one, this says which are.
    pub held: bool,
    pub bytes: usize,
    pub pinned: bool,
}

/// What a build did.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BuildOutcome {
    /// One per `hold_after`, in order; the last is the head.
    pub snapshots: Vec<BuiltSnapshot>,
    pub tokens: usize,
    /// Tokens reused from the largest boundary already held.
    pub kept: usize,
    /// Tokens this build ran, or would run under `dry_run`.
    pub fed: usize,
    pub prefill_ms: f64,
}

impl BuildOutcome {
    pub fn head(&self) -> &BuiltSnapshot {
        self.snapshots.last().expect("a build always has its head")
    }
}

/// What the engine holds under one id.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct HeldInfo {
    pub tokens: usize,
    pub bytes: usize,
    pub pinned: bool,
}
