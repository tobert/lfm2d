//! The engine's side of `/council/v1/contexts` on the real LFM2.5-8B-A1B
//! (`Generator::council_put`, `council_inspect`, `council_unpin`). What this
//! certifies, on one backend, that the fake-engine route tests cannot:
//!
//! - a build with held boundaries reaches the state a one-shot `context_create`
//!   reaches, bit for bit (every held boundary is probed, not only the head),
//!   and under the same id: the engine's content id is a function of the
//!   tokens;
//! - an update resumes from the largest *declared* boundary already held, and
//!   the state it reaches is the one-shot state (so the contract's "same
//!   content, same snapshots, however it arrived" holds);
//! - an unmarked turn that happens to be held is not a boundary;
//! - a dry run reports what a build would keep and feed, and builds nothing;
//! - a pin shows on lookup and `council_unpin` releases it.
//!
//! States are compared through `Adjudicator::chat_checkpoint_probe` (the
//! logits after a fixed probe, from a clone of the held state), never through
//! a read: a read of these contexts says nothing about the state under test.
//! The described-state cache is keyed by the prompt's ids, so reads of the
//! same content share an entry whichever state they ran from, and
//! `use_cache: false` re-prefills from token 0 and ignores the held state.
//! An earlier version of this test compared reads and passed against an
//! engine that resumed from a blank state (found by mutation).
mod support;
use lfm2d::adjudicator::{Adjudicator, Generator};
use lfm2d::council::{ContextBody, boundaries, render_segments};
use lfm2d::council_context::{BuildOutcome, BuildRequest};
use serde_json::{Value, json};

const SPEC: &str = "email-triage-v1";
const SYSTEM: &str = "You are a helpful assistant for Dana, who runs customer support for a small \
                      online kitchenware store. Help her work through her inbox. Be concise.";

/// The first `n` of three turns, marked `snap` as `snaps` says.
fn body(n: usize, snaps: [bool; 3]) -> ContextBody {
    let all = [
        ("user", "Morning! Does our enameled dutch oven work on induction?"),
        ("assistant", "Yes, it does: enameled cast iron is magnetic."),
        ("user", "Thanks. The next one is angrier."),
    ];
    let turns: Vec<Value> = all[..n]
        .iter()
        .zip(snaps)
        .map(|((role, content), snap)| json!({"role": role, "content": content, "snap": snap}))
        .collect();
    serde_json::from_value(json!({"system": SYSTEM, "turns": turns})).unwrap()
}

fn put(a: &mut Adjudicator, body: &ContextBody, pin: Option<bool>, dry_run: bool) -> BuildOutcome {
    let request = BuildRequest {
        segments: render_segments(body).unwrap(),
        hold_after: boundaries(body),
        pin,
        dry_run,
        timeout_ms: 120_000,
    };
    a.council_put(&request, &|| Ok(())).expect("a council build")
}

fn forget(a: &mut Adjudicator, outcome: &BuildOutcome) {
    for s in &outcome.snapshots {
        let _ = a.context_delete(&s.engine_id);
    }
}

/// Every held boundary of a build, as the probe sees it, by segment.
fn probes(a: &Adjudicator, outcome: &BuildOutcome) -> std::collections::BTreeMap<usize, Vec<f32>> {
    outcome
        .snapshots
        .iter()
        .filter(|s| s.held)
        .map(|s| (s.after_segment, probe(a, &s.engine_id)))
        .collect()
}

/// A build's states are the one-shot build's, boundary by boundary: a state
/// built by resuming must not differ at an intermediate boundary even where
/// the head comes out right.
fn assert_same_states(
    expected: &std::collections::BTreeMap<usize, Vec<f32>>,
    got: &std::collections::BTreeMap<usize, Vec<f32>>,
    what: &str,
) {
    assert!(!got.is_empty(), "{what}: no held boundary to compare");
    for (segment, probe) in got {
        assert_eq!(expected.get(segment), Some(probe), "{what}: the state after segment {segment}");
    }
}

fn held(a: &mut Adjudicator, outcome: &BuildOutcome) -> Vec<bool> {
    let ids: Vec<String> = outcome.snapshots.iter().map(|s| s.engine_id.clone()).collect();
    a.council_inspect(&ids).unwrap().iter().map(Option::is_some).collect()
}

/// The state held under `id`, as the probe sees it.
fn probe(a: &Adjudicator, id: &str) -> Vec<f32> {
    a.chat_checkpoint_probe(id).expect("held").expect("the probe ran")
}

#[test]
#[ignore = "loads and hashes the 6 GB LFM2.5-8B-A1B GGUF; minutes on a GPU host"]
fn a_council_build_reaches_the_one_shot_state_however_it_arrived() {
    let mut a = support::load_adjudicator(&support::adjudicator_cli(&[SPEC]));
    let tokenizer = a.tokenizer_clone();
    let all = body(3, [true, true, false]);

    // The reference: the one-shot build `/v1/contexts` makes of the same content.
    let messages: Vec<Value> = all
        .turns
        .iter()
        .map(|t| json!({"role": serde_json::to_value(t.role).unwrap(), "content": t.content}))
        .collect();
    let request: lfm2d::contexts_api::ContextRequest =
        serde_json::from_value(json!({"system": SYSTEM, "messages": messages})).unwrap();
    let one_shot = a.context_create(&request, &|| Ok(())).expect("the one-shot build");
    let reference = probe(&a, &one_shot.id);
    a.context_delete(&one_shot.id).unwrap();

    // The instrument must be able to see a different state: the probe of a
    // shorter context is not the reference. (Without this, an instrument that
    // ignored the held state would make every equality below vacuous.)
    let shorter = put(&mut a, &body(1, [true, false, false]), None, false);
    assert_ne!(probe(&a, &shorter.head().engine_id), reference, "the probe must see a different state");
    forget(&mut a, &shorter);

    // 1. Built with held boundaries from nothing: the same id, the same bits.
    let built = put(&mut a, &all, None, false);
    assert_eq!((built.kept, built.fed), (0, built.tokens), "nothing was held");
    assert_eq!(built.head().engine_id, one_shot.id, "the engine's id is a function of the tokens");
    assert_eq!(built.snapshots.len(), 4, "the system, two snap turns and the head");
    assert_eq!(held(&mut a, &built), [true; 4]);
    let one_shot_states = probes(&a, &built);
    assert_eq!(one_shot_states.len(), 4, "every boundary has a state to compare against");
    assert_eq!(probe(&a, &built.head().engine_id), reference, "bit for bit the one-shot state");
    let text = render_segments(&all).unwrap().concat();
    let ids = tokenizer.encode(text.as_str(), false).unwrap().get_ids().to_vec();
    assert_eq!(built.head().engine_id, lfm2d::chat_session::context_id(&ids));

    // 2. An update resumes from the largest declared boundary held, and gets there too.
    forget(&mut a, &built);
    let two = put(&mut a, &body(2, [true, true, false]), None, false);
    let third = put(&mut a, &all, None, false);
    assert_eq!(third.kept, two.tokens, "everything through the second turn");
    assert_eq!(third.fed, third.tokens - two.tokens, "only the new turn");
    assert_eq!(third.head().engine_id, one_shot.id);
    assert_eq!(probe(&a, &third.head().engine_id), reference, "resumed, bit for bit");
    assert_same_states(&one_shot_states, &probes(&a, &third), "resumed");

    // 3. A dry run says what it would do and does nothing.
    a.context_delete(&third.head().engine_id).unwrap();
    let dry = put(&mut a, &all, None, true);
    assert_eq!((dry.kept, dry.fed), (two.tokens, dry.tokens - two.tokens));
    assert_eq!(held(&mut a, &dry), [true, true, true, false], "the head was not built");
    let real = put(&mut a, &all, None, false);
    assert_eq!((real.kept, real.fed), (dry.kept, dry.fed), "the dry run told the truth");
    assert_eq!(probe(&a, &real.head().engine_id), reference);
    assert_same_states(&one_shot_states, &probes(&a, &real), "after a dry run");

    // 4. An unmarked turn that is held is still not a boundary: a body that
    // declares only the system and the head resumes from the system.
    a.context_delete(&real.head().engine_id).unwrap();
    let unmarked = put(&mut a, &body(3, [false, false, false]), None, false);
    assert_eq!(unmarked.kept, built.snapshots[0].tokens, "only the system head, though the second turn is held");
    assert_eq!(probe(&a, &unmarked.head().engine_id), reference, "and it is the same state");
    assert_same_states(&one_shot_states, &probes(&a, &unmarked), "with no snaps");

    // 5. A pin shows on lookup and is released.
    let pinned = put(&mut a, &all, Some(true), false);
    let id = pinned.head().engine_id.clone();
    assert!(a.council_inspect(std::slice::from_ref(&id)).unwrap()[0].unwrap().pinned);
    assert!(a.council_unpin(&id).unwrap());
    assert!(!a.council_inspect(std::slice::from_ref(&id)).unwrap()[0].unwrap().pinned);
    assert!(!a.council_unpin(&"0".repeat(64)).unwrap(), "nothing held under an unknown id");
}
