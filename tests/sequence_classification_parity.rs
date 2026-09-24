//! `Lfm2SequenceClassifier::from_trunk` vs `from_dir`, on a real specialist
//! checkpoint — the frozen-trunk-many-heads deployment this feature exists
//! for.
//!
//! `sequence_classification_guards.rs` covers loud failure on synthetic
//! checkpoints and needs no weights. This suite is the opposite half: with
//! REAL weights, does reusing a shared trunk produce the exact same numbers
//! as loading everything together, and does the sharing actually share?
//!
//! `kube_ordinal_v6` is OUR OWN fine-tune (per `sequence_classification.rs`'s
//! module docs, no LiquidAI checkpoint ships this head) — a 350M
//! `Lfm2BidirForSequenceClassification` over kubectl-command risk, produced
//! outside this repo and kept outside the repo tree per this crate's
//! training-data convention. It is exactly the
//! shape `from_trunk` targets: a specialist head trained over an otherwise
//! frozen trunk.
//!
//! These tests need the real weights, which are far too large to commit.
//! They fail loudly with where to find them rather than skipping.

use std::path::PathBuf;
use std::sync::Arc;

use lfm2_encoder::{Lfm2SequenceClassifier, Lfm2Trunk};

/// Where the specialist checkpoint lives; override with `LFM2_SEQ_CLF_DIR`.
/// Not under `.models/` (which holds LiquidAI's own Hub checkpoints) — this
/// one is quarantined training output, per this crate's convention (see
/// `CLAUDE.md` and `.gitignore`).
fn checkpoint() -> PathBuf {
    if let Ok(d) = std::env::var("LFM2_SEQ_CLF_DIR") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").expect("HOME must be set to locate the default checkpoint");
    let dir = PathBuf::from(home).join(".local/share/lfm2-training-data/runs/kube_ordinal_v6");
    assert!(
        dir.join("model.safetensors").is_file(),
        "missing weights at {}\n\n  (point LFM2_SEQ_CLF_DIR at an \
         Lfm2BidirForSequenceClassification checkpoint dir — kube_ordinal_v6's shape, \
         a fine-tuned checkpoint produced outside this repo)\n",
        dir.display(),
    );
    dir
}

/// Kubectl-command-shaped probes spanning the checkpoint's three classes —
/// not asserted against a specific label here (that is the checkpoint's own
/// eval, which lives with its training); the point is only that BOTH
/// loading paths agree, whatever they say.
const PROBES: [&str; 5] = [
    "kubectl get pods -n payments -o wide",
    "kubectl delete namespace staging --wait=false",
    "kubectl rollout restart deployment/checkout -n prod",
    "kubectl exec -it redis-0 -- redis-cli FLUSHALL",
    "clean out everything in the loadtest namespace after standup",
];

/// The parity contract: `from_dir` (loads its own trunk) and `from_trunk`
/// (reuses one loaded via `load_shared`) run the SAME weights through the
/// SAME math, so their outputs must be exactly equal — not merely close.
/// Any divergence here means `from_trunk` is silently touching different
/// tensors (e.g. reading `classifier.*` from the wrong dtype/device, or the
/// shared trunk's forward differing from a freshly-loaded one), which is
/// exactly the silent-corruption shape this crate's house rule refuses.
#[test]
fn from_trunk_matches_from_dir_exactly() {
    let dir = checkpoint();

    let direct = Lfm2SequenceClassifier::from_dir(&dir).expect("from_dir");
    let trunk = Lfm2Trunk::load_shared(&dir).expect("load_shared");
    let shared = Lfm2SequenceClassifier::from_trunk(trunk, &dir).expect("from_trunk");

    assert_eq!(direct.labels(), shared.labels(), "label sets must agree");

    for text in PROBES {
        let want = direct.logits(text).expect("from_dir logits");
        let got = shared.logits(text).expect("from_trunk logits");
        assert_eq!(
            want, got,
            "from_dir and from_trunk must produce bit-identical logits for {text:?}"
        );

        let want_probs = direct.predict(text).expect("from_dir predict");
        let got_probs = shared.predict(text).expect("from_trunk predict");
        assert_eq!(
            want_probs, got_probs,
            "from_dir and from_trunk must produce bit-identical probabilities for {text:?}"
        );
    }
}

/// The sharing contract: two heads built with `from_trunk(Arc::clone(&t), …)`
/// must point at the SAME trunk allocation (no second trunk load happened
/// behind the scenes) and the `Arc`'s strong count must reflect exactly the
/// clones handed out — not more, which would mean an extra hidden clone/leak,
/// and not fewer, which would mean a head dropped its reference and is
/// somehow still working (or about to use-after-free if this were unsafe).
#[test]
fn two_heads_share_one_trunk_load() {
    let dir = checkpoint();

    let trunk = Lfm2Trunk::load_shared(&dir).expect("load_shared");
    assert_eq!(
        Arc::strong_count(&trunk),
        1,
        "a freshly loaded trunk should have exactly one owner"
    );

    let a = Lfm2SequenceClassifier::from_trunk(Arc::clone(&trunk), &dir).expect("from_trunk a");
    let b = Lfm2SequenceClassifier::from_trunk(Arc::clone(&trunk), &dir).expect("from_trunk b");

    assert_eq!(
        Arc::strong_count(&trunk),
        3,
        "the original handle plus two heads should hold three references to one trunk"
    );
    assert!(
        std::ptr::eq(a.trunk(), b.trunk()),
        "both heads must point at the identical trunk allocation, not two separate loads"
    );

    // Dropping one head must not disturb the other's trunk — it's a real
    // Arc, not a borrow with a lifetime trick.
    drop(a);
    assert_eq!(Arc::strong_count(&trunk), 2);
    let _ = b.trunk(); // still usable
}
