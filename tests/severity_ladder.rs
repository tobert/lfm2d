//! Pins the severity-ladder ordering invariant that consumers mapping
//! ordinal label position → policy depend on (docs/integration.md,
//! invariant 3). kaijutsu's `S50-lfm2d.kai` advisory hook reads
//! `GET /v1/models` and treats index 0 as least severe and the last index
//! as most severe — so "the `labels` array is in ascending severity order"
//! is a wire contract, not an implementation detail.
//!
//! Unlike the Hub configs beside them, the two `kube_ordinal_*.config.json`
//! fixtures are OUR trained heads' configs, captured 2026-08-24 from the
//! staged checkpoints (`.models/kube_ordinal_v9_cal`,
//! `~/.local/share/lfm2-training-data/runs/kube_ordinal_v6`). The daemon
//! serves labels in `id2label` id order (`labels.rs::order_labels`), so id
//! order IS the wire order these tests pin.

use std::collections::HashMap;
use std::path::PathBuf;

fn id_ordered_labels(fixture: &str) -> Vec<String> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(fixture);
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    let cfg: serde_json::Value = serde_json::from_str(&raw).expect("fixture is valid JSON");
    let map: HashMap<String, String> =
        serde_json::from_value(cfg["id2label"].clone()).expect("id2label is a string→string map");
    let mut pairs: Vec<(usize, String)> = map
        .into_iter()
        .map(|(k, v)| (k.parse::<usize>().expect("id2label key is an integer"), v))
        .collect();
    pairs.sort_by_key(|(id, _)| *id);
    pairs.into_iter().map(|(_, l)| l).collect()
}

/// The shipped head's ladder: ascending severity in id order. This is the
/// ordering kaijutsu's ordinal verdict mapping (index 0 → allow, last →
/// deny) is correct against. If a future checkpoint ships labels in any
/// other order, this invariant — and every consumer built on it — breaks.
#[test]
fn v9_cal_labels_are_ascending_severity_in_id_order() {
    assert_eq!(
        id_ordered_labels("kube_ordinal_v9_cal.config.json"),
        ["informative", "situation-normal", "data-critical"],
        "kube_ordinal_v9_cal must list labels least→most severe (invariant 3)"
    );
}

/// kube_ordinal_v6 VIOLATES the ladder invariant: its id2label is
/// alphabetical (`destructive` at id 0), a pre-convention artifact. Under
/// an ordinal-position mapping, id 0 reads as "allow" — so pointing an
/// ordinal consumer at v6 inverts the verdicts. This test exists so the
/// hazard stays pinned, not so anyone "fixes" it by editing the fixture:
/// reordering id2label without permuting the classifier weights would
/// corrupt predictions silently.
#[test]
fn v6_labels_are_alphabetical_not_severity_ordered() {
    let v6 = id_ordered_labels("kube_ordinal_v6.config.json");
    assert_eq!(
        v6,
        ["destructive", "informative", "mutating"],
        "kube_ordinal_v6's real (alphabetical) id order"
    );
    assert_ne!(
        v6,
        ["informative", "mutating", "destructive"],
        "if v6 ever becomes severity-ordered, delete this test and update \
         docs/integration.md invariant 3's rollback warning"
    );
}
