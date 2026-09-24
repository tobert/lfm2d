//! Reads the candle fork revision this build resolves from the workspace's
//! `Cargo.lock` and hands it to the crate as `LFM2D_CANDLE_REV`
//! (`adjudicator::CANDLE_REV`, part of every `snapshot_id`). A lockfile
//! without a candle-core entry is a build failure, never a blank identity.

use std::path::Path;

fn main() {
    let lock = Path::new(env!("CARGO_MANIFEST_DIR")).join("../Cargo.lock");
    println!("cargo:rerun-if-changed={}", lock.display());
    let text = std::fs::read_to_string(&lock)
        .unwrap_or_else(|e| panic!("reading {}: {e}", lock.display()));
    let rev = candle_rev(&text).unwrap_or_else(|| {
        panic!("{} has no candle-core package with a source; cannot name the candle build", lock.display())
    });
    println!("cargo:rustc-env=LFM2D_CANDLE_REV={rev}");
}

/// The git commit after `#` for a git source, or `crates.io:<version>` for a
/// registry one. The first `candle-core` package wins; the workspace has one.
fn candle_rev(lock: &str) -> Option<String> {
    for block in lock.split("[[package]]") {
        let field = |key: &str| {
            block.lines().find_map(|l| l.strip_prefix(&format!("{key} = \""))?.strip_suffix('"').map(str::to_string))
        };
        if field("name").as_deref() != Some("candle-core") {
            continue;
        }
        let source = field("source")?;
        return match source.split_once('#') {
            Some((_, commit)) if source.starts_with("git+") => Some(commit.to_string()),
            _ if source.starts_with("registry+") => Some(format!("crates.io:{}", field("version")?)),
            _ => None,
        };
    }
    None
}
