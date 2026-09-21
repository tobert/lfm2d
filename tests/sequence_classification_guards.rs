//! Loud-failure guards for [`Lfm2SequenceClassifier`], on a synthetic tiny
//! checkpoint — same spirit as `tests/trunk_guards.rs`, but this head's only
//! public entry point is directory-based (`from_dir`/`from_dir_with`), not
//! an in-memory `VarBuilder`, so these tests write a full synthetic
//! checkpoint directory (`config.json` + `tokenizer.json` +
//! `model.safetensors`) to a tempdir rather than skip straight to a
//! `VarBuilder::from_tensors`. No downloaded weights are needed; these run
//! anywhere.
//!
//! The house rule these encode: a wrong load or a wrong call must crash,
//! not return plausible numbers. Each test names the corruption it prevents.

use std::collections::HashMap;
use std::sync::Arc;

use ahash::AHashMap;
use candle_core::{DType, Device, Tensor};
use lfm2_encoder::{Lfm2SequenceClassifier, Lfm2Trunk};
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::Tokenizer;

const HIDDEN: usize = 64;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = HIDDEN / HEADS;
const FFN: usize = 128;
const VOCAB: usize = 100;
const K: usize = 3;
const NUM_LABELS: usize = 3;

/// Two layers, one of each kind, plus `id2label` — enough to exercise the
/// trunk AND the classifier head. `id2label` is passed in as a raw JSON
/// fragment so the gap-detection test can hand in a deliberately broken one.
fn tiny_config_json(id2label_json: &str) -> String {
    format!(
        r#"{{
            "architectures": ["Lfm2BidirForSequenceClassification"],
            "vocab_size": {VOCAB},
            "hidden_size": {HIDDEN},
            "num_hidden_layers": 2,
            "num_attention_heads": {HEADS},
            "num_key_value_heads": {KV_HEADS},
            "layer_types": ["conv", "full_attention"],
            "max_position_embeddings": 512,
            "intermediate_size": {FFN},
            "block_multiple_of": 1,
            "conv_L_cache": {K},
            "conv_bias": false,
            "use_pos_enc": true,
            "rope_theta": 1000000.0,
            "id2label": {id2label_json}
        }}"#
    )
}

/// A well-formed 3-label `id2label` — the happy-path fixture every test
/// except the gap test uses.
const GOOD_ID2LABEL: &str = r#"{"0": "safe", "1": "risky", "2": "dangerous"}"#;

fn zeros(dims: &[usize], map: &mut HashMap<String, Tensor>, name: String) {
    map.insert(
        name,
        Tensor::zeros(dims, DType::F32, &Device::Cpu).unwrap(),
    );
}

/// Every tensor the trunk needs, under the `lfm2.` prefix every
/// head-carrying checkpoint uses (see `tests/trunk_guards.rs`).
fn tiny_trunk_weights(map: &mut HashMap<String, Tensor>) {
    let p = "lfm2.";
    zeros(&[VOCAB, HIDDEN], map, format!("{p}embed_tokens.weight"));
    zeros(&[HIDDEN], map, format!("{p}embedding_norm.weight"));
    for i in 0..2 {
        let l = format!("{p}layers.{i}");
        zeros(&[HIDDEN], map, format!("{l}.operator_norm.weight"));
        zeros(&[HIDDEN], map, format!("{l}.ffn_norm.weight"));
        zeros(&[FFN, HIDDEN], map, format!("{l}.feed_forward.w1.weight"));
        zeros(&[HIDDEN, FFN], map, format!("{l}.feed_forward.w2.weight"));
        zeros(&[FFN, HIDDEN], map, format!("{l}.feed_forward.w3.weight"));
        if i == 0 {
            zeros(&[HIDDEN, 1, K], map, format!("{l}.conv.conv.weight"));
            zeros(&[3 * HIDDEN, HIDDEN], map, format!("{l}.conv.in_proj.weight"));
            zeros(&[HIDDEN, HIDDEN], map, format!("{l}.conv.out_proj.weight"));
        } else {
            let a = format!("{l}.self_attn");
            zeros(&[HEADS * HEAD_DIM, HIDDEN], map, format!("{a}.q_proj.weight"));
            zeros(&[KV_HEADS * HEAD_DIM, HIDDEN], map, format!("{a}.k_proj.weight"));
            zeros(&[KV_HEADS * HEAD_DIM, HIDDEN], map, format!("{a}.v_proj.weight"));
            zeros(&[HIDDEN, HEADS * HEAD_DIM], map, format!("{a}.out_proj.weight"));
            zeros(&[HEAD_DIM], map, format!("{a}.q_layernorm.weight"));
            zeros(&[HEAD_DIM], map, format!("{a}.k_layernorm.weight"));
        }
    }
}

/// A classifier head with a zero weight and a distinguishing bias
/// `[0, 1, 2, …]`. With an all-zero trunk the CLS-pooled vector is exactly
/// zero regardless of the input text, so `logits == bias` exactly — the
/// argmax (and the whole softmax) is pinned and predictable, which is what
/// lets the `predict`/`predict_label` consistency test assert an exact
/// expected label instead of merely "some label".
fn tiny_classifier_weights(map: &mut HashMap<String, Tensor>, num_labels: usize) {
    zeros(&[num_labels, HIDDEN], map, "classifier.weight".to_string());
    let bias: Vec<f32> = (0..num_labels).map(|i| i as f32).collect();
    map.insert(
        "classifier.bias".to_string(),
        Tensor::new(bias.as_slice(), &Device::Cpu).unwrap(),
    );
}

fn full_weights(num_labels: usize) -> HashMap<String, Tensor> {
    let mut map = HashMap::new();
    tiny_trunk_weights(&mut map);
    tiny_classifier_weights(&mut map, num_labels);
    map
}

/// A minimal real tokenizer (WordLevel + whitespace splitting) covering just
/// the words these tests use as input text. Built programmatically rather
/// than hand-written as JSON so the on-disk `tokenizer.json` is exactly what
/// the `tokenizers` crate itself would produce.
fn tiny_tokenizer() -> Tokenizer {
    let mut vocab: AHashMap<String, u32> = AHashMap::default();
    vocab.insert("<unk>".to_string(), 0);
    vocab.insert("hello".to_string(), 1);
    vocab.insert("world".to_string(), 2);
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token("<unk>".to_string())
        .build()
        .expect("build tiny WordLevel model");
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(Whitespace));
    tokenizer
}

/// Write a full checkpoint directory (config.json + tokenizer.json +
/// model.safetensors) to a tempdir and return it — the tempdir must be kept
/// alive for the duration of the test, since it deletes itself on drop.
fn write_checkpoint(config_json: &str, weights: HashMap<String, Tensor>) -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("create temp checkpoint dir");
    std::fs::write(dir.path().join("config.json"), config_json).expect("write config.json");
    tiny_tokenizer()
        .save(dir.path().join("tokenizer.json"), false)
        .expect("write tokenizer.json");
    candle_core::safetensors::save(&weights, dir.path().join("model.safetensors"))
        .expect("write model.safetensors");
    dir
}

#[test]
fn loads_under_lfm2_prefix_and_predicts_a_valid_softmax() {
    let dir = write_checkpoint(&tiny_config_json(GOOD_ID2LABEL), full_weights(NUM_LABELS));
    let clf = Lfm2SequenceClassifier::from_dir(dir.path()).expect("load synthetic checkpoint");
    assert_eq!(clf.num_labels(), NUM_LABELS);
    assert_eq!(clf.labels(), &["safe", "risky", "dangerous"]);

    let probs = clf.predict("hello world").expect("predict");
    assert_eq!(probs.len(), NUM_LABELS, "one probability per label");

    for &p in &probs {
        assert!((0.0..=1.0).contains(&p), "probability out of range: {p}");
    }
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "softmax must sum to 1.0, got {sum}");
}

/// `logits` is the pre-softmax escape hatch this head exists to provide
/// (calibration/thresholding on an advisory model); `predict` must be
/// exactly softmax(logits), not some other normalization.
#[test]
fn predict_is_softmax_of_logits() {
    let dir = write_checkpoint(&tiny_config_json(GOOD_ID2LABEL), full_weights(NUM_LABELS));
    let clf = Lfm2SequenceClassifier::from_dir(dir.path()).expect("load synthetic checkpoint");

    let logits = clf.logits("hello world").expect("logits");
    let probs = clf.predict("hello world").expect("predict");
    assert_eq!(logits.len(), probs.len());

    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let denom: f32 = exps.iter().sum();
    for (i, &p) in probs.iter().enumerate() {
        let expected = exps[i] / denom;
        assert!((p - expected).abs() < 1e-6, "index {i}: {p} vs {expected}");
    }
}

/// The corruption this prevents: a training bug or a hand-edited
/// `id2label` that skips an id would otherwise leave a class permanently
/// unreachable with no error — every prediction would silently be over the
/// wrong label set.
#[test]
fn an_id2label_gap_fails_loudly() {
    // Skips id 1: ids present are {0, 2} but the map has 3 entries, so id 2
    // is out of range for "3 labels" — a real gap, not merely fewer labels.
    let gappy = r#"{"0": "safe", "1": "risky", "3": "dangerous"}"#;
    let dir = write_checkpoint(&tiny_config_json(gappy), full_weights(3));

    let err = Lfm2SequenceClassifier::from_dir(dir.path())
        .expect_err("a gap in id2label must not silently resolve");
    let msg = err.to_string();
    assert!(msg.contains("id2label"), "{msg}");
    assert!(msg.contains('3'), "error should name the offending id: {msg}");
}

/// The corruption this prevents: a `classifier.weight` that disagrees with
/// `id2label`'s length (e.g. a trainer bug, or labels edited post-hoc
/// without retraining the head) would otherwise load a classifier whose
/// output rows don't correspond to the label vector at all — every label
/// past the shorter of the two would either be unreachable or index out of
/// bounds. A shape error is fine; silently loading is not.
#[test]
fn a_classifier_weight_shape_disagreeing_with_id2label_fails_loudly() {
    // id2label declares 3 labels; classifier.weight is built for 5.
    let dir = write_checkpoint(&tiny_config_json(GOOD_ID2LABEL), full_weights(5));

    let err = Lfm2SequenceClassifier::from_dir(dir.path())
        .expect_err("classifier.weight shape disagreeing with id2label must not silently load");
    let msg = err.to_string();
    assert!(msg.contains("classifier"), "{msg}");
}

/// `predict_label` must name the same class `predict` puts the most mass
/// on — not some independently-computed argmax that could drift from it.
/// With a zero trunk/classifier-weight and bias `[0, 1, 2]`, the logits are
/// pinned to the bias regardless of input text, so the expected label
/// ("dangerous", the highest bias) is known ahead of time too.
#[test]
fn predict_label_matches_the_argmax_of_predict() {
    let dir = write_checkpoint(&tiny_config_json(GOOD_ID2LABEL), full_weights(NUM_LABELS));
    let clf = Lfm2SequenceClassifier::from_dir(dir.path()).expect("load synthetic checkpoint");

    let probs = clf.predict("hello world").expect("predict");
    let (expected_idx, expected_prob) = probs
        .iter()
        .copied()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .expect("non-empty probability vector");
    let expected_label = clf.labels()[expected_idx].clone();

    let (label, prob) = clf.predict_label("hello world").expect("predict_label");
    assert_eq!(label, expected_label, "predict_label must agree with predict's argmax");
    assert!((prob - expected_prob).abs() < 1e-6);

    // Pinned by construction (see the doc comment above) — not just
    // internally consistent, but the specific class the bias favors.
    assert_eq!(label, "dangerous");
}

/// A config JSON shaped like [`tiny_config_json`], but with an
/// independently chosen `hidden_size`/head geometry — used only to build a
/// checkpoint whose trunk SHAPE disagrees with the standard tiny trunk, for
/// the `from_trunk` mismatch guard below. `from_trunk`'s
/// `check_compatible` check runs before any weight is ever read, so this
/// checkpoint's own (unused) weights don't need to match `OTHER_HIDDEN`.
fn mismatched_config_json() -> String {
    const OTHER_HIDDEN: usize = 128;
    const OTHER_HEADS: usize = 4;
    const OTHER_KV: usize = 2;
    format!(
        r#"{{
            "architectures": ["Lfm2BidirForSequenceClassification"],
            "vocab_size": {VOCAB},
            "hidden_size": {OTHER_HIDDEN},
            "num_hidden_layers": 2,
            "num_attention_heads": {OTHER_HEADS},
            "num_key_value_heads": {OTHER_KV},
            "layer_types": ["conv", "full_attention"],
            "max_position_embeddings": 512,
            "intermediate_size": {FFN},
            "block_multiple_of": 1,
            "conv_L_cache": {K},
            "conv_bias": false,
            "use_pos_enc": true,
            "rope_theta": 1000000.0,
            "id2label": {GOOD_ID2LABEL}
        }}"#
    )
}

/// The corruption this prevents: pairing a shared trunk with a specialist
/// checkpoint trained for a DIFFERENT hidden size. `from_trunk` must catch
/// this before ever touching a weight tensor, and must name the field that
/// disagrees rather than surfacing a bare matmul shape error.
#[test]
fn from_trunk_over_a_differently_shaped_head_checkpoint_is_refused() {
    let trunk_dir = write_checkpoint(&tiny_config_json(GOOD_ID2LABEL), full_weights(NUM_LABELS));
    let trunk = Lfm2Trunk::load_shared(trunk_dir.path()).expect("load_shared");

    let head_dir = write_checkpoint(&mismatched_config_json(), full_weights(NUM_LABELS));
    let err = Lfm2SequenceClassifier::from_trunk(Arc::clone(&trunk), head_dir.path())
        .expect_err("a head checkpoint with a different hidden_size must be refused");
    let msg = err.to_string();
    assert!(msg.contains("hidden_size"), "{msg}");
}

/// The parity contract, on a synthetic (hermetic, no real weights) fixture:
/// `from_dir` and `from_trunk(load_shared(dir), dir)` over the SAME
/// checkpoint directory must produce bit-identical logits. This is the same
/// property `tests/sequence_classification_parity.rs` checks against a real
/// checkpoint; kept here too because it needs no downloaded weights and so
/// runs on every `cargo test`, not just when a specialist checkpoint is
/// present on disk.
#[test]
fn from_trunk_over_a_matching_checkpoint_matches_from_dir_exactly() {
    let dir = write_checkpoint(&tiny_config_json(GOOD_ID2LABEL), full_weights(NUM_LABELS));

    let direct = Lfm2SequenceClassifier::from_dir(dir.path()).expect("from_dir");
    let trunk = Lfm2Trunk::load_shared(dir.path()).expect("load_shared");
    let shared = Lfm2SequenceClassifier::from_trunk(trunk, dir.path()).expect("from_trunk");

    let want = direct.logits("hello world").expect("from_dir logits");
    let got = shared.logits("hello world").expect("from_trunk logits");
    assert_eq!(want, got, "from_dir and from_trunk must agree exactly on a synthetic checkpoint too");
}
