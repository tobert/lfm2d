#!/usr/bin/env python3
"""Fine-tune `LiquidAI/LFM2.5-Encoder-350M` (a bidirectional masked-LM
encoder) into a whole-sequence classifier, and export it in the exact
layout the lfm2-encoder Rust crate expects to load directly.

Export contract (fixed — do not change without updating the Rust loader):

    out/model.safetensors
        lfm2.embed_tokens.weight        [vocab, hidden]
        lfm2.embedding_norm.weight      [hidden]
        lfm2.layers.<N>....             (whatever the base checkpoint has)
        classifier.weight               [num_labels, hidden]
        classifier.bias                 [num_labels]
    out/config.json                     base config + architectures override
                                         + id2label / label2id (id2label
                                         keys are the STRINGS "0","1",...)
    out/tokenizer.json                  copied verbatim from the base dir

Pooling is CLS (hidden state at sequence position 0), matching the Rust
side. This is NOT mean pooling.

See training/README.md for the two checkpoint-loading gotchas this script
works around (`transformers`' dynamic-module dotted-path bug, and a silent
weight-reinitialization footgun in `AutoModel.from_pretrained`).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import tempfile
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer, get_linear_schedule_with_warmup

EXPECTED_NON_BACKBONE_KEYS = {"classifier.weight", "classifier.bias"}


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# Checkpoint loading gotchas
# --------------------------------------------------------------------------


def dot_free_checkpoint_dir(base_dir: Path) -> Path:
    """`transformers`' dynamic-module loader (for `trust_remote_code=True`)
    turns the checkpoint directory's basename into a Python module path
    under `transformers_modules.<basename>...`. Every LFM2.5 checkpoint
    directory name contains a literal `.` (`LFM2.5-...`), which produces an
    invalid module path and breaks the import. Work around it with a
    dot-free symlink and load from that instead of the original path.
    """
    if "." not in base_dir.name:
        return base_dir
    link_parent = Path(tempfile.mkdtemp(prefix="lfm2-ckpt-"))
    link_dir = link_parent / base_dir.name.replace(".", "_")
    link_dir.symlink_to(base_dir.resolve())
    return link_dir


def load_backbone(base_dir: Path):
    """Load the pretrained bidirectional trunk.

    Two gotchas, fixture-verified against the real
    `LiquidAI/LFM2.5-Encoder-350M` checkpoint on 2026-08-04:

    1. `AutoModel.from_pretrained(dir, trust_remote_code=True)` on this
       checkpoint SILENTLY RE-INITIALIZES every backbone weight. The
       checkpoint's own safetensors keys are saved with a `lfm2.` prefix
       (the checkpoint is the `Lfm2BidirectionalForMaskedLM` wrapper), but
       the bare `Lfm2BidirectionalModel` class that plain `AutoModel`
       resolves to (via `auto_map`) never overrides `base_model_prefix`,
       so it defaults to transformers' generic `"model"`. HF's automatic
       checkpoint-prefix-stripping only fires when the checkpoint's prefix
       matches the target's `base_model_prefix`; `"lfm2."` != `"model."`,
       so every tensor comes back freshly (randomly) initialized. The only
       sign is an easy-to-miss log line ("newly initialized") — no
       exception, no loud failure. Fine-tuning on top of that silently
       trains a randomly-initialized network while believing it's the
       pretrained checkpoint.
    2. `Lfm2BidirectionalForMaskedLM.base_model_prefix = "lfm2"` IS set
       explicitly in the checkpoint's custom code, so loading through
       `AutoModelForMaskedLM` (which resolves to that wrapper class) loads
       correctly, and its `.lfm2` submodule is exactly what the export
       contract wants under that same attribute name.

    Conclusion: always load via `AutoModelForMaskedLM` and take `.lfm2`.
    Never load the backbone via bare `AutoModel`.
    """
    mlm = AutoModelForMaskedLM.from_pretrained(str(base_dir), trust_remote_code=True)
    return mlm.lfm2, mlm.config


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


TARGET_SUM_TOLERANCE = 1e-6


def _validate_target_row(target: object, context: str) -> dict[str, float]:
    """Validate one row's `"target"` field in isolation (no access to the
    global label set yet — that check happens later, once the label order
    is known, via `validate_target_keys`). Fails loudly: a malformed target
    is a data bug, not something to coerce or drop silently.
    """
    if not isinstance(target, dict) or isinstance(target, (list, tuple)):
        raise ValueError(f"{context}: target must be an object, got {type(target).__name__}")
    if not target:
        raise ValueError(f"{context}: target must not be empty")
    normalized: dict[str, float] = {}
    for key, value in target.items():
        if not isinstance(key, str):
            raise ValueError(f"{context}: target keys must be strings, got {type(key).__name__}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"{context}: target[{key!r}] must be numeric, got {type(value).__name__}"
            )
        normalized[key] = float(value)
    total = sum(normalized.values())
    if abs(total - 1.0) > TARGET_SUM_TOLERANCE:
        raise ValueError(
            f"{context}: target values sum to {total} (expected 1.0 ± {TARGET_SUM_TOLERANCE})"
        )
    return normalized


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "text" not in obj or "label" not in obj:
                raise ValueError(
                    f"{path}:{lineno}: expected keys 'text' and 'label', got {sorted(obj)}"
                )
            if not isinstance(obj["label"], str):
                raise ValueError(
                    f"{path}:{lineno}: label must be a string, got {type(obj['label']).__name__}"
                )
            target = None
            if "target" in obj:
                target = _validate_target_row(obj["target"], f"{path}:{lineno}")
            items.append({"text": str(obj["text"]), "label": obj["label"], "target": target})
    if not items:
        raise ValueError(f"{path}: no examples found")
    return items


def resolve_label_order(items: list[dict], explicit_order: list[str] | None = None) -> list[str]:
    """Decide the fixed 0..N-1 label order used everywhere downstream
    (label2id/id2label, classifier head width and slot assignment).

    Default (unchanged from before `target` existed): sort the labels
    observed in `items`. Pass `explicit_order` to pin the order instead
    (e.g. the v7 ordinal axis 0=informative/1=situation-normal/
    2=data-critical) rather than letting data order or alphabetical
    sorting decide it — `explicit_order` may also list classes with zero
    rows in `items`, which is how a fixed-width head survives an uneven
    split. Fails loudly on duplicates or on an observed label the
    explicit order doesn't account for.
    """
    observed = {item["label"] for item in items}
    if explicit_order is None:
        return sorted(observed)
    order = list(explicit_order)
    if len(order) != len(set(order)):
        raise ValueError(f"explicit label order contains duplicates: {order}")
    unaccounted = observed - set(order)
    if unaccounted:
        raise ValueError(
            f"training data contains labels not present in the explicit label order: "
            f"{sorted(unaccounted)} (explicit order: {order})"
        )
    return order


def validate_target_keys(items: list[dict], label_set: set[str], source: str) -> None:
    """Second-stage target validation: per-row shape (dict, numeric,
    sums to 1) is already checked by `load_jsonl`; this checks each
    target's keys are a subset of the resolved label set, which isn't
    known until label order is resolved across the whole file. Fails
    loudly rather than silently dropping unknown classes.
    """
    for i, item in enumerate(items):
        target = item.get("target")
        if target is None:
            continue
        unknown = set(target) - label_set
        if unknown:
            raise ValueError(
                f"{source} item {i}: target keys {sorted(unknown)} not in label set "
                f"{sorted(label_set)}"
            )


def target_to_vector(
    label_id: int, target: dict[str, float] | None, label2id: dict[str, int]
) -> torch.Tensor:
    """Build the per-row soft-target distribution the loss trains against,
    in label2id's fixed slot order. Rows without a `"target"` field get an
    exact one-hot at their majority `label_id` — this is what makes the
    soft-target loss collapse to today's hard cross-entropy for old-style
    rows (see the numeric equivalence test).
    """
    vec = torch.zeros(len(label2id), dtype=torch.float32)
    if target is None:
        vec[label_id] = 1.0
    else:
        for label, prob in target.items():
            vec[label2id[label]] = prob
    return vec


class ClassificationDataset(Dataset):
    def __init__(self, items: list[dict], label2id: dict[str, int]):
        self.items = items
        self.label2id = label2id

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        label_id = self.label2id[item["label"]]
        target_vec = target_to_vector(label_id, item.get("target"), self.label2id)
        return item["text"], label_id, target_vec


def make_collate(tokenizer, max_len: int):
    def collate(batch):
        texts, labels, targets = zip(*batch)
        enc = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        return enc, torch.tensor(labels, dtype=torch.long), torch.stack(targets)

    return collate


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


class Lfm2ForSequenceClassification(nn.Module):
    """Backbone kept under the attribute name `lfm2` so `state_dict()`
    produces keys matching the export contract exactly
    (`lfm2.embed_tokens.weight`, `lfm2.layers.N...`,
    `lfm2.embedding_norm.weight`), plus a CLS-pooled linear head.
    """

    def __init__(self, lfm2_backbone: nn.Module, hidden_size: int, num_labels: int):
        super().__init__()
        self.lfm2 = lfm2_backbone
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.lfm2(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        cls_hidden = out.last_hidden_state[:, 0, :]  # position 0 = CLS/BOS, NOT mean pooling
        return self.classifier(cls_hidden)


# --------------------------------------------------------------------------
# Train / eval
# --------------------------------------------------------------------------


def soft_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, class_weight: torch.Tensor | None = None
) -> torch.Tensor:
    """-sum(target * log_softmax(logits)), averaged over the batch.

    `targets` is a [batch, num_labels] probability distribution (rows sum
    to 1). When every row is an exact one-hot (the case for all "label"-
    only data, old and new), this is numerically identical to
    `nn.functional.cross_entropy(logits, hard_labels)` — see
    `test_soft_targets.py`'s one-hot-equivalence test.

    `class_weight`, if given, is a [num_labels] vector. Per-example loss is
    weighted by `(targets * class_weight).sum(-1)` (the target-weighted
    generalization of `nn.CrossEntropyLoss(weight=...)` — collapses to the
    usual per-class weight for one-hot targets) and the batch is averaged by
    SUM of weights, not count, matching `nn.CrossEntropyLoss`'s own
    normalization so an inverse-frequency weight vector actually rebalances
    the effective per-class contribution rather than just rescaling the
    overall loss magnitude.
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    per_example = -(targets * log_probs).sum(dim=-1)
    if class_weight is None:
        return per_example.mean()
    w = (targets * class_weight).sum(dim=-1)
    return (per_example * w).sum() / w.sum().clamp_min(1e-12)


@torch.no_grad()
def evaluate(model, loader, device, autocast_dtype, labels_sorted: list[str]):
    model.eval()
    correct = 0
    total = 0
    tp: Counter = Counter()
    fp: Counter = Counter()
    fn: Counter = Counter()
    support: Counter = Counter()

    for enc, labels, _targets in loader:
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        labels = labels.to(device)
        with torch.autocast(
            device_type=device.type, dtype=autocast_dtype, enabled=device.type == "cuda"
        ):
            logits = model(input_ids, attention_mask)
        preds = logits.float().argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
        for pred, gold in zip(preds.tolist(), labels.tolist()):
            support[gold] += 1
            if pred == gold:
                tp[gold] += 1
            else:
                fp[pred] += 1
                fn[gold] += 1

    acc = correct / max(1, total)
    per_class = {}
    for i, label in enumerate(labels_sorted):
        denom_p = tp[i] + fp[i]
        denom_r = tp[i] + fn[i]
        precision = tp[i] / denom_p if denom_p > 0 else 0.0
        recall = tp[i] / denom_r if denom_r > 0 else 0.0
        per_class[label] = (precision, recall, support[i])
    return acc, per_class


def export_checkpoint(
    model: Lfm2ForSequenceClassification,
    id2label: dict[str, str],
    base_dir: Path,
    out_dir: Path,
) -> None:
    """Write `model.safetensors` + `config.json` + `tokenizer.json` to
    `out_dir` in the exact layout the Rust loader depends on. Fails loudly
    (assertion, not a warning) rather than write a checkpoint the Rust side
    cannot load or would load wrong.
    """
    num_labels = len(id2label)

    clf_weight = model.classifier.weight
    assert clf_weight.shape[0] == num_labels, (
        f"classifier.weight first dim {clf_weight.shape[0]} != len(id2label) {num_labels} "
        "— refusing to export a head that disagrees with its own label map"
    )

    state_dict = model.state_dict()
    non_backbone = {k for k in state_dict if not k.startswith("lfm2.")}
    assert non_backbone == EXPECTED_NON_BACKBONE_KEYS, (
        "exported state_dict does not match the fixed Rust-loader contract: "
        f"expected non-backbone keys {sorted(EXPECTED_NON_BACKBONE_KEYS)}, got {sorted(non_backbone)}"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    tensors = OrderedDict(
        (k, v.detach().to("cpu").contiguous()) for k, v in state_dict.items()
    )
    weights_path = out_dir / "model.safetensors"
    save_file(tensors, str(weights_path))
    # safetensors writes through mkstemp, which creates mode 600 regardless
    # of umask — unlike every other file this export writes. Normalize so
    # downstream consumers that read as a different user (lfm2d's non-root
    # container over a hostPath staging copy) can read the weights; the
    # training tree's privacy is enforced by its 0700 top-level dir, not
    # by per-file modes. Found the hard way staging v6/v8 to k3s.
    weights_path.chmod(0o644)

    base_config_path = base_dir / "config.json"
    with open(base_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    config["architectures"] = ["Lfm2BidirForSequenceClassification"]
    config["id2label"] = id2label
    config["label2id"] = {label: int(idx) for idx, label in id2label.items()}
    config["num_labels"] = num_labels
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    tok_src = base_dir / "tokenizer.json"
    if not tok_src.is_file():
        raise FileNotFoundError(f"base checkpoint is missing tokenizer.json at {tok_src}")
    shutil.copyfile(tok_src, out_dir / "tokenizer.json")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", required=True, type=Path, help="training JSONL: {\"text\":..., \"label\":...}")
    p.add_argument("--val", required=True, type=Path, help="validation JSONL, same format")
    p.add_argument("--base", required=True, type=Path, help="base checkpoint directory")
    p.add_argument("--out", required=True, type=Path, help="output directory for the exported classifier")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument(
        "--class-weight",
        choices=["none", "inverse_freq"],
        default="none",
        help=(
            "none (default): unweighted, unchanged from before this flag existed. "
            "inverse_freq: sklearn-'balanced'-style weight, "
            "n_total / (n_classes * count[c]), computed from --train's own label "
            "counts and printed at startup. Tests whether a skewed class prior "
            "(2026-08-15 ablation: v9's data-critical share climbed pass over "
            "pass) is shifting the model toward over-predicting the majority "
            "class rather than genuinely separating cases."
        ),
    )
    p.add_argument(
        "--export-every-epoch",
        action="store_true",
        help=(
            "In addition to the best-val-acc export at --out, also export EVERY "
            "epoch's checkpoint to <out>-e<N>. For probing whether drift "
            "(e.g. benign-control score creep) appears at a specific epoch that "
            "val_acc alone doesn't flag -- accuracy has already been shown "
            "(2026-08-15 ablation) not to catch this by itself."
        ),
    )
    p.add_argument(
        "--label-order",
        type=str,
        default=None,
        help=(
            "comma-separated explicit label order, e.g. "
            "'informative,situation-normal,data-critical'. Fixes id2label's slot "
            "assignment instead of deriving it by sorting labels observed in "
            "--train; may include a class with zero training rows. Default: "
            "sorted(observed labels), unchanged from before this flag existed."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        autocast_dtype = torch.bfloat16
    elif device.type == "cuda":
        autocast_dtype = torch.float16
    else:
        autocast_dtype = torch.float32
    print(f"device={device}  autocast_dtype={autocast_dtype}", flush=True)

    train_items = load_jsonl(args.train)
    val_items = load_jsonl(args.val)

    explicit_order = args.label_order.split(",") if args.label_order else None
    labels_sorted = resolve_label_order(train_items, explicit_order)
    val_labels = {item["label"] for item in val_items}
    unseen = val_labels - set(labels_sorted)
    if unseen:
        raise ValueError(
            f"validation set contains labels never seen in training: {sorted(unseen)} "
            "— fix the data, this is not something to silently ignore"
        )

    label2id = {label: i for i, label in enumerate(labels_sorted)}
    id2label = {str(i): label for label, i in label2id.items()}
    print(f"labels ({len(labels_sorted)}): {labels_sorted}")
    print(f"train examples: {len(train_items)}  val examples: {len(val_items)}")

    label_set = set(labels_sorted)
    validate_target_keys(train_items, label_set, str(args.train))
    validate_target_keys(val_items, label_set, str(args.val))

    base_dir = dot_free_checkpoint_dir(args.base)
    tokenizer = AutoTokenizer.from_pretrained(str(base_dir))
    lfm2_backbone, base_config = load_backbone(base_dir)

    model = Lfm2ForSequenceClassification(
        lfm2_backbone, base_config.hidden_size, len(labels_sorted)
    ).to(device)

    collate = make_collate(tokenizer, args.max_len)
    train_ds = ClassificationDataset(train_items, label2id)
    val_ds = ClassificationDataset(val_items, label2id)

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate
    )

    class_weight_vec = None
    if args.class_weight == "inverse_freq":
        counts = Counter(label2id[item["label"]] for item in train_items)
        n_total = len(train_items)
        n_classes = len(labels_sorted)
        class_weight_vec = torch.tensor(
            [n_total / (n_classes * max(1, counts[i])) for i in range(n_classes)],
            dtype=torch.float32,
        ).to(device)
        print(
            "class_weight=inverse_freq  "
            + "  ".join(
                f"{labels_sorted[i]}={class_weight_vec[i].item():.3f}(n={counts[i]})"
                for i in range(n_classes)
            )
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    use_scaler = device.type == "cuda" and autocast_dtype == torch.float16
    scaler = torch.amp.GradScaler(device="cuda" if device.type == "cuda" else "cpu", enabled=use_scaler)

    best_val_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for enc, labels, targets in train_loader:
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            labels = labels.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=autocast_dtype, enabled=device.type == "cuda"
            ):
                logits = model(input_ids, attention_mask)
                loss = soft_cross_entropy(logits.float(), targets, class_weight_vec)

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss / max(1, n_batches)
        val_acc, per_class = evaluate(model, val_loader, device, autocast_dtype, labels_sorted)

        print(f"epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  val_acc={val_acc:.4f}")
        for label, (precision, recall, support) in per_class.items():
            print(
                f"    {label:>20s}  precision={precision:.3f}  recall={recall:.3f}  support={support}"
            )

        if args.export_every_epoch:
            epoch_out = args.out.parent / f"{args.out.name}-e{epoch}"
            export_checkpoint(model, id2label, base_dir, epoch_out)
            print(f"  -> exported epoch {epoch} to {epoch_out} (--export-every-epoch)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  -> new best val_acc={val_acc:.4f}; exporting to {args.out}")
            export_checkpoint(model, id2label, base_dir, args.out)

    if best_val_acc < 0:
        raise RuntimeError("training produced no validation result; nothing was exported")
    print(f"done. best val_acc={best_val_acc:.4f}. export at {args.out}")


if __name__ == "__main__":
    main()
