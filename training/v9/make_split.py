#!/usr/bin/env python3
"""Stratified train/val split of v9.jsonl, seeded and reproducible.

    python3 training/v9/make_split.py            # write train.jsonl / val.jsonl
    python3 training/v9/make_split.py --check    # verify they regenerate identically

Stratified BY LABEL, because the corpus is 50/19/31 and a naive shuffle on ~677
rows can easily hand the val set a distorted mix — at this size that is noise
large enough to move an accuracy number by a point or two on its own.

The split is seeded and `--check` asserts byte-identity, so a reported metric
can always be traced to the exact rows that produced it (commit-the-scorer).

NOT in here, deliberately:
  * `v9_holdout.jsonl` — the 14 severity-probe texts. build_v9.py already keeps
    them out of v9.jsonl; they are the out-of-corpus eval and must never be
    trained on or validated against.
  * contested rows are KEPT. They carry the generator's label with the
    relabeler's dissent recorded in `note`; dropping them would quietly
    discard the hardest ~4% of the corpus and flatter every metric.

Output rows carry only {"text", "label"} — the shape
finetune_sequence_classifier.py expects. The other five fields are provenance
for humans, not features.
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'v9.jsonl'
SEED = 20260813
VAL_FRACTION = 0.15
# Ordinal order, least to most severe. The Rust side reads id2label by index,
# so this order is part of the export contract, not a display preference.
LABEL_ORDER = ['informative', 'situation-normal', 'data-critical']


def build(val_fraction: float):
    rows = [json.loads(l) for l in SRC.read_text().splitlines() if l.strip()]
    by_label = defaultdict(list)
    for r in rows:
        by_label[r['label']].append(r)

    rng = random.Random(SEED)
    train, val = [], []
    for label in LABEL_ORDER:
        group = sorted(by_label[label], key=lambda r: r['text'])  # stable before shuffle
        rng.shuffle(group)
        n_val = max(1, round(len(group) * val_fraction))
        val.extend(group[:n_val])
        train.extend(group[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    slim = lambda rs: [{'text': r['text'], 'label': r['label']} for r in rs]
    return slim(train), slim(val), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--val-fraction', type=float, default=VAL_FRACTION,
                    help='default 0.15, unchanged from before this flag existed. '
                         'A wider split (e.g. 0.30) trades train rows for a val '
                         'set whose accuracy/gate reading is less exposed to '
                         'single-sample noise -- see the 2026-08-15 ablation, '
                         'where a one-seed 15%% val split was part of why B '
                         'could not reproduce pass 3 exactly.')
    ap.add_argument('--suffix', default='',
                    help="write to train<suffix>.jsonl/val<suffix>.jsonl instead "
                         "of train.jsonl/val.jsonl. Default '' preserves the "
                         "committed filenames every other script expects; use a "
                         "suffix for one-off variants (e.g. --suffix _wide) so "
                         "they never collide with the standing split.")
    args = ap.parse_args()

    train_path = HERE / f'train{args.suffix}.jsonl'
    val_path = HERE / f'val{args.suffix}.jsonl'

    train, val, rows = build(args.val_fraction)
    tb = '\n'.join(json.dumps(r, sort_keys=True) for r in train) + '\n'
    vb = '\n'.join(json.dumps(r, sort_keys=True) for r in val) + '\n'

    if args.check:
        for p, body in ((train_path, tb), (val_path, vb)):
            if not p.exists():
                print(f'{p.name} missing'); return 2
            if p.read_text() != body:
                print(f'DRIFT: {p.name} does not match v9.jsonl'); return 1
        print(f'ok — split regenerates byte-identical '
              f'({len(train)} train / {len(val)} val)')
        return 0

    train_path.write_text(tb)
    val_path.write_text(vb)

    print(f'source: {len(rows)} rows')
    print(f'  train {len(train)}  {dict(sorted(Counter(r["label"] for r in train).items()))}')
    print(f'  val   {len(val)}  {dict(sorted(Counter(r["label"] for r in val).items()))}')
    print(f'\nlabel order (ordinal, least->most severe): {",".join(LABEL_ORDER)}')
    print(f'seed {SEED}, val fraction {args.val_fraction}')
    print(f'wrote {train_path.name} / {val_path.name}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
