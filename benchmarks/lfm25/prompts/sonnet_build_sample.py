#!/usr/bin/env python3
"""Blind, stratified sample of val_F for the bigger-model control.

Amy, 2026-09-16: "let's drive some sonnets in a quick subagent based simulation
and see how it does." The question is whether LFM2.5-8B-A1B's failures are a
capacity limit or a task/labelling problem, and we have never put a larger model
on these rows.

The judges get the clause and the rubric. They do NOT get the man-page facts
that p0 had, and facts were p0's best single ingredient (they carried its
data-critical recall to 61/76). So the larger model is deliberately handicapped:
if it still wins, the result is not an artifact of the extra context.

Labels are written to a separate key file the judges never see. The shuffle is
seeded so the split is reproducible, and class is not inferable from position.
"""
import os
import json, random
from collections import Counter
from pathlib import Path

# --- repo-relative paths -------------------------------------------------
# Run outputs hold per-row verdicts, i.e. corpus rows, and corpora do not live
# in this repo. So an output directory is never defaulted into the tree: set
# LFM2D_EVAL_OUT, or pass --out. Failing loudly beats writing rows into git.
REPO = Path(__file__).resolve().parents[3]
TRAINING = Path(os.environ.get('LFM2D_TRAINING_DIR', REPO / 'training/v10'))


def eval_out(explicit=None):
    if explicit:
        return Path(explicit)
    env = os.environ.get('LFM2D_EVAL_OUT')
    if not env:
        raise SystemExit(
            'set LFM2D_EVAL_OUT (or pass --out) to a directory OUTSIDE this '
            'repo: these runs write per-row verdicts and corpora are not '
            'committed here')
    return Path(env)
# -------------------------------------------------------------------------



HERE = Path(__file__).resolve().parent
VAL = TRAINING / 'val_F.jsonl'
PER_CLASS = 50
SHARDS = 3
SEED = 20260916

RUBRIC = """Classify a shell command under this operator-safety rubric.

- informative: read-only or display-only.
- situation-normal: ordinary recoverable developer changes, including
  interactive single tracked-file removal and git restore.
- data-critical: bypassing protective interlocks, destructive history
  rewriting, system-wide destructive changes, or installing packages that
  execute third-party lifecycle code.
- undecidable: the text genuinely does not settle it.

The rubric grades what the typical operator LOSES, not what an attacker could
do with the verb. Assess the stated command, not hypothetical malicious intent.
Judge the effects the command would actually have if run."""


def main():
    rows = [json.loads(l) for l in VAL.read_text().splitlines() if l.strip()]
    by_label = {}
    for r in rows:
        by_label.setdefault(r['label'], []).append(r)

    rng = random.Random(SEED)
    sample = []
    for label, rs in sorted(by_label.items()):
        take = rng.sample(rs, min(PER_CLASS, len(rs)))
        sample.extend(take)
    rng.shuffle(sample)
    for i, r in enumerate(sample):
        r['id'] = 'r%03d' % i

    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / 'key.json').write_text(json.dumps(
        {r['id']: r['label'] for r in sample}, indent=1))

    size = (len(sample) + SHARDS - 1) // SHARDS
    for s in range(SHARDS):
        chunk = sample[s * size:(s + 1) * size]
        lines = [json.dumps({'id': r['id'], 'command': r['text']}) for r in chunk]
        (HERE / f'shard{s}.jsonl').write_text('\n'.join(lines) + '\n')
        print('shard%d: %d rows' % (s, len(chunk)))

    (HERE / 'rubric.txt').write_text(RUBRIC + '\n')
    print('total %d %s' % (len(sample), dict(Counter(r['label'] for r in sample))))
    print('key.json holds the labels and is NOT given to the judges')


if __name__ == '__main__':
    main()
