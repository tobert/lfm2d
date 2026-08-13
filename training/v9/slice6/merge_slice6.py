#!/usr/bin/env python3
"""Merge slice-6 incoming/ into one deduped set, and record what overlapped.

Four agents worked adjacent surfaces, so some rows were generated more than
once. Those collisions are not waste — they are unplanned inter-annotator
agreement between independent families, and whether the labels AGREED is
worth more than the rows themselves. So this refuses to merge on a label
conflict rather than silently picking a winner.

Keep order is deterministic (sorted filename, then line) so re-running
reproduces the same file byte-for-byte.

    python3 merge_slice6.py            # write slice6.jsonl
    python3 merge_slice6.py --check    # verify the committed file regenerates

Committed with the numbers it produces, per commit-the-scorer.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
INCOMING = HERE / 'incoming'
OUT = HERE / 'slice6.jsonl'


def norm(t):
    return ' '.join(str(t).lower().split())


def build():
    idx = defaultdict(list)
    order = []
    for f in sorted(INCOMING.glob('*.jsonl')):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if not line.strip():
                continue
            r = json.loads(line)
            k = norm(r['text'])
            if k not in idx:
                order.append(k)
            idx[k].append((f.name, i, r))

    conflicts, dups = [], []
    rows = []
    for k in order:
        entries = idx[k]
        labels = {r['label'] for _, _, r in entries}
        if len(labels) > 1:
            conflicts.append((entries[0][2]['text'], sorted(labels),
                              [f'{fn}:{i}' for fn, i, _ in entries]))
            continue
        if len(entries) > 1:
            dups.append((entries[0][2]['text'], entries[0][2]['label'],
                         [f"{fn}:{i}({r['author']})" for fn, i, r in entries]))
        rows.append(entries[0][2])
    return rows, dups, conflicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()

    rows, dups, conflicts = build()

    if conflicts:
        print(f'REFUSING TO MERGE — {len(conflicts)} label conflict(s):')
        for text, labels, where in conflicts:
            print(f'  {text!r}')
            print(f'    labels {labels} at {where}')
        print('\nTwo families disagreed on the same text. That is a rubric '
              'question, not a merge question — settle it before merging.')
        return 1

    body = '\n'.join(json.dumps(r, sort_keys=True) for r in rows) + '\n'

    if args.check:
        if not OUT.exists():
            print('slice6.jsonl missing'); return 2
        if OUT.read_text() != body:
            print('DRIFT: slice6.jsonl does not match incoming/'); return 1
        print(f'ok — slice6.jsonl regenerates byte-identical ({len(rows)} rows)')
        return 0

    OUT.write_text(body)
    labels = Counter(r['label'] for r in rows)
    authors = Counter(r['author'] for r in rows)
    print(f'wrote {OUT}')
    print(f'  rows: {len(rows)}  (deduped {len(dups)} cross-file repeats)')
    print(f'  labels: {dict(sorted(labels.items()))}')
    for lab, c in sorted(labels.items()):
        print(f'     {lab:<18} {c:4d}  {c/len(rows):.1%}')
    print(f'  contested: {sum(r["contested"] for r in rows)}')
    print(f'  authors: {dict(authors)}')
    print(f'\ncross-file agreement — {len(dups)} texts generated independently '
          f'by more than one family, {len(conflicts)} disagreed:')
    for text, label, where in dups:
        print(f'  [{label}] {text!r}')
        print(f'      {", ".join(where)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
