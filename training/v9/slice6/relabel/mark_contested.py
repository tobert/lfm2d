#!/usr/bin/env python3
"""Mark generator/relabeler disagreements as `contested`, without resolving them.

Amy, 2026-08-13: *"the extension splits seem fine to me, I would hesitate where
gen/relabel disagree."*

So: where a blind relabel disagreed with the generator, the row keeps its
ORIGINAL label and gains `contested: true`, with the dissent appended to
`note`. Nothing is silently flipped to the relabeler's view and nothing is
dropped. That is the schema's own meaning of the field, and it matches
measure-disagreement-dont-declare-it — a disagreement is signal to carry
forward, not a tie to break.

Contrast with rule 13, which Amy ruled on explicitly: THOSE rows were flipped,
because there was a ruling. Absent a ruling, hesitate.

    python3 mark_contested.py <stem> [--apply]

Dry-run by default. Idempotent: re-running never double-appends, so it is safe
after a later relabel round adds a family.

Committed with the numbers it produces, per commit-the-scorer.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
INCOMING = HERE.parent / 'incoming'
MARK = '  [CONTESTED]'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stem')
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()

    src = INCOMING / f'{args.stem}.jsonl'
    rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]

    raw_dir = HERE / 'raw' / args.stem
    if not raw_dir.is_dir():
        print(f'no raw dir {raw_dir}'); return 2
    votes = {}
    for f in sorted(raw_dir.glob('*.jsonl')):
        fam = f.stem.rsplit('_c', 1)[0]
        for line in f.read_text().splitlines():
            if line.strip():
                v = json.loads(line)
                votes.setdefault(v['id'], []).append((fam, v))

    changed = 0
    already = 0
    for i, r in enumerate(rows, 1):
        rid = f'r{i:03d}'
        dissent = [(fam, v) for fam, v in votes.get(rid, []) if v['label'] != r['label']]
        if not dissent:
            continue
        if MARK in r['note']:
            already += 1
            continue
        parts = '; '.join(f'{fam} said {v["label"]} — {v["why"]}' for fam, v in dissent)
        r['note'] = f'{r["note"]}{MARK} blind relabel disagreed: {parts} '\
                    f'Label held pending a ruling (Amy: hesitate where gen/relabel disagree).'
        r['contested'] = True
        changed += 1
        print(f'  {rid}  {r["label"]:<16} {r["text"][:56]!r}')
        for fam, v in dissent:
            print(f'        {fam} -> {v["label"]}')

    print(f'\n{args.stem}: {changed} newly contested, {already} already marked')
    if args.apply and changed:
        src.write_text('\n'.join(json.dumps(r, sort_keys=True) for r in rows) + '\n')
        print(f'applied -> {src}')
    elif changed:
        print('(dry run — pass --apply to write)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
