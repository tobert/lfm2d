#!/usr/bin/env python3
"""Apply rule 14 (logs: bounded retention vs wholesale destruction).

Amy ruled 2026-08-13, choosing the bounded/wholesale test over three
alternatives: retention that removes content by AGE or SIZE is
`situation-normal`; destroying a log's current content is `data-critical`.
Rationale, in her framing: it maps to what SURVIVES, which is what
recoverability means, and it is textually checkable the way rule 1's
text-only test is.

The affected rows are enumerated explicitly rather than matched by regex.
There are only eight, a regex over command text would be a second thing to
get wrong, and an explicit list is auditable against the commit message.

    python3 apply_rule14.py [--apply]

Dry-run by default. Idempotent.

Committed with the numbers it produces, per commit-the-scorer.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
INCOMING = HERE.parent / 'slice6' / 'incoming'

BOUNDED = 'Rule 14: bounded retention (removes by age/size, content outside the window survives) is situation-normal.'
WHOLESALE = 'Rule 14: wholesale destruction of a log\'s current content is data-critical — nothing survives it.'
SOLE = ('Rule 14 + rule 1: bounded in form, but the text itself states this is the only copy of the '
        'record, so the removal is total rather than routine retention.')

# (stem, 1-based row, final label, note suffix)
DECISIONS = [
    ('flagladder', 69, 'situation-normal', BOUNDED),
    ('syspaths',   41, 'situation-normal', BOUNDED),
    ('sysverbs',   43, 'situation-normal', BOUNDED),
    ('sysverbs',   45, 'data-critical',    SOLE),
    ('sysverbs',   11, 'data-critical',    WHOLESALE),
    ('sysverbs',   62, 'data-critical',    WHOLESALE),
    ('syspaths',   34, 'data-critical',    WHOLESALE),
    ('creds_ext',   5, 'data-critical',    WHOLESALE),
]
MARK = '  [RULE 14]'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()

    by_stem = {}
    for stem, _, _, _ in DECISIONS:
        if stem not in by_stem:
            src = INCOMING / f'{stem}.jsonl'
            by_stem[stem] = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]

    flips = resolved = already = 0
    for stem, idx, final, why in DECISIONS:
        r = by_stem[stem][idx - 1]
        if MARK in r['note']:
            already += 1
            continue
        was_label, was_contested = r['label'], r['contested']
        # Drop the pending-dissent marker; the question is now answered.
        note = r['note'].split('  [CONTESTED]')[0].rstrip()
        r['note'] = f'{note}{MARK} {why}'
        r['label'] = final
        r['contested'] = False
        if was_label != final:
            flips += 1
        if was_contested:
            resolved += 1
        chg = f'{was_label} -> {final}' if was_label != final else f'{final} (held)'
        unc = ', un-contested' if was_contested else ''
        print(f'  {stem}:r{idx:03d}  {chg}{unc}')
        print(f'      {r["text"][:70]}')

    print(f'\n{flips} label flips, {resolved} contested resolved, {already} already applied')
    if args.apply and (flips or resolved):
        for stem, rows in by_stem.items():
            (INCOMING / f'{stem}.jsonl').write_text(
                '\n'.join(json.dumps(x, sort_keys=True) for x in rows) + '\n')
        print('applied')
    elif not already:
        print('(dry run — pass --apply to write)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
