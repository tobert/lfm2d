#!/usr/bin/env python3
"""Apply Amy's rule-8 bias ruling.

Amy, 2026-08-13, asked what rule 8's scope was and then ruled:

  "on rule 8, I feel it should bias towards labeling data-critical or
   situation-normal when they're detected in the data and/or it's ambiguous"

Rule 8 says a destructive string in DATA position does not make a statement
destructive. The generators read the second sentence -- "a statement that
merely echoes or greps a dangerous-looking string observes it" -- as licensing
`informative` for any carrier with an inert payload. That is wrong in two
directions at once, and the blind relabel found both:

  * a carrier that creates its own durable artifact (`git commit`, a wiki
    POST) is at least `situation-normal` regardless of what its payload says
  * a POST carrying a destructive string is not something to resolve DOWNWARD
    when the persistence behaviour is unknown

Amy's ruling settles both with one direction: when a destructive command is
detected in the data, or the call is ambiguous, resolve UP (situation-normal
or data-critical), never down to informative.

Every affected row converges on `situation-normal`. That is not a coincidence:
all nine are carriers with a real durable effect AND an inert payload, which is
precisely the population the ruling describes.

    python3 apply_rule8_bias.py [--apply]

Dry-run by default. Idempotent. Rows enumerated explicitly.

Committed with the numbers it produces, per commit-the-scorer.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V9 = HERE.parent

CARRIER = ('Rule 8 bias (Amy 2026-08-13): the payload is inert, but the carrier itself creates a '
           'durable artifact — resolve UP to situation-normal, never down to informative.')
POST = ('Rule 8 bias (Amy 2026-08-13): destructive string detected in the data and the endpoint\'s '
        'persistence is unknown — ambiguity resolves UP to situation-normal, not down to informative.')

# (slice, stem, 1-based row, final label, note suffix)
DECISIONS = [
    ('slice2', 'history', 41, 'situation-normal', CARRIER),
    ('slice2', 'history', 57, 'situation-normal', CARRIER),
    ('slice3', 'datapos', 56, 'situation-normal', CARRIER),
    ('slice3', 'datapos', 1, 'situation-normal', POST),
    ('slice3', 'datapos', 3, 'situation-normal', POST),
    ('slice3', 'datapos', 14, 'situation-normal', POST),
    ('slice3', 'datapos', 15, 'situation-normal', POST),
    ('slice3', 'datapos', 23, 'situation-normal', POST),
    ('slice3', 'datapos', 25, 'situation-normal', POST),
]
MARK = '  [RULE 8 BIAS]'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()

    files = {}
    for sl, stem, _, _, _ in DECISIONS:
        key = (sl, stem)
        if key not in files:
            p = V9 / sl / 'incoming' / f'{stem}.jsonl'
            files[key] = (p, [json.loads(l) for l in p.read_text().splitlines() if l.strip()])

    flips = resolved = already = 0
    for sl, stem, idx, final, why in DECISIONS:
        _, rows = files[(sl, stem)]
        r = rows[idx - 1]
        if MARK in r['note']:
            already += 1
            continue
        was_label, was_contested = r['label'], r['contested']
        note = r['note'].split('  [CONTESTED]')[0].rstrip()
        r['note'] = f'{note}{MARK} {why}'
        r['label'] = final
        r['contested'] = False
        if was_label != final:
            flips += 1
        if was_contested:
            resolved += 1
        chg = f'{was_label} -> {final}' if was_label != final else f'{final} (held)'
        print(f'  {sl}/{stem}:r{idx:03d}  {chg}'
              f'{", un-contested" if was_contested else ""}')
        print(f'      {r["text"][:70]}')

    print(f'\n{flips} label flips, {resolved} contested resolved, {already} already applied')
    if args.apply and (flips or resolved):
        for p, rows in files.values():
            p.write_text('\n'.join(json.dumps(x, sort_keys=True) for x in rows) + '\n')
        print('applied')
    elif not already:
        print('(dry run — pass --apply to write)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
