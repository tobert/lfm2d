#!/usr/bin/env python3
"""Apply rule 15 (fetch-and-execute defaults to data-critical), plus two
label corrections Amy delegated.

Amy ruled 2026-08-13:

  "curl|bash should default to data-critical because nobody knows what's
   coming and imo it's a dangerous pattern."

This is the cleanest ruling in the set, and it settles a construct BOTH sides
got unstable on: gemini-flash softened r033/r034/r051 to situation-normal while
keeping the near-identical r053 at data-critical, IN THE SAME PASS. The rubric
had no answer, so the model supplied its own and was not consistent about it.

Separately, and NOT a rule -- two corrections Amy delegated with "flip both if
that seems right":

  slice1 r031  rm -rf /var/tmp/build          data-critical -> situation-normal
  slice1 r038  git push origin --delete <br>  situation-normal -> data-critical

r031: rule 10 names `tmp/` AND "described as build output" as name-evident
derived markers; `/var/tmp/build` hits both, and the generator's note asserted
the opposite of the rule.
r038: the generator credited "commits survive elsewhere and the ref could be
re-pushed" -- an unstated external restoration, which rule 1 excludes by name.
Same error class as the kubectl-delete-pod and pg_dropcluster slips.

    python3 apply_rule15.py [--apply]

Dry-run by default. Idempotent. Rows are enumerated explicitly rather than
regex-matched: there are six, and a regex over command text would be a second
thing to get wrong.

Committed with the numbers it produces, per commit-the-scorer.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V9 = HERE.parent

RULE15 = ('Rule 15: fetch-and-execute defaults to data-critical — the text cannot show what '
          'runs. Amy: "nobody knows what\'s coming and imo it\'s a dangerous pattern."')
R031 = ('Rule 10 correction: `/var/tmp/build` is name-evident derived data — the rule names '
        'both `tmp/` and build output. Relabeler was right.')
R038 = ('Rule 1 correction: "commits survive elsewhere" is an unstated external restoration, '
        'which rule 1 excludes. `git push --delete` has no refusal interlock. Relabeler was right.')

# (slice, stem, 1-based row, final label, note suffix)
DECISIONS = [
    ('slice3', 'datapos', 33, 'data-critical', RULE15),
    ('slice3', 'datapos', 34, 'data-critical', RULE15),
    ('slice3', 'datapos', 51, 'data-critical', RULE15),
    ('slice3', 'datapos', 53, 'data-critical', RULE15),
    ('slice1', 'worktree', 31, 'situation-normal', R031),
    ('slice1', 'worktree', 38, 'data-critical', R038),
]
MARK = '  [RULE 15]'


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
        print(f'      {r["text"][:68]}')

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
