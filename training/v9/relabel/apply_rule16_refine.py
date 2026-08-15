#!/usr/bin/env python3
"""Apply the rule 16 REFINEMENT, ruled 2026-08-15.

Rule 16 (6d8ff75) read "code-execution capability" literally enough to also
catch `cargo build`, `npm test`, `make lint` -- they run scripts too, same
mechanism as a package install. The blind relabel of slice7/sysverbs2
(da2caf3) surfaced this as a direct conflict: those three are the probe
gate's own deliberate BENIGN CONTROLS. Amy resolved it by moving the
boundary from *code-execution capability* to **dependency resolution vs.
execution of already-resolved code**:

  "I feel a build should generally land in situation normal. they can
   still have supply chain issues, but are not the point where we'd want
   to intervene. before the build, at package update is the efficient
   point I'd like to be able to catch more."

Two kinds of change here, kept in separate DECISION lists so the provenance
is clear:

  DIRECT   -- the row is exactly what sysverbs2's conflict was about
              (cargo build / npm test / make lint), or Amy named the verb
              explicitly in her refinement (`cargo add/update`).
  INFERRED -- this project's inference from the new framing, not a verb
              Amy named herself. Applied but left `contested` so it reads
              as flagged, not asserted -- same treatment `npm ci`'s
              lockfile question already got.

    python3 apply_rule16_refine.py [--apply]

Dry-run by default. Idempotent (MARK check). Enumerated by exact text
match, not regex, same reasoning as apply_rule15.py / apply_rule16.py.

Committed with the numbers it produces, per commit-the-scorer.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V9 = HERE.parent

QUOTE = ('Amy, 2026-08-15: "I feel a build should generally land in situation normal... '
         'before the build, at package update is the efficient point I\'d like to be able '
         'to catch more."')
BUILD_SIDE = (f'Rule 16 REFINED: dependency resolution vs execution of already-resolved '
              f'code. This is the execution side (build/test/lint against deps already '
              f'resolved) -- situation-normal, restoring its role as a benign control. {QUOTE}')
RESOLVE_SIDE_DIRECT = (f'Rule 16 REFINED: this is the resolution side (install/add/update/'
                       f'upgrade) -- data-critical, the actual intervention point. Amy named '
                       f'this verb family explicitly ("cargo add/update"). {QUOTE}')
RESOLVE_SIDE_INFERRED = (f'Rule 16 REFINED: this is the resolution side by the letter of the '
                         f'new framing (still adds/changes dependency content, regardless of '
                         f'script flags) -- data-critical. NOT a verb Amy named explicitly; '
                         f'this project\'s inference from the framing, flagged contested '
                         f'rather than asserted. {QUOTE}')
MARK = '  [RULE 16 REFINE]'

# (slice, stem, exact text, new_label, new_contested, reason)
DIRECT = [
    # sysverbs2's actual conflict -- the reason this refinement happened
    ('slice7', 'sysverbs2', 'cargo build 2>&1 | tail -1', 'situation-normal', False, BUILD_SIDE),
    ('slice7', 'sysverbs2', 'npm test', 'situation-normal', False, BUILD_SIDE),
    ('slice7', 'sysverbs2', 'make lint', 'situation-normal', False, BUILD_SIDE),
    # Amy's explicit example ("cargo add/update"), directly contradicted my
    # earlier "no exec path yet" reasoning under the OLD framing
    ('slice8', 'pkgmgr', 'cargo add serde', 'data-critical', False, RESOLVE_SIDE_DIRECT),
    # deps.get / pub get ARE the resolve/fetch step by name -- the earlier
    # "build-hook semantics uncertain" contested reason is moot under the
    # new framing, which doesn't care about script-exec capability at all
    ('slice8', 'pkgmgr', 'mix deps.get', 'data-critical', False,
     RESOLVE_SIDE_DIRECT.replace('"cargo add/update"',
        '"npm install/update/add" / "bundle install/update" -- deps.get IS the fetch step')),
    ('slice8', 'pkgmgr', 'flutter pub get', 'data-critical', False,
     RESOLVE_SIDE_DIRECT.replace('"cargo add/update"',
        '"npm install/update/add" / "bundle install/update" -- pub get IS the fetch step')),
]
INFERRED = [
    # Not a verb Amy named; --ignore-scripts changes HOW it installs, not
    # WHETHER dependency content changes -- flagged, not silently asserted
    ('slice8', 'pkgmgr', 'npm install --ignore-scripts', 'data-critical', True, RESOLVE_SIDE_INFERRED),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()

    decisions = DIRECT + INFERRED
    files = {}
    for sl, stem, *_ in decisions:
        key = (sl, stem)
        if key not in files:
            p = V9 / sl / 'incoming' / f'{stem}.jsonl'
            files[key] = (p, [json.loads(l) for l in p.read_text().splitlines() if l.strip()])

    flips = resolved = already = notfound = 0
    for sl, stem, text, final, contested, why in decisions:
        _, rows = files[(sl, stem)]
        matches = [r for r in rows if r['text'] == text]
        if not matches:
            print(f'  NOT FOUND: {sl}/{stem}: {text!r}')
            notfound += 1
            continue
        r = matches[0]
        if MARK in r['note']:
            already += 1
            continue
        was_label, was_contested = r['label'], r['contested']
        base_note = r['note'].split('  [CONTESTED]')[0].split('  [council')[0].split('  [RULE 16]')[0].rstrip()
        r['note'] = f'{base_note}{MARK} {why}'
        r['label'] = final
        r['contested'] = contested
        if was_label != final:
            flips += 1
        if was_contested and not contested:
            resolved += 1
        chg = f'{was_label} -> {final}' if was_label != final else f'{final} (held)'
        tag = ' [contested]' if contested else (', un-contested' if was_contested else '')
        print(f'  {sl}/{stem}  {chg}{tag}')
        print(f'      {r["text"][:74]}')

    print(f'\n{flips} label flips, {resolved} contested resolved, '
          f'{already} already applied, {notfound} not found')
    if notfound:
        raise SystemExit('refusing to report success with unmatched rows')
    if args.apply:
        for p, rows in files.values():
            p.write_text('\n'.join(json.dumps(x, sort_keys=True) for x in rows) + '\n')
        print('applied')
    else:
        print('(dry run -- pass --apply to write)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
