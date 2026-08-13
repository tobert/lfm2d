#!/usr/bin/env python3
"""Generate the constraint pilot's pairs.jsonl FROM score_probes.py.

Single source of truth: the constraints live in the scorer, and this emits
the blind-labelable form of the `proposed` ones. Regenerating must produce a
byte-identical file (v6-features-regenerate-exactly) — if it does not, the
scorer and the pilot have drifted and the pilot's verdicts no longer apply
to the gate they were meant to review.

    python3 make_pairs.py            # write pairs.jsonl
    python3 make_pairs.py --check    # verify committed file still regenerates

The AUTHORED answer (`proposed_higher`) is written into pairs.jsonl for the
scorer's use and MUST NOT be shown to the labeling families — see prompt.txt,
which is built from this file with that field stripped.
"""
import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROBE_DIR = HERE.parent

# Provenance values that mean "authored severity judgement" — the population
# this pilot reviews. Promotion to gating changes the value but not membership.
AUTHORED = {'proposed', 'pilot:3f'}

spec = importlib.util.spec_from_file_location('sp', PROBE_DIR / 'score_probes.py')
sp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sp)


def build():
    """Emit one row per `proposed` constraint.

    Presentation order is a SEEDED coin flip per pair. Without it the authored
    answer would sit on side B in all 22 rows (the constraints are all written
    "higher > lower"), and any family with a position bias would score 100%
    without reading a single command. The seed is fixed so the pilot is
    reproducible and so re-running does not silently re-randomize a round that
    families have already labeled.
    """
    probes = sp.load_probes()
    rng = random.Random(20260813)
    rows = []
    for name, left, op, right, prov, why in sp.CONSTRAINTS:
        # Every AUTHORED constraint belongs to this pilot round, whether or not
        # the round later promoted it to gating. Selecting on `proposed` alone
        # would make pairs.jsonl shrink the moment constraints are promoted,
        # and --check would then report drift against the round the families
        # actually labeled.
        if prov not in AUTHORED:
            continue
        # Normalize to "which side should score HIGHER".
        # op '<' means dc(left) < dc(right), i.e. RIGHT is the higher one.
        higher, lower = (right, left) if op == '<' else (left, right)
        flip = rng.random() < 0.5
        a_id, b_id = (higher, lower) if flip else (lower, higher)
        rows.append({
            'id': name,
            'op': op,
            'a': {'id': a_id, 'cmd': probes[a_id]['cmd']},
            'b': {'id': b_id, 'cmd': probes[b_id]['cmd']},
            # authored, provisional, NOT shown to families
            'proposed_higher': 'a' if flip else 'b',
            'allows_tie': op == '>=',
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()

    body = '\n'.join(json.dumps(r, sort_keys=True) for r in build()) + '\n'
    out = HERE / 'pairs.jsonl'

    if args.check:
        if not out.exists():
            print('pairs.jsonl missing'); return 2
        if out.read_text() != body:
            print('DRIFT: pairs.jsonl does not match score_probes.py CONSTRAINTS')
            return 1
        print(f'ok — pairs.jsonl regenerates byte-identical ({len(build())} pairs)')
        return 0

    out.write_text(body)
    print(f'wrote {out} ({len(build())} pairs)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
