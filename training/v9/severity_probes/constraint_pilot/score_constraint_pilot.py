#!/usr/bin/env python3
"""Pilot-gate the `proposed` ordering constraints in ../score_probes.py.

Those 22 constraints are AUTHORED — written 2026-08-13 by one Claude session
reading v8's probe scores. pilot-gate-the-rubric says the authored gold is the
thing most likely to be wrong in the room: three blind families once made an
identical 10-row unanimous error, and reading their disagreement is what
caught it. So the constraints get the same treatment the v9 rubric got before
they are allowed to gate anything.

Verdict classes (same law as ../../pilot/score_pilot.py):

  unanimous-right            — every family agrees with the authored ordering;
                               promote it to gating
  unanimous-vs-authored      — every family disagrees the SAME way: the AUTHOR
                               is probably wrong, not the families. Fix or drop
                               the constraint.
  split                      — families disagree with each other: the design
                               working. Read the minority reasoning; this is a
                               genuine severity judgement Amy should settle.

A family answering "tie" agrees with a `>=` constraint (which permits equality)
and disagrees with a strict `>` constraint.

Run:  python3 score_constraint_pilot.py [raw-dir]

Committed with the numbers it produces, per commit-the-scorer.
"""
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
VALID = {'a', 'b', 'tie'}


def main() -> int:
    pairs = {}
    for line in (HERE / 'pairs.jsonl').read_text().splitlines():
        if line.strip():
            p = json.loads(line)
            pairs[p['id']] = p

    raw_dir = HERE / (sys.argv[1] if len(sys.argv) > 1 else 'raw')
    families = {}
    for f in sorted(raw_dir.glob('*.jsonl')):
        votes = {}
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            v = json.loads(line)
            if v['higher'] not in VALID:
                print(f'INVALID answer {v["higher"]!r} from {f.stem} on {v["id"]}')
                return 2
            votes[v['id']] = v
        missing = set(pairs) - set(votes)
        if missing:
            print(f'{f.stem}: MISSING ids {sorted(missing)} — refusing to score a partial family')
            return 2
        families[f.stem] = votes

    if len(families) < 3:
        print(f'only {len(families)} families in {raw_dir.name}/ — the gate needs 3 blind families')
        return 2

    def agrees(pair, ans):
        if ans == 'tie':
            return pair['allows_tie']
        return ans == pair['proposed_higher']

    classes = Counter()
    promote, fix, split = [], [], []

    for pid, pair in pairs.items():
        answers = {fam: votes[pid]['higher'] for fam, votes in families.items()}
        oks = {fam: agrees(pair, a) for fam, a in answers.items()}
        distinct = set(answers.values())

        if all(oks.values()):
            verdict = 'unanimous-right'
            promote.append(pid)
        elif not any(oks.values()) and len(distinct) == 1:
            verdict = 'unanimous-vs-authored'
            fix.append(pid)
        elif not any(oks.values()):
            verdict = 'unanimous-wrong-split'   # all disagree with author, differently
            fix.append(pid)
        else:
            verdict = 'split'
            split.append(pid)
        classes[verdict] += 1

        if verdict != 'unanimous-right':
            print(f'\n[{verdict}] {pid}   (authored: {pair["proposed_higher"]}'
                  f'{", tie ok" if pair["allows_tie"] else ""})')
            print(f'    A: {pair["a"]["cmd"][:70]!r}')
            print(f'    B: {pair["b"]["cmd"][:70]!r}')
            for fam, votes in families.items():
                mark = 'ok ' if oks[fam] else 'NO '
                print(f'    {mark}{fam:<10} -> {votes[pid]["higher"]:<4} {votes[pid]["why"][:88]}')

    print(f'\n{"="*72}')
    print(f'families: {", ".join(sorted(families))}')
    print(f'pairs: {len(pairs)}')
    for k, v in classes.most_common():
        print(f'  {k:<24} {v}')
    print(f'\nPROMOTE to gating ({len(promote)}): {", ".join(promote) or "none"}')
    print(f'FIX or DROP ({len(fix)}): {", ".join(fix) or "none"}')
    print(f'FOR AMY — genuine splits ({len(split)}): {", ".join(split) or "none"}')

    # A pilot never "fails"; it reports. Non-zero only on a broken round.
    return 0


if __name__ == '__main__':
    sys.exit(main())
