#!/usr/bin/env python3
"""Score a blind relabel pass against the generator's proposed labels.

flagladder.jsonl was authored entirely by one family (sonnet), because the
local-model trial failed and the slice carries the subtlest rule (rule 4:
which interlocks count). Single-author data on the hardest rule is exactly
what a second family should see.

This is NOT a "who was right" scorer. A disagreement is a candidate rubric
question, and the generator is not privileged: measure-disagreement-dont-
declare-it. Rows are reported so a human reads the reasoning, not so a
majority overwrites a label.

    python3 score_relabel.py flagladder

Committed with the numbers it produces, per commit-the-scorer.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
INCOMING = HERE.parent / 'incoming'
RANK = {'informative': 0, 'situation-normal': 1, 'data-critical': 2}


def main():
    stem = sys.argv[1] if len(sys.argv) > 1 else 'flagladder'
    rows = [json.loads(l) for l in (INCOMING / f'{stem}.jsonl').read_text().splitlines()
            if l.strip()]
    by_id = {f'r{i:03d}': r for i, r in enumerate(rows, 1)}

    families = defaultdict(dict)
    for f in sorted((HERE / 'raw').glob('*.jsonl')):
        fam = f.stem.rsplit('_c', 1)[0]
        for line in f.read_text().splitlines():
            if line.strip():
                v = json.loads(line)
                families[fam][v['id']] = v

    for fam, votes in families.items():
        missing = set(by_id) - set(votes)
        if missing:
            print(f'{fam}: MISSING {len(missing)} ids '
                  f'({sorted(missing)[:5]}...) — refusing to score a partial pass')
            return 2

    print(f'rows: {len(by_id)}   families: {", ".join(sorted(families))}\n')

    for fam, votes in sorted(families.items()):
        agree = 0
        conf = Counter()
        harsher, softer = [], []
        for rid, gen in by_id.items():
            got = votes[rid]['label']
            if got == gen['label']:
                agree += 1
            else:
                conf[(gen['label'], got)] += 1
                (harsher if RANK[got] > RANK[gen['label']] else softer).append(rid)

        n = len(by_id)
        print(f'=== {fam} vs generator ({stem}) ===')
        print(f'  agreement: {agree}/{n} = {agree/n:.1%}')
        print(f'  relabeler HARSHER on {len(harsher)}, SOFTER on {len(softer)}')
        print('  confusion (generator -> relabeler):')
        for (a, b), c in conf.most_common():
            print(f'    {a:<16} -> {b:<16} {c}')

        print('\n  disagreements (read these; they are rubric questions):')
        for rid in sorted(set(harsher) | set(softer)):
            gen, got = by_id[rid], votes[rid]
            arrow = 'HARSHER' if RANK[got['label']] > RANK[gen['label']] else 'SOFTER '
            print(f'\n   [{arrow}] {rid}  {gen["text"][:66]!r}')
            print(f'      generator  {gen["label"]:<16} {gen["note"][:74]}')
            print(f'      relabeler  {got["label"]:<16} {got["why"][:74]}')
        print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
