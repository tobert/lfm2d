#!/usr/bin/env python3
"""Tests for build_v10.py's pure half.

Run:  python3 training/v10/test_build_v10.py
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_v10 import GateError, merge_sources, pick_shape_holdout, skew_gate, stratified_split  # noqa: E402

FAILURES = []


def check(label, got, expected):
    ok = got == expected
    print(f'{"ok " if ok else "FAIL"}  {label}')
    if not ok:
        print(f'      got:      {got!r}')
        print(f'      expected: {expected!r}')
        FAILURES.append(label)


def row(text, label, shape=None):
    r = {'text': text, 'label': label}
    if shape:
        r['shape'] = shape
    return r


def main():
    v9 = [row('rm -rf /data', 'data-critical'), row('ls  -la', 'informative'), row('git push --force', 'data-critical')]
    live = [row('ls -la', 'informative'), row('cat x', 'informative'), row('git push  --force', 'data-critical')]
    merged, held, conflicts, dups, _ = merge_sources({'v9': v9, 'live': live}, quarantine={'git push --force'})
    check('agreeing duplicate kept once (whitespace-normalized)', [r['text'] for r in merged], ['rm -rf /data', 'ls  -la', 'cat x'])
    check('first source wins provenance', merged[1]['source'], 'v9')
    check('dups reported', [d['where'] for d in dups], [['v9[1]', 'live[0]'], ['v9[2]', 'live[2]']])
    check('probe text held out of training', [r['text'] for r in held], ['git push --force'])
    check('no conflicts', conflicts, [])

    _, _, conflicts, _, _ = merge_sources({'v9': [row('rm x', 'data-critical')], 'live': [row('rm x', 'situation-normal')]})
    check('label conflict is reported, not resolved', [(c['labels'], c['where']) for c in conflicts],
          [(['data-critical', 'situation-normal'], ['v9[0]', 'live[0]'])])

    # instance grain beats shape grain on identical text -- and says so
    merged, _, conflicts, _, resolved = merge_sources(
        {'v9': [row('git worktree list', 'informative')], 'live': [row('git worktree list', 'situation-normal', 'git worktree')]},
        grain={'v9': 'instance', 'live': 'shape'})
    check('instance label wins', (merged[0]['label'], merged[0]['source']), ('informative', 'v9'))
    check('resolution reported', [(r['label'], r['over']) for r in resolved], [('informative', ['informative', 'situation-normal'])])
    check('no conflict left', conflicts, [])
    _, _, conflicts, _, resolved = merge_sources(
        {'a': [row('x', 'informative')], 'b': [row('x', 'data-critical')]}, grain={'a': 'instance', 'b': 'instance'})
    check('two instance sources disagreeing is still a conflict', (len(conflicts), resolved), (1, []))

    try:
        merge_sources({'x': [row('a', 'benign')]})
        print('FAIL  invalid label is loud'); FAILURES.append('invalid label')
    except GateError:
        print('ok   invalid label is loud')

    # -- shape holdout: only low-rank shapes, seeded, whole shapes move together
    rows = [row(f't{i}', 'informative', shape=f's{i % 6}') for i in range(30)]
    ranks = {'s0': 1, 's1': 2, 's2': 250, 's3': 300, 's4': 350, 's5': 400}
    hold, keep, shapes = pick_shape_holdout(rows, ranks, 2, 200, seed=1)
    check('holdout draws from rank >= min only', all(ranks[s] >= 200 for s in shapes), True)
    check('holdout has k shapes', len(shapes), 2)
    check('whole shapes move', Counter(r['shape'] for r in hold), Counter({s: 5 for s in shapes}))
    check('holdout + keep = rows', len(hold) + len(keep), 30)
    check('seeded and stable', pick_shape_holdout(rows, ranks, 2, 200, seed=1)[2], shapes)

    # -- split: stratified by label, seeded
    rows = [row(f'i{i}', 'informative') for i in range(60)] + [row(f's{i}', 'situation-normal') for i in range(30)] \
        + [row(f'd{i}', 'data-critical') for i in range(10)]
    train, val = stratified_split(rows, 0.2, seed=7)
    check('val per label', Counter(r['label'] for r in val), Counter({'informative': 12, 'situation-normal': 6, 'data-critical': 2}))
    check('train + val = rows', len(train) + len(val), 100)
    check('disjoint', set(r['text'] for r in train) & set(r['text'] for r in val), set())
    check('seeded and stable', stratified_split(rows, 0.2, seed=7)[1][0], val[0])

    # -- skew gate
    try:
        skew_gate([row('a', 'informative')] * 8 + [row('b', 'data-critical')] * 2, 'x')
        print('FAIL  skew gate fires'); FAILURES.append('skew gate')
    except GateError:
        print('ok   skew gate fires')
    skew_gate([row('a', 'informative')] * 7 + [row('b', 'data-critical')] * 3, 'x')
    print('ok   skew gate passes at 70%')

    print()
    if FAILURES:
        print(f'FAILED: {len(FAILURES)} case(s): {FAILURES}')
        return 1
    print('ALL OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
