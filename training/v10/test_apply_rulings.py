#!/usr/bin/env python3
"""Tests for apply_rulings.apply (v10 slice 3).

Run:  python3 training/v10/test_apply_rulings.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from apply_rulings import RulingError, apply  # noqa: E402

FAILURES = []


def check(label, got, expected):
    ok = got == expected
    print(f'{"ok " if ok else "FAIL"}  {label}')
    if not ok:
        print(f'      got:      {got!r}')
        print(f'      expected: {expected!r}')
        FAILURES.append(label)


def rec(shape, consensus):
    return {'shape': shape, 'consensus': consensus, 'kind': 'unanimous' if consensus else 'split'}


def main():
    records = [rec('a', 'informative'), rec('b', 'data-critical'), rec('c', None), rec('d', None)]
    rulings = [{'shape': 'b', 'label': 'informative', 'by': 'Amy'}, {'shape': 'c', 'label': 'situation-normal', 'by': 'Amy'}]
    out, unresolved = apply(records, rulings)
    by = {r['shape']: r for r in out}
    check('consensus becomes the label', by['a']['label'], 'informative')
    check('ruling overrides consensus', by['b']['label'], 'informative')
    check('ruling resolves a split', by['c']['label'], 'situation-normal')
    check('ruling record kept without the shape key', by['b']['ruling'], {'label': 'informative', 'by': 'Amy'})
    check('unruled shapes carry no ruling', 'ruling' in by['a'], False)
    check('unresolved splits are reported', unresolved, ['d'])

    # idempotent: re-applying a smaller ruling set drops the stale ruling
    out2, _ = apply(out, [{'shape': 'c', 'label': 'situation-normal'}])
    by2 = {r['shape']: r for r in out2}
    check('re-apply drops stale rulings', ('ruling' in by2['b'], by2['b']['label']), (False, 'data-critical'))

    try:
        apply(records, [{'shape': 'zz', 'label': 'informative'}])
        print('FAIL  unknown shape is loud'); FAILURES.append('unknown shape')
    except RulingError:
        print('ok   unknown shape is loud')

    print()
    if FAILURES:
        print(f'FAILED: {len(FAILURES)} case(s): {FAILURES}')
        return 1
    print('ALL OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
