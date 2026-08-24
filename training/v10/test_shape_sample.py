#!/usr/bin/env python3
"""Tests for shape_sample.py's pure half (v10 slice 3).

Run:  python3 training/v10/test_shape_sample.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shape_sample import rank_shapes  # noqa: E402

FAILURES = []


def check(label, got, expected):
    ok = got == expected
    print(f'{"ok " if ok else "FAIL"}  {label}')
    if not ok:
        print(f'      got:      {got!r}')
        print(f'      expected: {expected!r}')
        FAILURES.append(label)


def main():
    ranked = rank_shapes([
        ['sed -n 5p a.txt', 'echo done'],
        ['sed -n 9p b.txt'],
        ['sed -n 12,14p c.txt', 'sed -n 1p d.txt'],
    ])
    # all four sed -n clauses share one shape and outrank the echo
    check('ranked by clause count', [(s.split()[0], n) for s, n, _ in ranked][0][1], 4)
    check('echo is second', ranked[1][1], 1)

    sed_examples = ranked[0][2]
    check('examples capped at 3', len(sed_examples), 3)
    check('examples are real clause texts', sed_examples[0], 'sed -n 5p a.txt')

    # identical clause text doesn't duplicate an example
    ranked = rank_shapes([['ls -la'], ['ls -la'], ['ls -la']])
    check('duplicate text counted, not re-exampled', (ranked[0][1], len(ranked[0][2])), (3, 1))

    check('empty input', rank_shapes([]), [])

    print()
    if FAILURES:
        print(f'FAILED: {len(FAILURES)} case(s): {FAILURES}')
        return 1
    print('ALL OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
