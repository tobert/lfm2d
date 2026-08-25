#!/usr/bin/env python3
"""Tests for bulk_label.py's pure half (v10 slice 3).

Run:  python3 training/v10/test_bulk_label.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bulk_label import (ManifestError, VoteError, chunk_rows, chunk_text, collect_votes, merge,  # noqa: E402
                        parse_votes, plan_chunks, summarize, update_manifest)

FAILURES = []


def check(label, got, expected):
    ok = got == expected
    print(f'{"ok " if ok else "FAIL"}  {label}')
    if not ok:
        print(f'      got:      {got!r}')
        print(f'      expected: {expected!r}')
        FAILURES.append(label)


def raises(label, fn, needle):
    try:
        fn()
    except VoteError as e:
        ok = needle in str(e)
        print(f'{"ok " if ok else "FAIL"}  {label}')
        if not ok:
            print(f'      error text: {e}')
            FAILURES.append(label)
        return
    print(f'FAIL  {label}')
    print('      no VoteError raised')
    FAILURES.append(label)


def row(rank, shape, clauses=10, share=0.01):
    return {'rank': rank, 'shape': shape, 'clauses': clauses, 'share': share,
            'examples': [f'{shape} /home/x/secret-{rank}']}


def main():
    rows = [row(i, f's{i}') for i in range(1, 8)]

    # -- chunking
    chunks = chunk_rows(rows, 3)
    check('chunk sizes', [len(c) for c in chunks], [3, 3, 1])
    check('chunks preserve order', [c[0]['rank'] for c in chunks], [1, 4, 7])
    check('empty input', chunk_rows([], 3), [])

    # -- a second round labels only the delta and continues the numbering
    planned = plan_chunks(rows, 2, exclude=['s1', 's2', 's5'], start=4)
    check('delta chunks skip labeled shapes', [[r['shape'] for r in c] for _, c in planned], [['s3', 's4'], ['s6', 's7']])
    check('delta chunks continue numbering', [n for n, _ in planned], ['chunk_04', 'chunk_05'])
    only = plan_chunks(rows, 10, only=['s2', 's6'], start=12)
    check('re-vote round takes only the named shapes', [[r['shape'] for r in c] for _, c in only], [['s2', 's6']])
    check('re-vote round numbering', [n for n, _ in only], ['chunk_12'])
    m = update_manifest({'chunk_01': ['s1', 's2']}, planned)
    check('manifest accumulates', sorted(m), ['chunk_01', 'chunk_04', 'chunk_05'])
    check('manifest is idempotent for identical chunks', update_manifest(m, planned), m)
    try:
        update_manifest(m, [('chunk_04', [row(9, 's9')])])
        print('FAIL  manifest refuses a renumbered chunk'); FAILURES.append('manifest conflict')
    except ManifestError:
        print('ok   manifest refuses a renumbered chunk')

    text = chunk_text(rows[:2])
    lines = [json.loads(l) for l in text.splitlines()]
    check('chunk text is the prompt\'s input schema', set(lines[0]), {'shape', 'clauses', 'examples'})
    check('chunk text keeps real examples', lines[1]['examples'], ['s2 /home/x/secret-2'])

    # -- vote parsing: the happy path tolerates fences and blank lines
    reply = '```json\n{"shape": "s1", "label": "informative"}\n\n{"shape": "s2", "label": "mixed"}\n```\n'
    check('parse tolerates fences', parse_votes(reply, ['s1', 's2']), {'s1': 'informative', 's2': 'mixed'})

    # -- and is loud on everything off-spec
    raises('unknown shape', lambda: parse_votes('{"shape": "zz", "label": "informative"}', ['s1']), 'unexpected shape')
    raises('bad label', lambda: parse_votes('{"shape": "s1", "label": "benign"}', ['s1']), 'bad label')
    raises('duplicate vote', lambda: parse_votes(
        '{"shape": "s1", "label": "informative"}\n{"shape": "s1", "label": "data-critical"}', ['s1']), 'duplicate')
    raises('missing vote', lambda: parse_votes('{"shape": "s1", "label": "informative"}', ['s1', 's2']), 'unvoted')
    raises('non-JSON line', lambda: parse_votes('s1: informative', ['s1']), 'not JSON')
    raises('extra keys', lambda: parse_votes('{"shape": "s1", "label": "informative", "why": "x"}', ['s1']), 'expected')

    # -- merging
    gold = {'s1': 'informative', 's2': 'data-critical'}
    fam = {
        'A': {'s1': 'informative', 's2': 'situation-normal', 's3': 'informative', 's4': 'mixed', 's5': 'informative'},
        'B': {'s1': 'informative', 's2': 'data-critical', 's3': 'situation-normal', 's4': 'mixed', 's5': 'informative'},
        'C': {'s1': 'informative', 's2': 'data-critical', 's3': 'data-critical', 's4': 'mixed'},
    }
    recs = {r['shape']: r for r in merge(rows[:6], fam, gold)}
    check('unanimous', (recs['s1']['kind'], recs['s1']['consensus']), ('unanimous', 'informative'))
    check('2-1 majority', (recs['s2']['kind'], recs['s2']['consensus']), ('majority', 'data-critical'))
    check('3-way split has no consensus', (recs['s3']['kind'], recs['s3']['consensus']), ('split', None))
    check('escape flagged even when unanimous', (recs['s4']['kind'], recs['s4']['escape']), ('unanimous', True))
    check('two voters agreeing is unanimous', (recs['s5']['kind'], len(recs['s5']['votes'])), ('unanimous', 2))
    check('no voters is incomplete', recs['s6']['kind'], 'incomplete')
    check('gold hits per family', recs['s2']['gold_hits'], {'A': False, 'B': True, 'C': True})
    check('no gold field off-gold', 'gold' in recs['s3'], False)
    # examples never leave the sample: the merged record is publishable
    check('records carry no example text', any('examples' in r for r in recs.values()), False)

    # -- re-votes: a later chunk wins regardless of manifest key order
    manifest = {'chunk_12': ['s1'], 'chunk_01': ['s1', 's2']}  # deliberately reversed
    chunk_votes = {'chunk_01': {'s1': 'data-critical', 's2': 'informative'}, 'chunk_12': {'s1': 'situation-normal'}}
    votes, orphans, n = collect_votes(manifest, chunk_votes, in_sample={'s1'})
    check('later chunk supersedes despite dict order', votes, {'s1': 'situation-normal'})
    check('re-vote counted', n, 1)
    check('shapes gone from the sample are orphans', orphans, ['s2'])
    check('missing chunk file is tolerated', collect_votes({'chunk_01': ['s1']}, {}, {'s1'}), ({}, [], 0))

    s = summarize(list(recs.values()))
    check('summary kinds', s['kinds'], {'unanimous': 3, 'majority': 1, 'split': 1, 'incomplete': 1})
    check('summary canary', s['gold_canary'], {'A': '1/2', 'B': '2/2', 'C': '2/2'})
    check('canary denominator is every gold, not golds voted on',
          summarize(list(recs.values()), families=['A', 'D'])['gold_canary'], {'A': '1/2', 'D': '0/2'})
    check('summary escape count', s['escape_votes'], 1)
    check('share buckets by consensus', round(s['share_by_consensus']['informative'], 5), 0.02)

    print()
    if FAILURES:
        print(f'FAILED: {len(FAILURES)} case(s): {FAILURES}')
        return 1
    print('ALL OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
