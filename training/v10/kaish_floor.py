#!/usr/bin/env python3
"""How much of real agent bash does `kaish --plan` parse, and what does it
refuse? Written 2026-08-23 for the v10 input-contract decision.

AGGREGATES ONLY. Plans every distinct command the live head scored
(`kaish --plan-file -` analyzes without running), buckets the failures by
kaish's error token, and masks the context around the top two buckets
(letters->a, digits->0) so the shape is visible and the text is not.

Measured 2026-08-23, kaish 0.15.0, 11,000 distinct commands / 11,085 rows:
86.0% of rows plan; 43,265 simple commands with name/args/redirect-target
fields. Failure buckets: unquoted `echo === ... ===;` (adjacent-words, 621),
heredoc inside `$( )` (unterminated, 200), lexer chars, `=~`, `{`, bash
`for ... do`. Unplannable rows fire at 16.3% vs 10.7% for plannable.

    .venv-train/bin/python training/v10/kaish_floor.py --model-id kube_ordinal_v9_cal
"""
import argparse
import collections
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from soak_shapes import DEFAULT_LOG, load_rows, top, winner  # noqa: E402


def plan(cmd):
    try:
        r = subprocess.run(['kaish', '--plan-file', '-'], input=cmd, capture_output=True,
                           text=True, timeout=10)
    except subprocess.TimeoutExpired:
        return ('timeout', None)
    try:
        j = json.loads(r.stdout)
    except json.JSONDecodeError:
        return ('nojson', (r.stderr.strip() or r.stdout)[:80])
    if 'errors' in j:
        return ('error', j['errors'][0])
    return ('ok', j)


def fired(d):
    sc = [c['scores'] for c in d['lfm2d']['clauses']]
    return top(sc[winner(sc)]) == 'data-critical'


def mask(s):
    return re.sub(r'[A-Za-z_]', 'a', re.sub(r'\d', '0', s)).replace('\n', '⏎')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', type=Path, default=DEFAULT_LOG)
    ap.add_argument('--model-id', required=True)
    args = ap.parse_args()
    ver = subprocess.run(['kaish', '--version'], capture_output=True, text=True).stdout.strip()
    rows = load_rows(args.log, args.model_id)
    cmds = list(dict.fromkeys(d['command'] for d in rows))
    print(f'{ver}; distinct commands {len(cmds)} of {len(rows)} rows')
    with ThreadPoolExecutor(8) as ex:
        res = dict(zip(cmds, ex.map(plan, cmds)))
    print(collections.Counter(k for k, _ in res.values()))
    fails = [d for d in rows if res[d['command']][0] != 'ok']
    oks = [d for d in rows if res[d['command']][0] == 'ok']
    print(f'rows that fail to plan: {len(fails)}/{len(rows)} = {len(fails)/len(rows):.1%}')
    print(f'firing among unplannable {sum(map(fired, fails))}/{len(fails)} = '
          f'{sum(map(fired, fails))/max(1, len(fails)):.1%}; among plannable '
          f'{sum(map(fired, oks))}/{len(oks)} = {sum(map(fired, oks))/len(oks):.1%}')
    tok = collections.Counter()
    ctx = collections.defaultdict(collections.Counter)
    for cmd, (k, e) in res.items():
        if k != 'error':
            continue
        m = re.match(r"found '([^']*)'", e['message'])
        key = m.group(1) if m else e['message'][:40]
        tok[key] += 1
        s = e['start']
        ctx[key][mask(cmd[max(0, s - 6):s + 8])] += 1
    print('failure buckets:')
    for key, n in tok.most_common(12):
        print(f'  {n:5d}  {key!r}')
    for key, _ in tok.most_common(2):
        print(f'masked context, {key!r}:')
        for c, n in ctx[key].most_common(8):
            print(f'  {n:4d} {c!r}')
    plans = [j for k, j in res.values() if k == 'ok']
    simple = [c for j in plans for st in j['statements'] for c in st['plan']['commands']]
    print(f'simple commands from successful plans: {len(simple)}')
    print('top names:', collections.Counter(c['name'] for c in simple).most_common(15))
    print('redirect kinds:', collections.Counter(r['kind'] for c in simple for r in c['redirects']))


if __name__ == '__main__':
    main()
