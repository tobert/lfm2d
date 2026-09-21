#!/usr/bin/env python3
"""Compare two verdict_eval runs that differ only in the facts they sent.

    python3 benchmarks/lfm25/prompts/facts_ab.py RUN_A RUN_B

Built for F7 (2026-09-21): same binary, same prompt, same data, the user turn
rebuilt by two versions of build_facts. The comparison carries its own control.
A row whose INPUT bytes are identical in both runs was answered on the same warm
schedule by a greedy decoder, so its output must be identical too; any that is
not says the runs differ by more than the facts, and the rest of the table is
then not about the facts. That count is printed first and a non-zero one exits 1.

Aggregates only (repo convention): counts by gold, by label, and by what the
input change was. No row text.
"""
import json
import sys
from collections import Counter
from pathlib import Path


def load(run):
    rows = [json.loads(l) for l in (Path(run) / 'rows.jsonl').read_text().splitlines() if l.strip()]
    return {r['text']: r for r in rows}


def flagged(r):
    return r.get('verdict') in ('ask', 'review')


def compare(a, b):
    if set(a) != set(b):
        raise SystemExit(f'the runs cover different rows: {len(set(a) ^ set(b))} differ')
    out = Counter()
    for text in a:
        ra, rb = a[text], b[text]
        if ra['input'] == rb['input']:
            out['input_same'] += 1
            if ra.get('output') != rb.get('output'):
                out['CONTROL_output_moved_on_same_input'] += 1
            continue
        out['input_changed'] += 1
        gold = ra['gold']
        fa, fb = flagged(ra), flagged(rb)
        out[f'changed/{gold}/{"flag" if fa else "pass"}->{"flag" if fb else "pass"}'] += 1
        out[f'changed/label={ra["label"]}/{"flag" if fa else "pass"}->{"flag" if fb else "pass"}'] += 1
    return out


def totals(rows):
    t = Counter()
    for r in rows.values():
        t[f'{r["gold"]}/{"flag" if flagged(r) else "pass"}'] += 1
        t[f'outcome/{r.get("outcome")}'] += 1
    return t


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    cmp = compare(a, b)
    control = cmp.pop('CONTROL_output_moved_on_same_input', 0)
    print(f'control: {control} rows with identical input and different output '
          f'(of {cmp["input_same"]} identical inputs)')
    ta, tb = totals(a), totals(b)
    for k in sorted(set(ta) | set(tb)):
        print(f'{k:32s} {ta[k]:5d} -> {tb[k]:5d}')
    for k in sorted(cmp):
        print(f'{k:48s} {cmp[k]:5d}')
    return 1 if control else 0


if __name__ == '__main__':
    sys.exit(main())
