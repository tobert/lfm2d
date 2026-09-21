#!/usr/bin/env python3
"""How much of val_F does `kaish --plan` parse, and what does it refuse?

AGGREGATES ONLY. val_F.jsonl is a training corpus and never enters the
repo or this script's output: each row's `text` goes to `kaish
--plan-file -` (analyze, never execute) and the result is bucketed by
kaish's error token and by the row's classifier `label` -- never printed
as row text. Per-row diagnosis (translation gap / kaish bug / not a
command) is a human judgement call, made once per reject from the same
plan calls this script makes, and it lives in ~/exomemory/lfm2d/ --
never in this repo, same rule as the corpus itself
(`personal-process-lives-in-exomemory`).

Re-run whenever kaish or val_F changes; the table goes stale exactly the
way bash_to_kaish.py's does, for the same reason (`commit-the-scorer`).

    .venv-train/bin/python training/v10/kaish_rejects.py
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
DEFAULT_CORPUS = HERE / 'val_F.jsonl'


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


def error_key(e):
    msg = e['message']
    if 'unterminated heredoc' in msg:
        return 'unterminated heredoc'
    if msg.startswith('lexer error: unexpected character'):
        return 'unexpected character'
    if 'adjacent words' in msg:
        return 'adjacent words (quote-to-join)'
    if 'variable name contains' in msg:
        return 'invalid variable-name chars'
    # kaish's message quotes the offending token, and a token longer than one
    # character is a piece of the row (`+feature/new-ui` is a branch name).
    # Buckets are printed, so only punctuation survives into one.
    m = re.match(r"found '([^']*)'", msg)
    if m:
        tok = m.group(1)
        return f"found {tok!r}" if len(tok) == 1 and not tok.isalnum() else 'found a word'
    m = re.match(r"([a-zA-Z ]+error)", msg)
    if m:
        return m.group(1).strip()
    return 'other error'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', type=Path, default=DEFAULT_CORPUS)
    ap.add_argument('--jobs', type=int, default=8)
    args = ap.parse_args()
    if not args.corpus.exists():
        sys.exit(f'{args.corpus}: not found -- corpora are not in git, '
                  f'point --corpus at a real training/v10/val_F.jsonl')

    ver = subprocess.run(['kaish', '--version'], capture_output=True, text=True).stdout.strip()
    rows = [json.loads(line) for line in args.corpus.read_text().splitlines() if line.strip()]
    print(f'{ver}; {len(rows)} rows from {args.corpus.name}')

    with ThreadPoolExecutor(args.jobs) as ex:
        results = list(ex.map(plan, (r['text'] for r in rows)))

    kinds = collections.Counter(k for k, _ in results)
    print('outcomes:', dict(kinds))

    by_label_total = collections.Counter()
    by_label_reject = collections.Counter()
    by_bucket = collections.Counter()
    by_bucket_label = collections.Counter()

    for row, (kind, detail) in zip(rows, results):
        label = row.get('label', '?')
        by_label_total[label] += 1
        if kind == 'ok':
            continue
        by_label_reject[label] += 1
        if kind == 'error':
            bucket = error_key(detail)
        elif kind == 'timeout':
            bucket = 'timeout'
        else:
            bucket = 'nojson'
        by_bucket[bucket] += 1
        by_bucket_label[(bucket, label)] += 1

    n_reject = sum(by_label_reject.values())
    n_total = len(rows)
    print(f'\nrejected: {n_reject}/{n_total} = {n_reject / n_total:.1%}')

    print('\nby label (total / rejected / reject rate):')
    for label in sorted(by_label_total):
        t = by_label_total[label]
        r = by_label_reject[label]
        print(f'  {label:20s} {t:4d}  {r:4d}  {r / t:.1%}')

    print('\nby rejection bucket (kaish error token), total then per label:')
    for bucket, n in by_bucket.most_common():
        labels = ', '.join(f'{lbl}={c}' for (b, lbl), c in sorted(by_bucket_label.items())
                            if b == bucket)
        print(f'  {n:4d}  {bucket!r:40s} {labels}')


if __name__ == '__main__':
    main()
