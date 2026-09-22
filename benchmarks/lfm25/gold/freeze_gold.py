#!/usr/bin/env python3
"""Freeze a labelled pilot/bulk directory into gold.jsonl + gold.meta.json.

The verdict rule, in order:
  1. a pool row with `drop_reason` is left out (a leak or a duplicate found
     after blinding must not stay in the gold by inertia);
  2. Amy's ruling in amy-ruling.md (`## <blind_id> ...` then `Amy: <verdict>`)
     is the verdict, whatever the labelers said;
  3. otherwise every labeler must agree, and that is the verdict;
  4. a row the labelers split on with no ruling REFUSES the freeze -- gold is
     never a majority vote.

gold.meta.json records the rubric's version line and sha256, the sha256 of
every label file and of the ruling file, and counts by set/family/verdict,
so a number computed on this gold names exactly what it was computed on.
Prints aggregates only; the rows go to the file. stdlib only.
"""
import argparse, hashlib, json, re, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from pilot_tally import load_labels, load_unique

VERDICTS = ('allow', 'ask', 'unsure')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_rulings(text):
    """{blind_id: verdict} from the ruling file. The verdict is the first word
    after `Amy:`; `nope` reads as ask. A heading with an empty `Amy:` line is a
    row she has not ruled yet and is reported, not skipped."""
    rulings, unruled = {}, []
    for m in re.finditer(r'^## (\S+) [^\n]*\n(.*?)(?=^## |\Z)', text, re.M | re.S):
        bid, body = m.group(1), m.group(2)
        a = re.search(r'^Amy:[ \t]*(\S*)', body, re.M)
        if a is None or not a.group(1):
            unruled.append(bid)
            continue
        word = a.group(1).lower().strip('.,;:/-')
        word = {'nope': 'ask'}.get(word, word)
        if word not in VERDICTS:
            raise SystemExit(f'ruling for {bid} reads {a.group(1)!r}, not a verdict in {VERDICTS}')
        rulings[bid] = word
    return rulings, unruled


def freeze(key, pool, labels, rulings):
    gold, problems, dropped = [], [], []
    for bid, pid in sorted(key.items()):
        r = pool[pid]
        if r.get('drop_reason'):
            dropped.append(bid)
            continue
        votes = {name: lab[bid] for name, lab in labels.items()}
        distinct = set(votes.values())
        if bid in rulings:
            verdict, source = rulings[bid], 'amy'
        elif len(distinct) == 1:
            verdict, source = distinct.pop(), 'unanimous'
        else:
            problems.append(f'{bid}: labelers split {votes} and no ruling')
            continue
        gold.append({'id': pid, 'blind_id': bid, 'command': r['command'], 'set': r['set'],
                     'family': r['family'], 'pair': r.get('pair'), 'gen': r.get('gen'),
                     'intended': r.get('intended'), 'verdict': verdict, 'source': source,
                     'votes': votes})
    return gold, problems, dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('pilot', type=Path, help='directory with blind_key.jsonl, pool.jsonl, labels/, rubric.md')
    ap.add_argument('--ruling', type=Path, help='amy-ruling.md; default <pilot>/amy-ruling.md if present')
    ap.add_argument('--out', type=Path, help='output directory; default <pilot>')
    a = ap.parse_args()
    out = a.out or a.pilot
    key = load_unique(a.pilot / 'blind_key.jsonl', 'blind_id', lambda r: r['id'])
    pool = load_unique(a.pilot / 'pool.jsonl', 'id', lambda r: r)
    label_files = sorted((a.pilot / 'labels').glob('*.tsv'))
    labels = {}
    for path in label_files:
        got, problems = load_labels(path, set(key))
        if problems:
            raise SystemExit(f'REFUSED {path.name}: ' + '; '.join(problems))
        labels[path.stem] = got
    if len(labels) < 2:
        raise SystemExit('need at least two labelers')
    ruling_path = a.ruling or (a.pilot / 'amy-ruling.md')
    rulings, unruled = ({}, [])
    if ruling_path.exists():
        rulings, unruled = parse_rulings(ruling_path.read_text())
    if unruled:
        raise SystemExit(f'{len(unruled)} rows in {ruling_path.name} have an empty ruling: {unruled}')
    rubric = a.pilot / 'rubric.md'
    version = rubric.read_text().splitlines()[0]
    gold, problems, dropped = freeze(key, pool, labels, rulings)
    if problems:
        raise SystemExit('cannot freeze:\n  ' + '\n  '.join(problems))
    ruled_dropped = sorted(set(rulings) & set(dropped))
    meta = {
        'frozen_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'rubric': {'version_line': version, 'sha256': sha256(rubric)},
        'labels': {p.stem: sha256(p) for p in label_files},
        'ruling': {'file': ruling_path.name, 'sha256': sha256(ruling_path)} if ruling_path.exists() else None,
        'rows': len(gold), 'dropped': dropped, 'rulings_on_dropped_rows_ignored': ruled_dropped,
        'source': dict(Counter(g['source'] for g in gold)),
        'by_set_verdict': {f'{s}/{v}': n for (s, v), n in sorted(Counter((g['set'], g['verdict']) for g in gold).items())},
        'by_family_verdict': {f'{f}/{v}': n for (f, v), n in sorted(Counter((g['family'], g['verdict']) for g in gold).items())},
        'ruled_against_unanimous': sorted(g['blind_id'] for g in gold if g['source'] == 'amy'
                                          and len(set(g['votes'].values())) == 1
                                          and g['verdict'] not in g['votes'].values()),
    }
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'gold.jsonl').open('w', encoding='utf-8') as f:
        for g in gold:
            f.write(json.dumps(g, ensure_ascii=False) + '\n')
    (out / 'gold.meta.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(json.dumps(meta, indent=1))
    print('wrote', out / 'gold.jsonl', file=sys.stderr)


if __name__ == '__main__':
    main()
