#!/usr/bin/env python3
"""Tally blind verdict labels for an F9 gold pilot. Aggregates only.

Inputs, all in one pilot directory (outside the repo; corpora never live here):
  blind_key.jsonl   {"blind_id": "b01", "id": "<pool id>"}
  pool.jsonl        rows with "id", "set", "family", "intended", "gen"
  labels/<name>.tsv one line per row: blind_id<TAB>verdict<TAB>reason<TAB>note

The pilot gate (memory pilot-gate-the-rubric): a row every labeler answers the
same way AGAINST the generator's intent is a rubric bug or a generator bug —
read it. Splits are the design working: genuine ambiguity, not errors to chase.
The generator's "intended" is one more opinion, never gold; the report says
which labelers share a family with the row's generator, because a family
grading its own rows is not an independent vote.

Labeler output is validated before anything is counted: unknown, skipped or
repeated ids and verdicts outside the vocabulary are reported and the file is
refused, rather than counted around.
"""
import argparse, json, sys
from collections import Counter, defaultdict
from pathlib import Path

VERDICTS = ('allow', 'ask', 'unsure')


def load_labels(path, ids):
    got, problems = {}, []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split('\t')
        if len(parts) < 2:
            problems.append(f'line {n}: fewer than 2 fields')
            continue
        bid, verdict = parts[0].strip(), parts[1].strip().lower()
        if bid not in ids:
            problems.append(f'line {n}: unknown id {bid!r}')
        elif bid in got:
            problems.append(f'line {n}: repeated id {bid}')
        elif verdict not in VERDICTS:
            problems.append(f'line {n}: verdict {verdict!r} outside {VERDICTS}')
        else:
            got[bid] = verdict
    missing = sorted(ids - set(got))
    if missing:
        problems.append(f'{len(missing)} ids never labelled, first {missing[:5]}')
    return got, problems


def load_unique(path, field, value):
    """Rows of a jsonl file keyed by `field`; a repeated key is refused, never
    last-wins: a duplicated blind id would let one labeler answer count twice."""
    out = {}
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        k = r[field]
        if k in out:
            raise SystemExit(f'{path.name} line {n}: repeated {field} {k!r}')
        out[k] = value(r)
    return out


def tally(key, pool, labels, families):
    rows = []
    for bid, pid in sorted(key.items()):
        r = pool[pid]
        votes = {name: lab[bid] for name, lab in labels.items()}
        rows.append((bid, r, votes))
    out = {'rows': len(rows), 'labelers': sorted(labels)}
    # Each labeler against the generator's intent, overall and by set.
    per = {}
    for name in labels:
        c = Counter()
        for _, r, v in rows:
            c[(r['set'], r['intended'], v[name])] += 1
        agree = sum(n for (s, i, v), n in c.items() if i == v)
        per[name] = {'agrees_with_intended': [agree, len(rows)],
                     'unsure': sum(n for (s, i, v), n in c.items() if v == 'unsure'),
                     'by_set_intended_verdict': {f'{s}/{i}->{v}': n for (s, i, v), n in sorted(c.items())}}
    out['per_labeler'] = per
    # Row shapes.
    shape = Counter()
    by_family = defaultdict(Counter)
    flagged = []
    for bid, r, v in rows:
        vals = list(v.values())
        top, n = Counter(vals).most_common(1)[0]
        if n == len(vals):
            kind = 'unanimous_with_intended' if top == r['intended'] else 'unanimous_against_intended'
        elif n > len(vals) / 2:
            kind = 'majority'
        else:
            kind = 'split'
        shape[kind] += 1
        by_family[r['family']][kind] += 1
        if kind != 'unanimous_with_intended':
            # Same family only when both sides are KNOWN: a row without `gen`
            # or a labeler without --family is unattributed, and an
            # unattributed vote must never read as independent.
            gen = r.get('gen')
            own = [name for name in v if gen is not None and families.get(name) == gen]
            unattributed = [name for name in v if gen is None or name not in families]
            flagged.append({'blind_id': bid, 'kind': kind, 'set': r['set'], 'family': r['family'],
                            'intended': r['intended'], 'gen': r.get('gen'), 'votes': v,
                            'same_family_labelers': own,
                            'unattributed_labelers': unattributed})
    out['shape'] = dict(shape)
    out['shape_by_family'] = {f: dict(c) for f, c in sorted(by_family.items())}
    # Pairwise agreement between labelers.
    names = sorted(labels)
    out['pairwise_agreement'] = {
        f'{a}~{b}': round(sum(v[a] == v[b] for _, _, v in rows) / len(rows), 3)
        for i, a in enumerate(names) for b in names[i + 1:]}
    out['to_read'] = flagged
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('pilot', type=Path)
    ap.add_argument('--family', action='append', default=[],
                    help='labeler=generator-family, e.g. deepseek=deepseek; marks same-family votes')
    a = ap.parse_args()
    key = load_unique(a.pilot / 'blind_key.jsonl', 'blind_id', lambda r: r['id'])
    pool = load_unique(a.pilot / 'pool.jsonl', 'id', lambda r: r)
    unknown = sorted(pid for pid in key.values() if pid not in pool)
    if unknown:
        raise SystemExit(f'blind_key names {len(unknown)} ids absent from pool.jsonl, first {unknown[:5]}')
    labels, bad = {}, False
    for path in sorted((a.pilot / 'labels').glob('*.tsv')):
        got, problems = load_labels(path, set(key))
        if problems:
            bad = True
            print(f'REFUSED {path.name}:', *problems, sep='\n  ', file=sys.stderr)
        else:
            labels[path.stem] = got
    if bad:
        raise SystemExit('fix or re-request the refused label files; nothing was counted')
    if len(labels) < 2:
        raise SystemExit('need at least two labelers')
    families = dict(f.split('=', 1) for f in a.family)
    print(json.dumps(tally(key, pool, labels, families), indent=1))


if __name__ == '__main__':
    main()
