#!/usr/bin/env python3
"""Slice 3: the shape-labeled live sample's INPUT — top plan-rendered
shapes from the baseline window, ranked by clause count, with real
example clauses attached for the labeling pilot.

The renderer is the SERVING renderer (lfm2d/hooks/kaish_plan.py), not a
reimplementation: training text must match what the hook scores, or the
sample labels one distribution and the model serves another. The shape
key is soak_shapes.shape() — argv0 + subcommand + flags + redirect class.

Output discipline (ruling 2026-08-24, option (b)): the ranked-shape
JSONL carries REAL clause text as examples, so it writes to a 0600 file
and stays local — `*.jsonl` is gitignored and this tool refuses to
write inside the repo. Scrubbing happens at training-set construction,
not here; stdout prints AGGREGATES ONLY (shape keys and counts).

    .venv-train/bin/python training/v10/shape_sample.py \
        --model-id kube_ordinal_v9_cal --top 400
"""
import argparse
import collections
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / 'lfm2d' / 'hooks'))
from kaish_plan import plan_clauses  # noqa: E402  (the serving renderer)
from soak_shapes import DEFAULT_LOG, QUOTING_PROMPT_TS, load_rows, shape  # noqa: E402

EXAMPLES_PER_SHAPE = 3


def rank_shapes(clause_lists):
    """[(shape, count, [examples])] ranked by clause count, from lists of
    rendered clause texts (one list per command). Pure — testable."""
    counts = collections.Counter()
    examples = collections.defaultdict(list)
    for clauses in clause_lists:
        for text in clauses:
            s = shape(text)
            counts[s] += 1
            if len(examples[s]) < EXAMPLES_PER_SHAPE and text not in examples[s]:
                examples[s].append(text)
    return [(s, n, examples[s]) for s, n in counts.most_common()]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', default=DEFAULT_LOG)
    ap.add_argument('--model-id', required=True)
    ap.add_argument('--until', type=float, default=QUOTING_PROMPT_TS)
    ap.add_argument('--top', type=int, default=400)
    ap.add_argument('--out', default=str(Path.home() / '.cache/claude-hooks/v10-shape-sample.jsonl'))
    ap.add_argument('--jobs', type=int, default=8)
    args = ap.parse_args(argv)

    out = Path(args.out).resolve()
    if HERE.parent.parent in out.parents:
        raise SystemExit(f'{out} is inside the repo — the sample carries real '
                         f'clause text and must stay local (ruling (b): scrub at '
                         f'training-set construction, not before)')

    rows = load_rows(args.log, args.model_id, until=args.until)
    commands = sorted({d['command'] for d in rows})
    print(f'{len(rows)} rows, {len(commands)} distinct commands in window')

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        plans = list(ex.map(plan_clauses, commands))

    planned, fallback = [], 0
    for p in plans:
        if p.get('ok'):
            planned.append([c['text'] for c in p['clauses']])
        else:
            fallback += 1
    print(f'planned {len(planned)} ({len(planned)/len(commands):.1%}); '
          f'fallback {fallback} — fallback commands are the clause_split '
          f'population, sampled separately')

    ranked = rank_shapes(planned)
    total = sum(n for _, n, _ in ranked)
    cum = 0
    for k, (_, n, _) in enumerate(ranked[:args.top], 1):
        cum += n
        if k in (10, 25, 50, 100, 200, 400):
            print(f'  top {k:4d} shapes cover {cum/total:.1%} of {total} clauses')

    with open(os.open(out, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600), 'w') as f:
        for rank, (s, n, ex_list) in enumerate(ranked[:args.top], 1):
            f.write(json.dumps({'rank': rank, 'shape': s, 'clauses': n,
                                'share': round(n / total, 5), 'examples': ex_list}) + '\n')
    print(f'wrote top {min(args.top, len(ranked))} shapes -> {out} (0600, local only)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
