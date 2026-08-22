#!/usr/bin/env python3
"""Extract the rows where a local CPU replay of v9_cal disagrees with the
recorded live GPU verdict, as a 0600 markdown file for manual review.

CONTRACT DIFFERS FROM clause_replay.py ON PURPOSE: that script emits
aggregates only; this one writes raw command text, because a human review
of borderline rows is the point. The output file is chmod 0600, local-only,
and inherits the advisory log's privacy contract -- never committed, never
quoted outside this box. The SCRIPT is committable (it contains no data).

These flips are not "CPU vs GPU disagreement" so much as the model sitting
on a fence: same weights, two numerics, ~2% of rows land within epsilon of
a decision boundary. That makes the flip set the cheapest labeled-by-shape
sample available: on each row, the question "which side SHOULD this be on?"
is exactly the tau/precision question the soak exists to answer.

    .venv-train/bin/python training/v9/severity_probes/flip_review.py \\
        --model-id kube_ordinal_v9_cal \\
        --out ~/.cache/claude-hooks/flip-review-2026-08-22.md
"""
import argparse
import datetime
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
from backtest_candidate import VALID_DEVICES, load  # noqa: E402
from clause_replay import (  # noqa: E402
    DEFAULT_LOG, aggregate_row, day_key, length_buckets, load_rows,
    probs_batch, replay_labels, row_inputs, severity_weights,
)

SEVERE_ORDER = ['situation-normal', 'data-critical']  # deploy/k8s-zorak.yaml


def replay_with_probs(model_dir, base_dir, device, rows, weights, batch_size,
                      probs_cache=None):
    """Same scoring path as clause_replay.replay, but keeping per-input
    probabilities: the review needs margins and per-clause verdicts.

    probs_cache, when given, is a 0600 JSON of the flat per-input
    probability rows: scoring this window takes ~40 min on CPU, and report
    formatting is where iteration happens -- a cache keeps the two apart.
    """
    labels = replay_labels(model_dir)
    if 'data-critical' not in labels:
        raise SystemExit(f'{model_dir.name}: no data-critical in {labels}')
    spans, texts = [], []
    for r in rows:
        endpoint, inputs = row_inputs(r)
        spans.append((endpoint, inputs))
        texts.extend(inputs)
    if probs_cache is not None and probs_cache.exists():
        with open(probs_cache) as f:
            probs = json.load(f)
        if len(probs) != len(texts):
            raise SystemExit(
                f'probs cache has {len(probs)} inputs but the log now replays '
                f'{len(texts)} -- the log grew since the cache was written; '
                're-score without --load-probs')
        print(f'[flip_review] loaded {len(probs)} probability rows from {probs_cache}',
              flush=True)
    else:
        trunk, tok, W, b, labels, device = load(model_dir, base_dir, device)
        probs = [None] * len(texts)
        lengths = [len(tok(t)['input_ids']) for t in texts]
        for bucket in length_buckets(lengths):
            for i in range(0, len(bucket), batch_size):
                chunk = bucket[i:i + batch_size]
                for j, p in zip(chunk, probs_batch(trunk, tok, W, b, device,
                                                   [texts[k] for k in chunk])):
                    probs[j] = p
        del trunk
        if probs_cache is not None:
            with open(probs_cache, 'w') as f:
                json.dump(probs, f)
            os.chmod(probs_cache, 0o600)
            print(f'[flip_review] cached probabilities to {probs_cache}', flush=True)
    results, pos = [], 0
    for (endpoint, inputs), r in zip(spans, rows):
        clause_probs = probs[pos:pos + len(inputs)]
        pos += len(inputs)
        v = aggregate_row(endpoint, clause_probs, labels, weights)
        tops = [labels[max(range(len(labels)), key=lambda i: p[i])]
                for p in clause_probs]
        if v['winner'] is not None:
            w = sorted(clause_probs[v['winner']], reverse=True)
            margin = w[0] - w[1]
        else:
            margin = None
        results.append({
            'endpoint': endpoint, 'inputs': inputs, 'verdict': v,
            'clause_tops': tops, 'margin': margin,
        })
    return labels, results


def recorded_outcome(row):
    """(top, fired) as the live pod recorded them."""
    lf = row['lfm2d']
    if lf.get('endpoint') == 'classify_batch':
        fired = any(c.get('top') == 'data-critical' for c in lf.get('clauses', []))
        return None, fired
    return lf.get('top'), lf.get('top') == 'data-critical'


def fence(text):
    return '```shell\n' + text + '\n```'


def fmt_margin(margin):
    # classify_batch rows have no winner, hence no margin -- say so instead
    # of formatting None (which crashed the first full run AFTER its 40-min
    # scoring pass; the cache flag exists because of exactly this).
    return f'{margin:.3f}' if margin is not None else 'n/a, batch row'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='/tank/ml/models/lfm2d/kube_ordinal_v9_cal', type=Path)
    ap.add_argument('--base', default='.models/LFM2.5-Encoder-350M', type=Path)
    ap.add_argument('--log', default=DEFAULT_LOG, type=Path)
    ap.add_argument('--model-id', required=True)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--sample-agreed-firings', type=int, default=15)
    ap.add_argument('--save-probs', type=Path,
                    help='cache scored probabilities here after scoring')
    ap.add_argument('--load-probs', type=Path,
                    help='skip scoring; rebuild the report from this cache')
    ap.add_argument('--device', choices=VALID_DEVICES, default='cpu')
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)

    rows = load_rows(args.log.expanduser(), args.model_id)
    if not rows:
        print(f'no replayable rows for model_id={args.model_id}', file=sys.stderr)
        return 1
    weights = severity_weights(replay_labels(args.model), SEVERE_ORDER)
    cache = args.load_probs or args.save_probs
    labels, results = replay_with_probs(args.model, args.base, args.device,
                                        rows, weights, args.batch_size,
                                        probs_cache=cache)

    flips, agreed_firings = [], []
    for r, res in zip(rows, results):
        rec_top, rec_fired = recorded_outcome(r)
        v = res['verdict']
        is_flip = (v['top'] != rec_top) if res['endpoint'] != 'classify_batch' \
            else (v['fired'] != rec_fired)
        if is_flip:
            flips.append((r, res, rec_top, rec_fired))
        elif v['fired']:
            agreed_firings.append((r, res))

    dc_flips = [(r, res, rt, rf) for r, res, rt, rf in flips
                if 'data-critical' in (rt, res['verdict']['top'])]
    rng = random.Random(20260822)
    sampled = rng.sample(agreed_firings, min(args.sample_agreed_firings, len(agreed_firings)))

    out = args.out.expanduser()
    with open(out, 'w') as f:
        f.write(f'# v9_cal soak flip review — CPU replay vs recorded live\n\n')
        f.write(f'Rows: {len(rows)}   flips: {len(flips)} '
                f'({100 * len(flips) / len(rows):.2f}%)   '
                f'data-critical flips: {len(dc_flips)}\n\n')
        f.write('Both verdicts come from the SAME weights (one GPU forward at '
                'traffic time, one CPU forward now). A flip means the row sits '
                'within rounding distance of a decision boundary -- "which side '
                'SHOULD it be on" is the review question.\n\n')

        f.write('## data-critical flips (the consequential ones)\n\n')
        for r, res, rec_top, _ in dc_flips:
            v = res['verdict']
            when = datetime.datetime.fromtimestamp(r['ts']).strftime('%m-%d %H:%M')
            f.write(f'### {when} — recorded `{rec_top}` vs replayed `{v["top"]}`'
                    f' (replay margin {fmt_margin(res["margin"])}, endpoint {res["endpoint"]})\n\n')
            f.write(fence(r['command']) + '\n\n')
            if res['endpoint'] != 'classify' or len(res['inputs']) == 1:
                if v['winner'] is not None and res['endpoint'] == 'cascade':
                    f.write(f'winner clause (cpu tops {res["clause_tops"]}): '
                            f'{res["inputs"][v["winner"]]!r}\n\n')
                elif res['endpoint'] == 'classify_batch':
                    f.write(f'clause tops (cpu): {res["clause_tops"]}\n\n')

        f.write('## other flips (non-dc; direction counts)\n\n')
        dirs = Counter(f'{rt} -> {res["verdict"]["top"]}'
                       for r, res, rt, rf in flips
                       if 'data-critical' not in (rt, res['verdict']['top']))
        for k, c in dirs.most_common():
            f.write(f'- {k}: {c}\n')
        f.write('\n')

        f.write(f'## {len(sampled)} agreed firings (both sides data-critical), '
                'random sample\n\n')
        for r, res in sampled:
            when = datetime.datetime.fromtimestamp(r['ts']).strftime('%m-%d %H:%M')
            f.write(f'### {when} (margin {fmt_margin(res["margin"])}, '
                    f'endpoint {res["endpoint"]})\n\n')
            f.write(fence(r['command']) + '\n\n')
            if res['endpoint'] == 'cascade':
                f.write(f'winner clause: {res["inputs"][res["verdict"]["winner"]]!r}\n\n')

    os.chmod(out, 0o600)
    dirs = Counter(f'{rt} -> {res["verdict"]["top"]}' for r, res, rt, rf in flips)
    print(f'{len(rows)} rows, {len(flips)} flips ({100 * len(flips) / len(rows):.2f}%), '
          f'{len(dc_flips)} dc flips, {len(agreed_firings)} agreed firings')
    for k, c in dirs.most_common():
        print(f'  {k}: {c}')
    print(f'review file: {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
