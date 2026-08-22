#!/usr/bin/env python3
"""Forward-looking CLAUSE-FAITHFUL shadow scorer: tail the advisory log,
score each row's RECORDED clauses with a candidate checkpoint exactly the
way the daemon scored them -- whole clauses, no truncation, unpadded --
aggregate with the live cascade rule, and log agree/disagree locally.

Replaces the pre-2026-08-22 scorer, which scored the whole command text in
one pass truncated to 128 tokens. Production splits compounds into clauses
(/v1/cascade, since 2026-08-12) and truncates NOTHING
(sequence_classification.rs's compute_logits sets no truncation; the
checkpoint tokenizer.json carries `truncation: None`), so that scorer's
verdicts disagreed with production on three axes at once and could not be
compared with live. Measured 2026-08-22: a clause-faithful replay
reproduces the pod on 15,079/15,079 rows -- verdict-level CPU/GPU drift is
below measurement, so disagreement logged by THIS scorer is signal. See
the 2026-08-22 signoff entry and clause_replay.py.

Fields are SELF-DESCRIBING and must stay that way: the old schema's
'v8_verdict'/'candidate_verdict' silently inverted on the 2026-08-16
cutover (production became v9 and the scorer was repointed at v8, so a
field called 'v8_verdict' carried v9's answer). Mislabelled data is data
corruption, and a reader has no way to notice. Record which model said
what. Field semantics are pinned by test_shadow_score.py.

Privacy: same contract as the advisory log itself (0600, local-only, real
command text stays on this box, never committed to git). The output file is
chmod 0600 immediately on creation.

    .venv-train/bin/python training/v9/severity_probes/shadow_score.py \\
        --model /tank/ml/models/lfm2d/kube_ordinal_v9_cal \\
        --log ~/.cache/claude-hooks/lfm2d-advisory.jsonl \\
        --out ~/.cache/claude-hooks/lfm2d-shadow.jsonl
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# MUST precede the torch import (OpenMP reads these once, at library init).
# A long-lived poll loop that scores a few rows a minute has no business
# holding a spinning thread pool: measured 2026-08-16, torch's default
# 16-thread ATen pool busy-waited 1.5 cores while COMPLETELY IDLE. PASSIVE
# makes idle workers sleep; 1 thread is ample for a 350M encoder at this
# rate.
os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')
os.environ.setdefault('OMP_NUM_THREADS', '1')

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
from backtest_candidate import VALID_DEVICES, load  # noqa: E402
from clause_replay import (  # noqa: E402
    DEFAULT_SEVERE_ORDER, FIRED_LABELS, aggregate_row, length_buckets,
    probs_batch, replay_labels, row_inputs, severity_weights,
)


def is_replayable(row):
    """True when an advisory row carries everything a faithful replay needs:
    a live verdict to compare against and the exact texts the daemon saw."""
    lf = row.get('lfm2d') or {}
    if not lf.get('ok') or not row.get('command'):
        return False
    endpoint = lf.get('endpoint') or 'classify'
    if endpoint in ('cascade', 'classify_batch'):
        return bool(lf.get('clauses'))
    return lf.get('top') is not None


def shadow_row(row, replay, shadow_model):
    """One self-describing output row. See the module docstring for why the
    field names are load-bearing; test_shadow_score.py pins them."""
    lf = row['lfm2d']
    endpoint = lf.get('endpoint') or 'classify'
    out = {
        'ts': time.time(),
        'command': row['command'],
        'endpoint': endpoint,
        'live_model': lf.get('model_id'),
        'shadow_model': shadow_model,
        'clause_count': len(row_inputs(row)[1]),
    }
    if endpoint == 'classify_batch':
        # no winner exists on these rows by design -- compare firings
        live_fired = any(c.get('top') in FIRED_LABELS
                         for c in lf.get('clauses', []))
        out['live_verdict'] = None
        out['shadow_verdict'] = None
        out['live_fired'] = live_fired
        out['shadow_fired'] = bool(replay['fired'])
        out['agree'] = live_fired == bool(replay['fired'])
    else:
        out['live_verdict'] = lf.get('top')
        out['shadow_verdict'] = replay['top']
        out['agree'] = replay['top'] == lf.get('top')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, type=Path)
    ap.add_argument('--base', default='.models/LFM2.5-Encoder-350M', type=Path)
    ap.add_argument('--log', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--poll-seconds', type=float, default=5.0)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--device', choices=VALID_DEVICES, default='cpu',
                    help="default 'cpu' ON PURPOSE: this is a poll loop, and "
                         "the ROCm HIP runtime busy-waits a thread at 100%% "
                         "of a core for the whole run (measured 2026-08-16: "
                         "13.5 core-hours for 260 forwards at 0%% GPU util).")
    ap.add_argument('--threads', type=int, default=1,
                    help='ATen intra-op threads (default 1; see the OMP note above)')
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)

    labels = replay_labels(args.model)
    # Loud refusal, not a guess: a candidate that does not speak the live
    # three-rung vocabulary cannot shadow it. severity_weights raises on the
    # same condition per-clause, but failing at startup is cheaper.
    weights = severity_weights(labels, DEFAULT_SEVERE_ORDER)
    if 'data-critical' not in labels:
        raise SystemExit(f'{args.model.name}: no data-critical in {labels}')

    trunk, tok, W, b, labels, device = load(args.model, args.base, args.device)
    print(f'[shadow] loaded {args.model.name} on {device} '
          f'(clause-faithful: whole clauses, no truncation, unpadded)',
          flush=True)

    log_path = args.log.expanduser()
    out_path = args.out.expanduser()
    if not out_path.exists():
        out_path.touch()
    os.chmod(out_path, 0o600)

    # Start at end of file -- forward-looking only. Backfilling the past is
    # clause_replay.py's job (one-shot, bucketed, much faster); this loop
    # only ever scores traffic it saw arrive.
    offset = log_path.stat().st_size
    print(f'[shadow] tailing {log_path} from offset {offset}', flush=True)

    while True:
        size = log_path.stat().st_size
        if size < offset:
            # log rotated/truncated -- resync rather than crash
            offset = 0
        if size > offset:
            with open(log_path) as f:
                f.seek(offset)
                chunk = f.read()
                offset = f.tell()
            rows = []
            for line in [l for l in chunk.splitlines() if l.strip()]:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if is_replayable(r):
                    rows.append(r)
            if rows:
                spans, texts = [], []
                for r in rows:
                    endpoint, inputs = row_inputs(r)
                    spans.append((endpoint, len(inputs)))
                    texts.extend(inputs)
                # length-uniform batches: padding leaks into the trunk's
                # unmasked convolutions (clause_replay.length_buckets)
                lengths = [len(tok(t)['input_ids']) for t in texts]
                probs = [None] * len(texts)
                for bucket in length_buckets(lengths):
                    for i in range(0, len(bucket), args.batch_size):
                        part = bucket[i:i + args.batch_size]
                        for j, p in zip(part, probs_batch(
                                trunk, tok, W, b, device,
                                [texts[k] for k in part])):
                            probs[j] = p
                out_rows, pos, agree = [], 0, 0
                for r, (endpoint, n) in zip(rows, spans):
                    v = aggregate_row(endpoint, probs[pos:pos + n],
                                      labels, weights)
                    pos += n
                    sr = shadow_row(r, v, args.model.name)
                    agree += sr['agree']
                    out_rows.append(sr)
                with open(out_path, 'a') as out:
                    for sr in out_rows:
                        out.write(json.dumps(sr) + '\n')
                os.chmod(out_path, 0o600)
                print(f'[shadow] scored {len(rows)} row(s), {agree} agree',
                      flush=True)
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
