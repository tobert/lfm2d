#!/usr/bin/env python3
"""Replay the advisory log's RECORDED clauses through exported checkpoints,
reproducing the live verdict path clause by clause.

Written 2026-08-22 after six soak days of shadow scoring turned out to
compare two different verdict paths: the live hook splits compounds into
clauses and reports the winner clause's argmax (/v1/cascade), while
shadow_score.py scores the whole command text in one pass. Whole-command
classification dilutes severe clauses (clause_split.py measured 11x), so
whole-text shadow verdicts are not comparable with live verdicts. This
replayer scores the SAME texts the daemon saw -- the clauses the hook
recorded on each row -- and aggregates them with the live rule, so two
checkpoints replayed through it are comparable with each other AND with
what production actually did.

Endpoints reproduced (pre_command_advisory.py dispatch):
  cascade        2..20 clauses: winner = max ordinal severity, ties to the
                 earlier clause (stable descending, cascade.rs contract);
                 row verdict = winner clause's argmax label.
  classify       one clause: its argmax label (uses `sent`, the text the
                 hook actually scored after comment/keyword stripping).
  classify_batch >20 clauses: no winner by design; firing is
                 set-membership -- any clause argmax in the fired set.

Privacy: same contract as backtest_candidate.py -- the advisory log is
0600, local-only, real command text; this script emits AGGREGATE counts
only. Do not add per-row text output.

Scoring batches are length-uniform ON PURPOSE: the trunk's convolutions
are not attention-masked, so padded batches leak pad embeddings into real
tokens and move verdicts (measured 2026-08-22: 80.3% row agreement with
the live pod at batch=64 padded vs 98.3% with no padding; see
length_buckets). The daemon scores clauses one at a time, so it never hits
this.

    .venv-train/bin/python training/v9/severity_probes/clause_replay.py \\
        --model /tank/ml/models/lfm2d/kube_ordinal_v9_cal \\
        --model-id kube_ordinal_v9_cal
    .venv-train/bin/python training/v9/severity_probes/clause_replay.py \\
        --model /tank/ml/models/lfm2d/kube_ordinal_v8 \\
        --compare /tank/ml/models/lfm2d/kube_ordinal_v9_cal \\
        --model-id kube_ordinal_v9_cal
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

# MUST precede the torch import (OpenMP reads these once, at library init);
# same lesson as shadow_score.py. A batch replay keeps the pool busy, but
# PASSIVE still matters for the load phase and any idle gap between batches.
os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
from backtest_candidate import VALID_DEVICES, load  # noqa: E402
from shape_impact import shapes_in  # noqa: E402

DEFAULT_LOG = Path('~/.cache/claude-hooks/lfm2d-advisory.jsonl')
FIRED_LABELS = frozenset({'data-critical'})  # the hook's SEVERE_LABELS default
# Winner-ranking weights, matching deploy/k8s-zorak.yaml's
# --cascade-severe-label flags: ascending severity, least severe first,
# Nth label weighs N. informative weighs 0: prose/data-position mass does
# not outrank real commands when picking which clause represents a row.
DEFAULT_SEVERE_ORDER = ['situation-normal', 'data-critical']


def severity_weights(labels, severe_labels):
    """Per-label ordinal rungs: the Nth severe label (ascending) weighs N,
    every other label 0. Mirrors cascade::severity_rank_weights, including
    its refusals: an empty list, a duplicate, or a severe label the
    checkpoint does not speak all raise rather than rank on a guess."""
    if not severe_labels:
        raise ValueError('severity ranking needs at least one severe label')
    if len(set(severe_labels)) != len(severe_labels):
        raise ValueError(f'duplicate severe label in {severe_labels!r}: its rung '
                         'would be ambiguous')
    unknown = [l for l in severe_labels if l not in labels]
    if unknown:
        raise ValueError(f'severe label(s) {unknown!r} not in checkpoint labels {labels!r}')
    weights = {l: 0.0 for l in labels}
    for rank, l in enumerate(severe_labels, start=1):
        weights[l] = float(rank)
    return weights


def clause_severity(probs, labels, weights):
    """Expected ordinal rank: sum(weight[label] * p[label])."""
    return sum(w * p for p, w in zip(probs, (weights[l] for l in labels)))


def pick_winner(scores):
    """Index of the max severity; exact ties go to the EARLIER clause
    (stable-descending contract, tests/cascade.rs)."""
    winner = 0
    for i, s in enumerate(scores[1:], start=1):
        if s > scores[winner]:
            winner = i
    return winner


def aggregate_row(endpoint, clause_probs, labels, weights, fired=FIRED_LABELS):
    """One row's replayed verdict from per-input probability rows.

    Returns {'top', 'fired', 'winner'}: top is None on classify_batch rows
    by design (no winner exists there), fired is the hook's flag decision.
    """
    if not clause_probs:
        raise ValueError('no clause probabilities: nothing was scored')
    argmaxes = [max(range(len(labels)), key=lambda i: p[i]) for p in clause_probs]
    if endpoint == 'classify':
        top = labels[argmaxes[0]]
        return {'top': top, 'fired': top in fired, 'winner': 0}
    if endpoint == 'classify_batch':
        return {'top': None, 'fired': any(labels[i] in fired for i in argmaxes),
                'winner': None}
    if endpoint == 'cascade':
        sev = [clause_severity(p, labels, weights) for p in clause_probs]
        winner = pick_winner(sev)
        top = labels[argmaxes[winner]]
        return {'top': top, 'fired': top in fired, 'winner': winner}
    raise ValueError(f'unknown endpoint {endpoint!r}')


def row_inputs(row):
    """(endpoint, [texts the daemon actually scored]) for one advisory row."""
    lf = row['lfm2d']
    endpoint = lf.get('endpoint') or 'classify'
    if endpoint in ('cascade', 'classify_batch'):
        return endpoint, [c['clause'] for c in lf['clauses']]
    # classify rows record `sent` when the splitter cleaned the command;
    # that, not the raw command, is what the daemon saw.
    return 'classify', [lf.get('sent') or row['command']]


def load_rows(log_path, model_id):
    """Advisory rows scored live by model_id, replayable clause text intact."""
    rows = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            lf = r.get('lfm2d') or {}
            if not lf.get('ok') or lf.get('model_id') != model_id or not r.get('command'):
                continue
            if not lf.get('clauses') and lf.get('endpoint') in ('cascade', 'classify_batch'):
                continue  # recorded without clause text: not replayable
            rows.append(r)
    return rows


def length_buckets(lengths):
    """Group indices so every batch is length-uniform and needs no padding.

    Padding is NOT neutral for this trunk: its short convolutions are not
    masked the way attention is, so pad embeddings mix into real tokens and
    move verdicts. Measured 2026-08-22 on 300 soak rows: batch=64 with
    padding agreed with the live pod on 80.3% of row verdicts; batch=1 on
    the same rows agreed on 98.3% -- the remaining ~2% being the known
    CPU/GPU tie drift. Bucketing by encoded length keeps batched scoring
    equivalent to the daemon's per-clause forwards, which never pad.
    """
    buckets, order = {}, []
    for i, n in enumerate(lengths):
        if n not in buckets:
            buckets[n] = []
            order.append(n)
        buckets[n].append(i)
    return [buckets[n] for n in order]


@torch.no_grad()
def probs_batch(trunk, tok, W, b, device, texts, max_len=None):
    """Softmax probabilities per input, same forward math as the shadow
    scorer (CLS pooling) but returning the full probability row -- the
    cascade winner needs more than the argmax.

    max_len=None (the default) is a DAEMON-FIDELITY choice, verified
    2026-08-22: sequence_classification.rs's compute_logits calls
    tokenizer.encode with NO truncation params, and the checkpoint's
    tokenizer.json carries `truncation: None` -- the live pod scores the
    whole clause, however long. The shadow scorer's historical max_len=128
    does NOT match production; truncating here silently disagreed with the
    pod on exactly the long payload-heavy clauses (merge messages, heredoc
    scripts) where the verdicts are interesting."""
    if max_len is None:
        enc = tok(texts, return_tensors='pt', padding=True)
    else:
        enc = tok(texts, return_tensors='pt', truncation=True,
                  max_length=max_len, padding=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    out = trunk(**enc)
    h = out.last_hidden_state[:, 0, :].float()
    logits = h @ W.T + b
    return torch.softmax(logits, dim=-1).cpu().tolist()


def replay(model_dir, base_dir, device, rows, weights, batch_size):
    """Row-level replayed verdicts for one checkpoint, in row order."""
    trunk, tok, W, b, labels, device = load(model_dir, base_dir, device)
    if 'data-critical' not in labels:
        raise SystemExit(f'{model_dir.name}: no data-critical in {labels}')
    # one flat scoring pass over every clause, then reassemble rows
    spans, texts = [], []
    for r in rows:
        endpoint, inputs = row_inputs(r)
        spans.append((endpoint, len(inputs)))
        texts.extend(inputs)
    probs = [None] * len(texts)
    # Length-uniform batches: padding would leak into conv layers (see
    # length_buckets). Encode once for lengths, then score bucket by bucket.
    # No truncation here, matching the daemon (see probs_batch).
    lengths = [len(tok(t)['input_ids']) for t in texts]
    for bucket in length_buckets(lengths):
        for i in range(0, len(bucket), batch_size):
            chunk = bucket[i:i + batch_size]
            for j, p in zip(chunk, probs_batch(trunk, tok, W, b, device,
                                               [texts[k] for k in chunk])):
                probs[j] = p
    del trunk
    verdicts, pos = [], 0
    for endpoint, n in spans:
        v = aggregate_row(endpoint, probs[pos:pos + n], labels, weights)
        pos += n
        verdicts.append(v)
    return labels, verdicts


def day_key(ts):
    import datetime
    return datetime.datetime.fromtimestamp(ts).strftime('%m-%d')


def report(rows, verdicts, name):
    n = len(rows)
    fired = [i for i, v in enumerate(verdicts) if v['fired']]
    tops = Counter(v['top'] for v in verdicts)
    print(f'\n== {name}: {n} rows, {len(fired)} firings '
          f'({100 * len(fired) / n:.2f}%) ==')
    for t, c in tops.most_common():
        print(f'  {t or "(batch, no winner)"}: {c}')
    per_day = Counter()
    per_day_fired = Counter()
    for i, r in enumerate(rows):
        d = day_key(r['ts'])
        per_day[d] += 1
        per_day_fired[d] += verdicts[i]['fired']
    for d in sorted(per_day):
        print(f'  {d}: rows={per_day[d]:6d} firings={per_day_fired[d]:4d} '
              f'({100 * per_day_fired[d] / per_day[d]:.1f}%)')
    return fired


def shape_breakdown(rows, idxs, name):
    if not idxs:
        return
    n_all = len(rows)
    shape_fired = Counter()
    for i in idxs:
        for s in shapes_in(rows[i]['command']):
            shape_fired[s] += 1
    print(f'  shapes among {len(idxs)} {name} firings '
          f'(vs their share of all {n_all} rows):')
    for s, c in shape_fired.most_common():
        in_all = sum(1 for r in rows if s in shapes_in(r['command']))
        print(f'    {s}: {c} ({100 * c / len(idxs):.1f}% of firings; '
              f'{100 * in_all / n_all:.1f}% of all rows)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, type=Path)
    ap.add_argument('--compare', type=Path,
                    help='second checkpoint, replayed on the identical inputs')
    ap.add_argument('--base', default='.models/LFM2.5-Encoder-350M', type=Path)
    ap.add_argument('--log', default=DEFAULT_LOG, type=Path)
    ap.add_argument('--model-id', required=True,
                    help='live model_id whose rows to replay (the log spans checkpoints)')
    ap.add_argument('--severe-order', default=','.join(DEFAULT_SEVERE_ORDER),
                    help='cascade winner-ranking rungs, ascending severity '
                         '(must match the deployed --cascade-severe-label flags)')
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--device', choices=VALID_DEVICES, default='cpu',
                    help="default 'cpu': a bulk replay here shares the box "
                         "with a live shadow scorer and llama-server; check "
                         "rocm-smi before asking for the GPU")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)

    rows = load_rows(args.log.expanduser(), args.model_id)
    if not rows:
        print(f'no replayable rows for model_id={args.model_id}', file=sys.stderr)
        return 1
    endpoints = Counter((r['lfm2d'].get('endpoint') or 'classify') for r in rows)
    n_inputs = sum(len(row_inputs(r)[1]) for r in rows)
    print(f'{len(rows)} live rows for {args.model_id} '
          f'({dict(endpoints)}), {n_inputs} scored inputs')

    # The severe ORDER is caller-supplied, exactly like the daemon's
    # --cascade-severe-label flags; validate it against each replayed
    # checkpoint's own vocabulary.
    severe_order = args.severe_order.split(',')

    labels_a, verdicts_a = replay(args.model, args.base, args.device, rows,
                                  severity_weights(
                                      replay_labels(args.model), severe_order),
                                  args.batch_size)
    fired_a = report(rows, verdicts_a, args.model.name)

    validate(rows, verdicts_a, args.model.name, args.model_id)

    if args.compare:
        labels_b, verdicts_b = replay(args.compare, args.base, args.device, rows,
                                      severity_weights(replay_labels(args.compare),
                                                       severe_order),
                                      args.batch_size)
        fired_b = report(rows, verdicts_b, args.compare.name)
        paired(rows, verdicts_a, verdicts_b, args.model.name, args.compare.name)
    return 0


def replay_labels(model_dir):
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    id2label = {int(k): v for k, v in cfg.id2label.items()}
    return [id2label[i] for i in sorted(id2label)]


def validate(rows, verdicts, model_name, model_id):
    """Replayed vs recorded live verdicts. Only interpretable when the
    replayed checkpoint IS the live one (same weights, different device:
    this measures CPU/GPU drift on real traffic)."""
    if model_name != model_id:
        print(f'\n(no validation: replayed {model_name} != live {model_id})')
        return
    agree_top = agree_fire = n_top = n_fire = 0
    flips = Counter()
    for r, v in zip(rows, verdicts):
        lf = r['lfm2d']
        if lf.get('endpoint') == 'classify_batch':
            live_fired = any(c.get('top') in FIRED_LABELS for c in lf.get('clauses', []))
            n_fire += 1
            agree_fire += live_fired == v['fired']
        else:
            n_top += 1
            live_top = lf.get('top')
            agree_top += live_top == v['top']
            if live_top != v['top']:
                flips[f'{live_top} -> {v["top"]}'] += 1
    print(f'\n== validation vs recorded live {model_id} (CPU replay vs GPU pod) ==')
    if n_top:
        print(f'  row-top agreement: {agree_top}/{n_top} ({100 * agree_top / n_top:.2f}%)')
        for k, c in flips.most_common(8):
            print(f'    {k}: {c}')
    if n_fire:
        print(f'  batch-row firing agreement: {agree_fire}/{n_fire}')


def paired(rows, va, vb, name_a, name_b):
    both = only_a = only_b = 0
    for x, y in zip(va, vb):
        both += x['fired'] and y['fired']
        only_a += x['fired'] and not y['fired']
        only_b += y['fired'] and not x['fired']
    n = len(rows)
    print(f'\n== paired firings: {name_a} vs {name_b}, identical inputs ==')
    print(f'  both: {both}   {name_a}-only: {only_a}   {name_b}-only: {only_b}   '
          f'neither: {n - both - only_a - only_b}')
    only_a_idxs = [i for i, (x, y) in enumerate(zip(va, vb)) if x['fired'] and not y['fired']]
    only_b_idxs = [i for i, (x, y) in enumerate(zip(va, vb)) if y['fired'] and not x['fired']]
    shape_breakdown(rows, only_a_idxs, f'{name_a}-only')
    shape_breakdown(rows, only_b_idxs, f'{name_b}-only')


if __name__ == '__main__':
    sys.exit(main())
