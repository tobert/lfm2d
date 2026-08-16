#!/usr/bin/env python3
"""Fold a class-prior correction into a checkpoint's classifier bias.

WHY THIS EXISTS. v9 fires `data-critical` 4.1x as often as v8 on identical
inputs (21.7% vs 5.3% over the same 851 real commands) while ranking BETTER
on the standing gate (14/23 vs 5/23 delta-margin). It ranks well and
thresholds badly -- `rank-within-dont-threshold-across`, which is what a
learned class prior does: v9's training mix is 52.6% data-critical.

The standard balanced-posterior correction is p ∝ p / prior^tau, which in
logit space is a per-class CONSTANT:

    logits = h @ W.T + b        ->        b' = b - tau * log(prior)

So it needs no serving code at all -- it is exactly a new bias vector. That
is the whole point of this script: the calibrated head is an ordinary
checkpoint, deployable by the same one-line --classifier-dir change, and
rollback-able the same way.

MEASURED at tau=0.5 (400 distinct real commands + the 68-probe gate):

                      severe probes   benign controls   real firing   gate
    v8                    34/55            0/7             5.3%        5/23
    v9 tau=0 (as shipped) 44/55            2/7            21.0%       14/23
    v9 tau=0.5            43/55            0/7             4.0%       14/23

tau=0.5 dominates the shipped v9 on every axis but one severe probe
(`dd if=/dev/zero of=/dev/sda`), and dominates v8 on every axis.

    python3 training/v9/severity_probes/calibrate_prior.py \
        --model .models/kube_ordinal_v9_candidate \
        --train training/v9/train.jsonl --tau 0.5 \
        --out .models/kube_ordinal_v9_cal
"""
import argparse
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path


def class_prior(train_jsonl: Path, labels):
    """Empirical training prior, in the checkpoint's own label order."""
    counts = Counter()
    with open(train_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                counts[json.loads(line)['label']] += 1
    missing = [l for l in labels if l not in counts]
    if missing:
        raise SystemExit(f'label(s) absent from {train_jsonl}: {missing} -- '
                         'refusing to invent a prior for a class we never trained on')
    n = sum(counts.values())
    return [counts[l] / n for l in labels], dict(counts)


def calibrated_bias(bias, prior, tau):
    """b' = b - tau*log(prior). A per-class constant; exactly p/prior^tau."""
    return [b - tau * math.log(p) for b, p in zip(bias, prior)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, type=Path)
    ap.add_argument('--train', required=True, type=Path, help='the split the head was TRAINED on')
    ap.add_argument('--tau', type=float, default=0.5)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()

    if not 0.0 <= args.tau <= 3.0:
        raise SystemExit(f'--tau {args.tau} outside the swept range [0, 3]; '
                         'nothing has been measured there')

    from safetensors.torch import load_file, save_file
    import torch

    cfg = json.loads((args.model / 'config.json').read_text())
    id2label = {int(k): v for k, v in cfg['id2label'].items()}
    labels = [id2label[i] for i in sorted(id2label)]

    prior, counts = class_prior(args.train, labels)
    sd = load_file(args.model / 'model.safetensors')
    b = sd['classifier.bias'].tolist()
    if len(b) != len(labels):
        raise SystemExit(f'bias has {len(b)} entries, config declares {len(labels)} labels')

    new_b = calibrated_bias(b, prior, args.tau)
    sd['classifier.bias'] = torch.tensor(new_b, dtype=sd['classifier.bias'].dtype)

    args.out.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(args.out / 'model.safetensors'))
    for name in ('config.json', 'tokenizer.json', 'LICENSE'):
        src = args.model / name
        if src.exists():
            shutil.copy2(src, args.out / name)

    print(f'labels     {labels}')
    print(f'counts     {counts}')
    print(f'prior      {[round(p, 4) for p in prior]}')
    print(f'tau        {args.tau}')
    print(f'bias       {[round(x, 4) for x in b]}')
    print(f'  ->       {[round(x, 4) for x in new_b]}')
    print(f'\nwrote {args.out}  (trunk tensors untouched; ONLY classifier.bias differs)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
