#!/usr/bin/env python3
"""Replay the advisory log through an exported checkpoint, offline.

Same load path as score_local.py, but scores arbitrary real command text
from the advisory JSONL instead of the fixed probe set, and compares
against each row's already-recorded live v8 verdict.

Privacy: the advisory log is 0600, local-only, real command text (per
memory guard-evasion-by-reformulation / the project's own advisory-hook
privacy contract). This script reads it but NEVER writes raw command text
anywhere -- output is aggregate counts only, matching validate_v8.py's own
"aggregates only, never a raw row" policy. Do not add per-row text logging
to this script without re-reading that constraint.

    .venv-train/bin/python training/v9/severity_probes/backtest_candidate.py \
        --model .models/kube_ordinal_v9_candidate \
        --log ~/.cache/claude-hooks/lfm2d-advisory.jsonl
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel, PreTrainedTokenizerFast


def load(model_dir: Path, base_dir: Path):
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    id2label = {int(k): v for k, v in cfg.id2label.items()}
    labels = [id2label[i] for i in sorted(id2label)]
    tok = PreTrainedTokenizerFast(tokenizer_file=str(model_dir / 'tokenizer.json'))
    if tok.pad_token is None:
        tok.pad_token = '<|pad|>'  # base checkpoint's own pad token (config.json)
    from safetensors.torch import load_file
    sd = load_file(model_dir / 'model.safetensors')
    trunk = AutoModel.from_pretrained(base_dir, trust_remote_code=True)
    trunk_sd = {k[len('lfm2.'):]: v for k, v in sd.items() if k.startswith('lfm2.')}
    trunk.load_state_dict(trunk_sd, strict=False)
    trunk.eval()
    W = sd['classifier.weight'].float()
    b = sd['classifier.bias'].float()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trunk.to(device)
    W, b = W.to(device), b.to(device)
    return trunk, tok, W, b, labels, device


@torch.no_grad()
def classify_batch(trunk, tok, W, b, labels, device, texts, max_len=128):
    enc = tok(texts, return_tensors='pt', truncation=True, max_length=max_len, padding=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    out = trunk(**enc)
    h = out.last_hidden_state[:, 0, :].float()
    logits = h @ W.T + b
    probs = torch.softmax(logits, dim=-1)
    tops = probs.argmax(dim=-1).tolist()
    return [labels[i] for i in tops]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, type=Path)
    ap.add_argument('--base', default='.models/LFM2.5-Encoder-350M', type=Path)
    ap.add_argument('--log', required=True, type=Path)
    ap.add_argument('--batch-size', type=int, default=32)
    args = ap.parse_args()

    trunk, tok, W, b, labels, device = load(args.model, args.base)
    print(f'loaded {args.model.name} on {device}, labels={labels}', file=sys.stderr)

    rows = []
    with open(Path(args.log).expanduser()) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            l = r.get('lfm2d', {})
            if not l.get('ok') or not l.get('top') or not r.get('command'):
                continue
            rows.append(r)
    print(f'{len(rows)} rows with a live v8 verdict', file=sys.stderr)

    agree = disagree = 0
    confusion = Counter()
    exo_total = exo_disagree = 0
    worktree_total = worktree_disagree = 0
    pkgmgr_total = pkgmgr_disagree = 0
    exo_confusion = Counter()
    worktree_confusion = Counter()
    pkgmgr_confusion = Counter()

    for i in range(0, len(rows), args.batch_size):
        batch = rows[i:i + args.batch_size]
        texts = [r['command'] for r in batch]
        preds = classify_batch(trunk, tok, W, b, labels, device, texts)
        for r, pred in zip(batch, preds):
            live = r['lfm2d']['top']
            is_exo = 'exomemory' in r.get('cwd', '')
            is_worktree = 'worktree remove' in r['command']
            is_pkgmgr = any(v in r['command'] for v in (
                ' install', ' add ', ' update', ' upgrade', 'pip-compile',
                'apt-get install', 'apt install', 'dnf install', 'pacman -S',
                'brew install', 'conda install', 'apk add', 'snap install'))
            if is_exo:
                exo_total += 1
            if is_worktree:
                worktree_total += 1
            if is_pkgmgr:
                pkgmgr_total += 1
            if pred == live:
                agree += 1
            else:
                disagree += 1
                confusion[(live, pred)] += 1
                if is_exo:
                    exo_disagree += 1
                    exo_confusion[(live, pred)] += 1
                if is_worktree:
                    worktree_disagree += 1
                    worktree_confusion[(live, pred)] += 1
                if is_pkgmgr:
                    pkgmgr_disagree += 1
                    pkgmgr_confusion[(live, pred)] += 1
        if (i // args.batch_size) % 20 == 0:
            print(f'  ...{i + len(batch)}/{len(rows)}', file=sys.stderr)

    total = agree + disagree
    print(f'\n=== backtest: {args.model.name} vs live v8 verdicts, {total} rows ===')
    print(f'agree: {agree} ({agree/total:.1%})   disagree: {disagree} ({disagree/total:.1%})')
    print('\nconfusion (v8_live -> candidate), most common first:')
    for (live, pred), n in confusion.most_common(15):
        print(f'  {n:5d}  {live:<18} -> {pred}')

    print(f'\nexomemory cwd rows: {exo_total}  disagree: {exo_disagree} '
          f'({exo_disagree/max(1,exo_total):.1%})')
    for (live, pred), n in exo_confusion.most_common(8):
        print(f'    {n:5d}  {live:<18} -> {pred}')
    print(f'\ngit worktree remove rows: {worktree_total}  disagree: {worktree_disagree} '
          f'({worktree_disagree/max(1,worktree_total):.1%})')
    for (live, pred), n in worktree_confusion.most_common(8):
        print(f'    {n:5d}  {live:<18} -> {pred}')
    print(f'\npackage-manager-shaped rows: {pkgmgr_total}  disagree: {pkgmgr_disagree} '
          f'({pkgmgr_disagree/max(1,pkgmgr_total):.1%})')
    for (live, pred), n in pkgmgr_confusion.most_common(8):
        print(f'    {n:5d}  {live:<18} -> {pred}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
