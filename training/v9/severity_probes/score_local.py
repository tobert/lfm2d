#!/usr/bin/env python3
"""Score the severity probes against an EXPORTED checkpoint, offline.

score_probes.py talks to a running lfm2d. That is the right thing for a
deployed model, but it cannot answer "did this training run help?" without a
deploy first. This loads an exported classifier directly and emits the same
JSON that `score_probes.py --results` consumes, so the gate is identical and
only the transport differs.

    .venv-train/bin/python training/v9/severity_probes/score_local.py \\
        --model .models/kube_ordinal_v9 --save baseline_v9_local.json
    python3 training/v9/severity_probes/score_probes.py --results <that file>

Pooling is CLS (hidden state at position 0), matching both the exporter and
the Rust loader. Getting this wrong would silently produce plausible garbage,
so it is asserted against the exported config rather than assumed.

Committed with the numbers it produces, per commit-the-scorer.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel, PreTrainedTokenizerFast

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--base', default='.models/LFM2.5-Encoder-350M',
                    help='base dir supplying modeling_lfm2_bidirectional.py; the '
                         'export deliberately omits it (the Rust loader does not '
                         'need it, but transformers trust_remote_code does)')
    ap.add_argument('--probes', default=str(HERE / 'probes.jsonl'))
    ap.add_argument('--save', required=True)
    ap.add_argument('--max-len', type=int, default=128)
    args = ap.parse_args()

    mdir = Path(args.model)
    cfg = AutoConfig.from_pretrained(mdir, trust_remote_code=True)
    id2label = {int(k): v for k, v in cfg.id2label.items()}
    labels = [id2label[i] for i in sorted(id2label)]
    if 'data-critical' not in labels:
        raise SystemExit(f'unexpected label set {labels}')

    # AutoTokenizer cannot resolve this config -- the export overrides
    # `architectures`, and Lfm2Config has no entry in TOKENIZER_MAPPING. The
    # exporter copies tokenizer.json verbatim, so load that file directly.
    tok = PreTrainedTokenizerFast(tokenizer_file=str(mdir / 'tokenizer.json'))
    from safetensors.torch import load_file
    sd = load_file(mdir / 'model.safetensors')
    if 'classifier.weight' not in sd:
        raise SystemExit('no classifier head in the export — wrong directory?')

    # Build the architecture from the BASE dir (it carries the custom modeling
    # file), then load the fine-tuned trunk weights out of the export. The
    # export's `lfm2.` prefix is the Rust loader's naming, stripped here.
    trunk = AutoModel.from_pretrained(args.base, trust_remote_code=True)
    trunk_sd = {k[len('lfm2.'):]: v for k, v in sd.items() if k.startswith('lfm2.')}
    missing, unexpected = trunk.load_state_dict(trunk_sd, strict=False)
    real_missing = [k for k in missing if not k.startswith('classifier')]
    if real_missing:
        raise SystemExit(f'trunk weights missing from export: {real_missing[:5]}')
    print(f'loaded {len(trunk_sd)} trunk tensors from the export '
          f'({len(unexpected)} unexpected)', file=sys.stderr)
    trunk.eval()
    W = sd['classifier.weight'].float()
    b = sd['classifier.bias'].float()

    probes = [json.loads(l) for l in Path(args.probes).read_text().splitlines() if l.strip()]
    results = {}
    with torch.no_grad():
        for p in probes:
            enc = tok(p['cmd'], return_tensors='pt', truncation=True, max_length=args.max_len)
            out = trunk(**enc)
            h = out.last_hidden_state[:, 0, :].float()   # CLS, matching the exporter
            logits = h @ W.T + b
            probs = torch.softmax(logits, dim=-1)[0]
            scores = {labels[i]: float(probs[i]) for i in range(len(labels))}
            results[p['id']] = {'top': max(scores, key=scores.get), 'scores': scores}

    run = {'meta': {'model_id': mdir.name, 'weight_hash': 'local-export'},
           'results': results}
    Path(args.save).write_text(json.dumps(run, indent=1) + '\n')
    print(f'scored {len(results)} probes -> {args.save}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
