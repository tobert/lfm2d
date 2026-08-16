#!/usr/bin/env python3
"""Forward-looking shadow scorer: tail the advisory log, score new rows with
a candidate checkpoint, log agree/disagree locally. Never touches the live
pod or any client-visible response -- pure local observation.

Privacy: same contract as the advisory log itself (0600, local-only, real
command text stays on this box, never committed to git). The output file is
chmod 0600 immediately on creation.

    .venv-train/bin/python training/v9/severity_probes/shadow_score.py \
        --model .models/kube_ordinal_v9_candidate \
        --log ~/.cache/claude-hooks/lfm2d-advisory.jsonl \
        --out ~/.cache/claude-hooks/lfm2d-v9-candidate-shadow.jsonl
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
# makes idle workers sleep; 1 thread is ample for a 350M encoder at this rate.
os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')
os.environ.setdefault('OMP_NUM_THREADS', '1')

# NOT done, because it was measured and bought nothing: hiding the GPU with
# HIP_VISIBLE_DEVICES='' before the import. A cpu run still opens /dev/kfd --
# the checkpoint's trust_remote_code path touches torch.cuda during
# from_pretrained, and ROCm opens the KFD node to enumerate regardless of
# visibility masking. But `rocm-smi --showpids` reports the process at
# **VRAM USED 0**, and loading the model with the mask on vs off gives a VRAM
# delta of exactly 0. The fd is cosmetic; the core was the whole cost.

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
from backtest_candidate import VALID_DEVICES, load, classify_batch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, type=Path)
    ap.add_argument('--base', default='.models/LFM2.5-Encoder-350M', type=Path)
    ap.add_argument('--log', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--poll-seconds', type=float, default=5.0)
    ap.add_argument('--device', choices=VALID_DEVICES, default='cpu',
                    help="default 'cpu' ON PURPOSE: this is a poll loop that "
                         "scores a handful of rows an hour, and the ROCm HIP "
                         "runtime busy-waits a thread at 100%% of a core for "
                         "the whole run. Measured 2026-08-16: 13.5 core-hours "
                         "burned for 260 forward passes, GPU util 0%%. Pass "
                         "--device auto only if you know why you want it.")
    ap.add_argument('--threads', type=int, default=1,
                    help='ATen intra-op threads (default 1; see the OMP note above)')
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)

    log_path = args.log.expanduser()
    out_path = args.out.expanduser()

    trunk, tok, W, b, labels, device = load(args.model, args.base, args.device)
    print(f'[shadow] loaded {args.model.name} on {device}', flush=True)

    if not out_path.exists():
        out_path.touch()
    os.chmod(out_path, 0o600)

    # Start at end of file -- forward-looking only, per the request.
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
            lines = [l for l in chunk.splitlines() if l.strip()]
            rows = []
            for line in lines:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                l = r.get('lfm2d', {})
                if l.get('ok') and l.get('top') and r.get('command'):
                    rows.append(r)
            if rows:
                texts = [r['command'] for r in rows]
                preds = classify_batch(trunk, tok, W, b, labels, device, texts)
                with open(out_path, 'a') as out:
                    for r, pred in zip(rows, preds):
                        live = r['lfm2d']['top']
                        # SELF-DESCRIBING field names, and they must stay that
                        # way. These were once 'v8_verdict'/'candidate_verdict',
                        # which silently INVERTED on 2026-08-16: production
                        # became v9 and this scorer was repointed at v8, so a
                        # field called 'v8_verdict' was carrying v9's answer.
                        # Mislabelled data is data corruption, and a reader has
                        # no way to notice. Record which model said what.
                        out.write(json.dumps({
                            'ts': time.time(),
                            'command': r['command'],
                            'live_model': r['lfm2d'].get('model_id'),
                            'live_verdict': live,
                            'shadow_model': args.model.name,
                            'shadow_verdict': pred,
                            'agree': pred == live,
                        }) + '\n')
                os.chmod(out_path, 0o600)
                print(f'[shadow] scored {len(rows)} new row(s), '
                      f'{sum(p == r["lfm2d"]["top"] for r, p in zip(rows, preds))} agree',
                      flush=True)
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
