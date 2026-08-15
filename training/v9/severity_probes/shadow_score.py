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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_candidate import load, classify_batch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, type=Path)
    ap.add_argument('--base', default='.models/LFM2.5-Encoder-350M', type=Path)
    ap.add_argument('--log', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--poll-seconds', type=float, default=5.0)
    args = ap.parse_args()

    log_path = args.log.expanduser()
    out_path = args.out.expanduser()

    trunk, tok, W, b, labels, device = load(args.model, args.base)
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
                        out.write(json.dumps({
                            'ts': time.time(),
                            'command': r['command'],
                            'v8_verdict': live,
                            'candidate_verdict': pred,
                            'agree': pred == live,
                        }) + '\n')
                os.chmod(out_path, 0o600)
                print(f'[shadow] scored {len(rows)} new row(s), '
                      f'{sum(p == r["lfm2d"]["top"] for r, p in zip(rows, preds))} agree',
                      flush=True)
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
