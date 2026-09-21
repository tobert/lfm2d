#!/usr/bin/env python3
"""System-1 opinion reads over a generative run's exact inputs, on OUR stack.

Amy, 2026-09-21: "get to a system 1 that's fast enough we hand it the command
and get back an allow/deny bool-ish" ... "main thing is that the undesired
*allow* is low, but I want to see how it fails before deciding thresholds."

So this harness picks no threshold. It sends every row of a finished
verdict_eval run (--pair) through POST /v1/adjudicate with `opinion: true` and
records the whole read: each option's raw sequence logprob, the renormalized
probabilities, the raw mass on the answer set, and the prompt hash the daemon
rendered. The report is a distribution, per gold verdict, beside the generative
verdict the paired run wrote for the same bytes.

Pairing, not re-rendering: the inputs are the paired run's `input` field
verbatim, so the only thing that differs between the arms is how the answer is
read. The facts are therefore whatever build_facts produced for that run, not
today's -- say which run in any result you cite.

val_F is a smoke check for the LLM (CLAUDE.md, "Separate data"). Nothing here is
a scorecard; F9's two instruments are.
"""
import argparse, hashlib, json, math, statistics, subprocess, time
import urllib.error, urllib.request
from collections import Counter
from pathlib import Path

import verdict_eval as V  # eval_out: never write rows into the repo


def quantiles(values, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    v = sorted(values)
    if not v:
        return {}
    return {f'p{int(q * 100)}': round(v[min(len(v) - 1, int(q * len(v)))], 4) for q in qs}


def read_record(pair, resp, seconds):
    rec = {k: pair.get(k) for k in ('text', 'label', 'gold')}
    rec['input_sha256'] = hashlib.sha256(pair['input'].encode()).hexdigest()
    rec['paired_verdict'] = pair.get('verdict')
    rec['paired_ms'] = round((pair.get('prefill_ms') or 0) + (pair.get('decode_ms') or 0), 1)
    rec['seconds'] = seconds
    if 'http_error' in resp:
        rec['outcome'] = 'http_error'
        rec['error'] = resp
        return rec
    op = resp['opinion']
    rec['outcome'] = 'read'
    rec['options'] = {o['option']: {'logprob': o['logprob'], 'prob': o['prob'],
                                    'tokens': o['tokens']} for o in op['options']}
    for k in ('sequence_mass', 'first_token_mass', 'shared_tokens', 'scored_tokens',
              'rendered_sha256'):
        rec[k] = op[k]
    for k in ('prompt_tokens', 'cached_tokens', 'prefill_ms', 'decode_ms', 'queue_ms'):
        rec[k] = resp[k]
    return rec


def summarize(results, allow='allow'):
    read = [x for x in results if x['outcome'] == 'read']
    out = {'rows': len(results), 'outcomes': dict(Counter(x['outcome'] for x in results))}
    for gold in sorted({x['gold'] for x in read}):
        g = [x for x in read if x['gold'] == gold]
        out[f'gold_{gold}'] = {
            'n': len(g),
            'p_allow': quantiles([x['options'][allow]['prob'] for x in g]),
            'sequence_mass': quantiles([x['sequence_mass'] for x in g]),
            'argmax': dict(Counter(max(x['options'], key=lambda o: x['options'][o]['prob'])
                                   for x in g)),
            'paired_verdict': dict(Counter(x['paired_verdict'] for x in g)),
        }
    # The curve Amy asked to see before any threshold: for each cutoff on
    # P(allow), how many gold-ask rows would be let through (the undesired
    # allow) and how many gold-allow rows would be stopped (the false alarm).
    ask = [x['options'][allow]['prob'] for x in read if x['gold'] == 'ask']
    ok = [x['options'][allow]['prob'] for x in read if x['gold'] == 'allow']
    curve = []
    for cut in (0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999):
        curve.append({'allow_if_p_allow_at_least': cut,
                      'undesired_allow': [sum(p >= cut for p in ask), len(ask)],
                      'false_alarm': [sum(p < cut for p in ok), len(ok)]})
    out['curve'] = curve
    if len(ask) and len(ok):
        # Rank AUC of "gold ask" vs "gold allow" by 1 - P(allow).
        wins = sum((a < o) + 0.5 * (a == o) for a in ask for o in ok)
        out['auc_ask_vs_allow'] = round(wins / (len(ask) * len(ok)), 4)
    ms = [x['prefill_ms'] + x['decode_ms'] for x in read]
    out['latency_ms'] = {'read': quantiles(ms), 'prefill': quantiles([x['prefill_ms'] for x in read]),
                         'score': quantiles([x['decode_ms'] for x in read]),
                         'paired_generation': quantiles([x['paired_ms'] for x in read])}
    out['rendered_sha256_distinct'] = len({x['rendered_sha256'] for x in read})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--binary', type=Path, required=True)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--tokenizer', type=Path, required=True)
    ap.add_argument('--prompt', type=Path, required=True, help='a spec with an `opinion` block')
    ap.add_argument('--pair', type=Path, required=True, help="a verdict_eval run's rows.jsonl")
    ap.add_argument('--name', help="output subdirectory; default is the prompt file's stem")
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--port', type=int, default=18154)
    ap.add_argument('--device', default='rocm')
    a = ap.parse_args()
    spec = json.loads(a.prompt.read_text())
    if 'opinion' not in spec:
        raise SystemExit(f'{a.prompt.name} has no opinion block')
    out = V.eval_out(a.out) / (a.name or a.prompt.stem)
    out.mkdir(parents=True, exist_ok=False)
    pairs = [json.loads(l) for l in a.pair.read_text().splitlines() if l.strip()]
    if a.limit:
        pairs = pairs[:a.limit]
    missing = [i for i, p in enumerate(pairs) if not p.get('input')]
    if missing:
        raise SystemExit(f'{len(missing)} paired rows carry no input bytes, first at {missing[0]}')

    address = f'http://127.0.0.1:{a.port}'

    def rpc(path, payload=None):
        req = urllib.request.Request(address + path,
                                     None if payload is None else json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=130) as r:
            body = r.read().decode()
            return json.loads(body) if body.startswith(('{', '[')) else body

    command = [str(a.binary), '--adjudicator-model', str(a.model),
               '--adjudicator-tokenizer', str(a.tokenizer),
               '--adjudicator-prompt', str(a.prompt), '--adjudicator-context', '4096',
               '--device', a.device, '--bind-addr', f'127.0.0.1:{a.port}', '--threads', '8']
    results = []
    with (out / 'daemon.log').open('x') as log:
        proc = subprocess.Popen(command, stdout=log, stderr=log)
        try:
            for _ in range(240):
                if proc.poll() is not None:
                    raise SystemExit(f'daemon exited {proc.returncode}; see {out}/daemon.log')
                try:
                    if rpc('/readyz') == 'ready':
                        break
                except (urllib.error.URLError, TimeoutError, ConnectionError):
                    pass
                time.sleep(.5)
            else:
                raise SystemExit('daemon did not become ready')
            info = rpc('/v1/adjudicator')
            print('prefix', json.dumps(info), flush=True)
            with (out / 'rows.jsonl').open('x') as sink:
                for n, p in enumerate(pairs, 1):
                    began = time.time()
                    try:
                        resp = rpc('/v1/adjudicate', {'input': p['input'], 'opinion': True})
                    except urllib.error.HTTPError as e:
                        resp = {'http_error': e.code, 'body': e.read().decode()[:400]}
                    rec = read_record(p, resp, round(time.time() - began, 3))
                    results.append(rec)
                    sink.write(json.dumps(rec) + '\n')
                    sink.flush()
                    if n % 100 == 0:
                        print(f'  {n}/{len(pairs)}', flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()

    summary = {'prompt': a.prompt.name,
               'prompt_sha256': hashlib.sha256(a.prompt.read_bytes()).hexdigest(),
               'pair': str(a.pair), 'prefix': info, **summarize(results)}
    (out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'prefix'}, indent=1))
    print('wrote', out)


if __name__ == '__main__':
    main()
