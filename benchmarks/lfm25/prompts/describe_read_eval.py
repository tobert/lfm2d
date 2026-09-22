#!/usr/bin/env python3
"""Describe-then-read (`POST /v1/opinion`) over a generative run's exact inputs.

The verdict-first opinion read carries nothing on this checkpoint (F9: AUC ~0.5
at 99% mass); the generative path's own verdict slot, read AFTER the model has
described the command, is the judgement (slot score AUC 0.74). `/v1/opinion`
serves that slot directly: it generates the fields before the question under the
grammar, stops at the question's value, and teacher-forces every option there.

This harness pairs on a finished verdict_eval run (--pair) so the only thing that
differs between the arms is the endpoint: the paired row's `input` is split back
into the facts block and the command exactly as `render_input` joined them, and
sent as the opinion's `state`. Beside the read it records what the daemon
described, whether that equals the paired run's parsed report field for field,
and how far the option's raw first-token logprob sits from the paired run's
`verdict_top` -- the same slot, read by two paths.

With --second-pass every row is sent again after the first pass, so the
described-cache hit path is measured on the same bytes.

No threshold is picked here. Rows land in the shape opinion_eval writes, so
score_instruments.py scores them as an opinion run, and every option also
carries `first_logprob` for the raw first-token cut the F9 slot score used.
"""
import argparse, hashlib, json, math, statistics, subprocess, time
import urllib.error, urllib.request
from collections import Counter
from pathlib import Path

import opinion_eval as O
import verdict_eval as V  # eval_out: never write rows into the repo

COMMAND_MARK = 'Command:\n'


def split_state(pair):
    """`render_input` wrote `{facts}{evidence}Command:\\n{text}`; hand the daemon
    the same two parts. Loud if the row was rendered some other way."""
    text, sent = pair['text'], pair['input']
    tail = COMMAND_MARK + text
    if not sent.endswith(tail):
        raise SystemExit(f'paired input does not end with Command:\\n<text>: {sent[-80:]!r}')
    facts = sent[:-len(tail)]
    return {'command': text, **({'facts': facts} if facts else {})}


def paired_raw_first(pair, option):
    top = pair.get('verdict_top') or []
    for text, logprob in top:
        if text == option:
            return logprob
    return None


def read_record(pair, resp, seconds, field, pass_option):
    rec = {k: pair.get(k) for k in ('text', 'label', 'gold')}
    rec['input_sha256'] = hashlib.sha256(pair['input'].encode()).hexdigest()
    rec['paired_verdict'] = pair.get('verdict')
    rec['paired_ms'] = round((pair.get('prefill_ms') or 0) + (pair.get('decode_ms') or 0), 1)
    rec['seconds'] = seconds
    if 'http_error' in resp:
        rec['outcome'] = 'http_error'
        rec['error'] = resp
        return rec
    answer = resp['answers'][0]
    rec['outcome'] = 'read'
    rec['field'] = answer['field']
    rec['options'] = {o['option']: {'logprob': o['logprob'], 'first_logprob': o['first_logprob'],
                                    'prob': o['prob'], 'tokens': o['tokens']}
                      for o in answer['options']}
    for k in ('sequence_mass', 'first_token_mass', 'margin', 'shared_tokens', 'scored_tokens',
              'rendered_sha256'):
        rec[k] = answer[k]
    rec['described'] = resp['described']
    rec['cache'] = resp['cache']
    for k in ('prompt_tokens', 'cached_tokens', 'described_tokens', 'queue_ms', 'prefill_ms',
              'describe_ms', 'read_ms'):
        rec[k] = resp[k]
    # opinion_eval.summarize reads prefill_ms + decode_ms; here the decode is the
    # description plus the read.
    rec['decode_ms'] = resp['describe_ms'] + resp['read_ms']
    report = pair.get('report') or {}
    rec['described_matches_paired'] = {
        d['field']: (d['value'] == report.get(d['field'])) if d['field'] in report else None
        for d in resp['described']}
    ours = rec['options'].get(pass_option, {}).get('first_logprob')
    theirs = paired_raw_first(pair, pass_option)
    rec['paired_first_logprob'] = theirs
    rec['first_logprob_gap'] = None if ours is None or theirs is None else round(abs(ours - theirs), 5)
    return rec


def summarize(results, second, allow, stop, mass_floor):
    out = O.summarize(results, allow, stop, mass_floor)
    read = [x for x in results if x['outcome'] == 'read']
    fields = sorted({f for x in read for f in x['described_matches_paired']})
    out['described_matches_paired'] = {
        f: [sum(x['described_matches_paired'].get(f) is True for x in read),
            sum(x['described_matches_paired'].get(f) is not None for x in read)]
        for f in fields}
    gaps = [x['first_logprob_gap'] for x in read if x['first_logprob_gap'] is not None]
    out['first_logprob_gap_vs_paired_slot'] = {
        'rows': len(gaps), **O.quantiles(gaps, (0.5, 0.9, 1.0))} if gaps else None
    out['cache_outcomes'] = dict(Counter(f"{x['cache']['state']}/{x['cache']['described']}" for x in read))
    out['latency_ms']['describe'] = O.quantiles([x['describe_ms'] for x in read])
    out['latency_ms']['read'] = O.quantiles([x['read_ms'] for x in read])
    if second:
        hits = [x for x in second if x['outcome'] == 'read']
        out['second_pass'] = {
            'rows': len(hits),
            'cache_outcomes': dict(Counter(x['cache']['described'] for x in hits)),
            'latency_ms': O.quantiles([x['prefill_ms'] + x['decode_ms'] for x in hits]),
            'identical_reads': sum(
                a['options'] == b['options'] and a['sequence_mass'] == b['sequence_mass']
                for a, b in zip(read, hits) if a['input_sha256'] == b['input_sha256']),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--binary', type=Path, required=True)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--tokenizer', type=Path, required=True)
    ap.add_argument('--prompt', type=Path, required=True,
                    help='the spec with the describe fields and the question (its stem names it)')
    ap.add_argument('--pair', type=Path, required=True, help="a verdict_eval run's rows.jsonl")
    ap.add_argument('--field', default='verdict', help="the spec's choice field to read")
    ap.add_argument('--name', help="output subdirectory; default is the prompt stem + '-describe-read'")
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--port', type=int, default=18155)
    ap.add_argument('--device', default='rocm')
    ap.add_argument('--pass-verdict', default='allow')
    ap.add_argument('--stop-verdict', default='ask')
    ap.add_argument('--mass-floor', type=float)
    ap.add_argument('--second-pass', action='store_true',
                    help='send every row again after the first pass to measure the described-cache hit')
    a = ap.parse_args()
    spec = json.loads(a.prompt.read_text())
    schema = spec.get('output_schema') or {}
    options = (schema.get('properties', {}).get(a.field) or {}).get('enum')
    if not options:
        raise SystemExit(f'{a.prompt.name} has no choice field {a.field!r}')
    for verdict in (a.pass_verdict, a.stop_verdict):
        if verdict not in options:
            raise SystemExit(f'{verdict!r} is not one of {a.field}\'s options {options}')
    out = V.eval_out(a.out) / (a.name or f'{a.prompt.stem}-describe-read')
    out.mkdir(parents=True, exist_ok=False)
    pairs = [json.loads(l) for l in a.pair.read_text().splitlines() if l.strip()]
    if a.limit:
        pairs = pairs[:a.limit]
    missing = [i for i, p in enumerate(pairs) if not p.get('input')]
    if missing:
        raise SystemExit(f'{len(missing)} paired rows carry no input bytes, first at {missing[0]}')
    states = [split_state(p) for p in pairs]

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
    body = lambda state: {'spec': a.prompt.stem, 'state': state, 'questions': [{'field': a.field}]}
    results, second = [], []
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
            menu = rpc('/v1/opinion/specs')
            print('prefix', json.dumps(info), flush=True)
            print('menu', json.dumps(menu), flush=True)
            for sink_name, results_list in (('rows.jsonl', results),) + \
                    ((('second-pass.jsonl', second),) if a.second_pass else ()):
                with (out / sink_name).open('x') as sink:
                    for n, (p, state) in enumerate(zip(pairs, states), 1):
                        began = time.time()
                        try:
                            resp = rpc('/v1/opinion', body(state))
                        except urllib.error.HTTPError as e:
                            resp = {'http_error': e.code, 'body': e.read().decode()[:400]}
                        rec = read_record(p, resp, round(time.time() - began, 3), a.field, a.pass_verdict)
                        results_list.append(rec)
                        sink.write(json.dumps(rec) + '\n')
                        sink.flush()
                        if n % 100 == 0:
                            print(f'  {sink_name} {n}/{len(pairs)}', flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()

    summary = {'prompt': a.prompt.name,
               'prompt_sha256': hashlib.sha256(a.prompt.read_bytes()).hexdigest(),
               'field': a.field, 'pair': str(a.pair), 'prefix': info, 'menu': menu,
               **summarize(results, second, a.pass_verdict, a.stop_verdict, a.mass_floor)}
    (out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('prefix', 'menu')}, indent=1))
    print('wrote', out)


if __name__ == '__main__':
    main()
