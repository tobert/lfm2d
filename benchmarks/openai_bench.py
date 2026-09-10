#!/usr/bin/env python3
"""Run the typed adjudicator corpus against an OpenAI-compatible endpoint.

    python3 benchmarks/openai_bench.py \
        --base-url http://localhost:2031/v1 --label lfm25 \
        --output /tmp/adj-lfm25

The point of comparison is `benchmarks/diffusiongemma/`: same cases, same
scorer (`structured_accuracy.check_tool_output`), so the expected-field
numbers here sit directly beside DiffusionGemma's. What differs is the
runtime -- an autoregressive server rather than a canvas denoiser.

Stdlib only, matching the rest of this tree. No `requests`, no SDK.

WHAT IS AND IS NOT COMPARABLE
-----------------------------
Comparable: expected-field pass counts, because the scorer is the same
file and the cases are the same bytes (both hashes are recorded).

NOT comparable without saying so: latency. These servers are warm and
resident; a first request after start pays model load, so `--warmup`
requests are recorded and excluded like the diffusion harness does.
Different backends quantize differently (nvfp4 on vllm vs Q5_K_M on
llama.cpp), context lengths differ, and one may be on another machine
over the network -- so a latency number here is "this endpoint as
configured today", not a property of the model. Provenance records the
base URL, the server-reported model id, and every knob passed.

`tool_choice` is "auto", matching what the DiffusionGemma driver sent
(ToolChoice::Auto). Forcing the call with "required" would measure a
different thing -- whether the model can fill a schema it was told to
fill, rather than whether it decides to report at all -- so if you change
it, say so in the label.

Reasoning content is RECORDED, never stripped and never assumed absent:
a thinking model that spends its budget before answering is a real cost
and shows up here as tokens and seconds rather than as a mystery.
"""
import argparse
import json
import hashlib
import random
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'diffusiongemma'))
from structured_accuracy import check_tool_output  # noqa: E402

DEFAULT_CASES = HERE / 'diffusiongemma' / 'tool_cases.jsonl'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def post(url, payload, timeout, api_key=None):
    """One POST. Returns (parsed json, elapsed seconds) or raises."""
    body = json.dumps(payload).encode()
    headers = {'content-type': 'application/json'}
    if api_key:
        headers['authorization'] = f'Bearer {api_key}'
    req = urllib.request.Request(url, data=body, headers=headers, method='POST')
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        parsed = json.loads(resp.read())
    return parsed, time.monotonic() - started


def server_models(base_url, timeout, api_key=None):
    """What the server says it serves. Recorded as provenance, because a
    `--model` we pass and a model the server actually loaded are not the
    same claim."""
    headers = {}
    if api_key:
        headers['authorization'] = f'Bearer {api_key}'
    req = urllib.request.Request(f'{base_url}/models', headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return [m.get('id') for m in json.loads(resp.read()).get('data', [])]
    except Exception as e:
        return [f'unreadable: {type(e).__name__}']


def schedule(cases, warmup, repetitions, order_seed):
    """Warmups first, then `repetitions` shuffled passes over every case.
    Same shape as the diffusion harness: warmups are recorded and excluded
    from the summary, and each repetition is independently shuffled so a
    slow case cannot sit at a fixed position."""
    rng = random.Random(order_seed)
    items = [{'phase': 'warmup', 'repeat': 0, 'case': cases[i % len(cases)]}
             for i in range(warmup)]
    for repeat in range(repetitions):
        shuffled = list(cases)
        rng.shuffle(shuffled)
        items += [{'phase': 'measure', 'repeat': repeat, 'case': c} for c in shuffled]
    return items


def run_case(base_url, model, case, args):
    payload = {
        'messages': case['messages'],
        'tools': case['tools'],
        'tool_choice': args.tool_choice,
        'max_tokens': args.max_tokens or case.get('max_tokens', 256),
        'stream': False,
    }
    if args.temperature != 'server':
        payload['temperature'] = float(args.temperature)
    if model:
        payload['model'] = model
    if args.extra_body:
        payload.update(json.loads(args.extra_body))
    parsed, elapsed = post(f'{base_url}/chat/completions', payload,
                           args.timeout, args.api_key)
    choice = (parsed.get('choices') or [{}])[0]
    message = choice.get('message') or {}
    calls = message.get('tool_calls') or []
    usage = parsed.get('usage') or {}
    return {
        'response_s': elapsed,
        'finish_reason': choice.get('finish_reason'),
        'prompt_tokens': usage.get('prompt_tokens'),
        'completion_tokens': usage.get('completion_tokens'),
        # Prefix-cache signal, when the server reports it. llama.cpp fills
        # prompt_tokens_details.cached_tokens; this vllm build returns
        # prompt_tokens_details as null, and its prefix cache is queried but
        # never hit (vllm:prefix_cache_hits_total stays 0), so a None here
        # means "not reported", NOT "nothing was cached".
        'cached_tokens': ((usage.get('prompt_tokens_details') or {}).get('cached_tokens')
                          if isinstance(usage.get('prompt_tokens_details'), dict) else None),
        # Recorded, not stripped: a thinking model's spend is a real cost.
        'reasoning_chars': len(message.get('reasoning_content') or
                               message.get('reasoning') or ''),
        'content_chars': len(message.get('content') or ''),
        # Bounded, so a zero-call row can be diagnosed rather than guessed at.
        # These are synthetic cases; no live traffic text reaches this file.
        'content': (message.get('content') or '')[:2000],
        'reasoning': (message.get('reasoning_content')
                      or message.get('reasoning') or '')[:2000],
        'tool_calls': calls,
        'check': check_tool_output(case, calls),
        # A call cut off at the token cap is a BUDGET failure, not a judgement
        # failure. The corpus's 256 was sized for the diffusion canvas; a
        # thinking model spends before it answers. Distinguishing these keeps
        # a headroom problem from being read as a model being wrong.
        'truncated': choice.get('finish_reason') == 'length'
        or (usage.get('completion_tokens') or 0) >= payload['max_tokens'],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--base-url', required=True)
    ap.add_argument('--model', default=None,
                    help='model id to request; omit to let the server pick')
    ap.add_argument('--label', required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--cases', type=Path, default=DEFAULT_CASES)
    ap.add_argument('--only', help='comma-separated case ids')
    ap.add_argument('--warmup', type=int, default=2)
    ap.add_argument('--repetitions', type=int, default=3)
    ap.add_argument('--order-seed', type=int, default=4242)
    ap.add_argument('--timeout', type=float, default=300.0)
    ap.add_argument('--max-tokens', type=int, default=None)
    ap.add_argument('--temperature', default='0.0',
                    help="number, or 'server' to omit the field entirely and let "
                         "the server's configured sampling stand. A request-level "
                         "temperature OVERRIDES the server default; other "
                         "server-side samplers (top-k, repeat-penalty) still "
                         "apply either way, so 'temperature: 0.0' against a "
                         "server started with --temp 0.2 --top-k 80 "
                         "--repeat-penalty 1.05 measures a HYBRID, not the "
                         "deployed configuration. Use 'server' for the latter.")
    ap.add_argument('--api-key', default=None)
    ap.add_argument('--tool-choice', default='auto',
                    choices=('auto', 'required', 'none'),
                    help="'auto' matches what the DiffusionGemma driver sent, and "
                         "measures whether the model DECIDES to report. 'required' "
                         "forces the call, which is what a production adjudicator "
                         "would do -- it never wants prose. These measure different "
                         "things; the label should say which.")
    ap.add_argument('--extra-body', default=None,
                    help='JSON object merged into the request payload, e.g. '
                         '\'{"chat_template_kwargs":{"enable_thinking":false}}\'. '
                         'Recorded in provenance: it changes what is measured.')
    args = ap.parse_args(argv)

    base_url = args.base_url.rstrip('/')
    cases = [json.loads(l) for l in args.cases.read_text().splitlines() if l.strip()]
    if args.only:
        wanted = {x.strip() for x in args.only.split(',')}
        cases = [c for c in cases if c['id'] in wanted]
    if not cases:
        raise SystemExit('no cases selected')

    args.output.mkdir(parents=True, exist_ok=False)
    items = schedule(cases, args.warmup, args.repetitions, args.order_seed)

    metadata = {
        'type': 'metadata', 'schema_version': 1, 'label': args.label,
        'base_url': base_url, 'requested_model': args.model,
        'server_models': server_models(base_url, args.timeout, args.api_key),
        'cases': str(args.cases), 'cases_sha256': sha256(args.cases),
        'harness_sha256': sha256(__file__),
        'scorer_sha256': sha256(HERE / 'diffusiongemma' / 'structured_accuracy.py'),
        'tool_choice': args.tool_choice, 'temperature': args.temperature,
        'sampling_note': ('request temperature omitted; server configuration stands'
                          if args.temperature == 'server' else
                          f'request temperature {args.temperature} OVERRIDES the '
                          'server default; other server-side samplers still apply'),
        'max_tokens_override': args.max_tokens,
        'extra_body': args.extra_body,
        'warmup': args.warmup, 'repetitions': args.repetitions,
        'order_seed': args.order_seed,
        'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    print(json.dumps({k: metadata[k] for k in
                      ('label', 'base_url', 'server_models', 'cases_sha256')}))

    rows = []
    with (args.output / 'results.jsonl').open('x') as sink:
        sink.write(json.dumps(metadata) + '\n')
        for i, item in enumerate(items, 1):
            case = item['case']
            try:
                row = run_case(base_url, args.model, case, args)
                row.update(phase=item['phase'], repeat=item['repeat'],
                           case_id=case['id'], type='request')
            except Exception as e:
                # Loud: a failed request is recorded as a failure, never as a
                # missing row. A silently short run would flatter the summary.
                detail = ''
                if isinstance(e, urllib.error.HTTPError):
                    try:
                        detail = e.read()[:400].decode('utf-8', 'replace')
                    except Exception:
                        detail = ''
                row = {'type': 'error', 'phase': item['phase'],
                       'repeat': item['repeat'], 'case_id': case['id'],
                       'error': f'{type(e).__name__}: {e}', 'detail': detail}
            sink.write(json.dumps(row) + '\n')
            rows.append(row)
            status = (row.get('check') or {}).get('status', row.get('error', '?'))
            print(f'  [{i}/{len(items)}] {item["phase"]:7} {case["id"]:24} '
                  f'{row.get("response_s", float("nan")):6.2f}s  {status}')

    measured = [r for r in rows if r.get('type') == 'request' and r['phase'] == 'measure']
    errors = [r for r in rows if r.get('type') == 'error']
    by_case = {}
    for r in measured:
        by_case.setdefault(r['case_id'], []).append(r)

    def med(values):
        vals = sorted(v for v in values if isinstance(v, (int, float)))
        return vals[len(vals) // 2] if vals else None

    summary = {
        'label': args.label, 'base_url': base_url,
        'server_models': metadata['server_models'],
        'measured': len(measured), 'errors': len(errors),
        'checks': dict(Counter((r.get('check') or {}).get('status') for r in measured)),
        'truncated': sum(1 for r in measured if r.get('truncated')),
        'response_s_median': med(r['response_s'] for r in measured),
        'prompt_tokens_median': med(r['prompt_tokens'] for r in measured),
        'cached_tokens_median': med(r.get('cached_tokens') for r in measured),
        'completion_tokens_median': med(r['completion_tokens'] for r in measured),
        'reasoning_chars_median': med(r['reasoning_chars'] for r in measured),
        'cases': {cid: {'samples': len(rs),
                        'response_s': med(r['response_s'] for r in rs),
                        'completion_tokens': med(r['completion_tokens'] for r in rs),
                        'checks': dict(Counter((r.get('check') or {}).get('status')
                                               for r in rs))}
                  for cid, rs in by_case.items()},
    }
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')

    passed = summary['checks'].get('pass', 0)
    print(f"\n{args.label}: {passed}/{len(measured)} expected-field pass, "
          f"median {summary['response_s_median']:.2f}s, "
          f"{len(errors)} request error(s)")
    if errors:
        print(f"  first error: {errors[0]['error']} {errors[0].get('detail', '')[:200]}")
    print(f'wrote {args.output}/summary.json')
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
