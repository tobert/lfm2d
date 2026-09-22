#!/usr/bin/env python3
"""Show `/v1/opinion` on a handful of commands against a running daemon.

For each command: one describe-then-read (a described-cache miss unless the
daemon has seen those bytes), then the same request again (a hit), and
optionally the generative `/v1/adjudicate` verdict for the same bytes — which
resumes from the described state, so the column is what escalation costs. Prints
what the model described, the renormalised probabilities beside the raw mass,
the margin, and the wall time of each path. Picks no winner: the columns are
what a caller would threshold on its own data.

    python3 benchmarks/lfm25/opinion_demo.py --url http://127.0.0.1:18170 \\
        --spec command-verdict-enum-v1 --field verdict commands.txt

`commands.txt` holds one command per line. Facts are not built here (the F7
facts block is an app's job); pass --facts-from to read a JSON map of
command -> facts if you have one.
"""
import argparse, json, sys, time, urllib.error, urllib.request


def rpc(url, path, payload=None):
    req = urllib.request.Request(url.rstrip('/') + path,
                                 None if payload is None else json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=130) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return {'http_error': e.code, 'body': e.read().decode()[:300]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('commands', help='file with one command per line')
    ap.add_argument('--url', default='http://127.0.0.1:18170')
    ap.add_argument('--spec', required=True)
    ap.add_argument('--field', default='verdict')
    ap.add_argument('--facts-from', help='JSON file mapping command -> facts block')
    ap.add_argument('--generative', action='store_true',
                    help='also run /v1/adjudicate on the same bytes after the read: it resumes from '
                         'the described state (escalation) and shows its verdict and time')
    ap.add_argument('--json', action='store_true', help='print the raw responses too')
    a = ap.parse_args()
    facts = json.load(open(a.facts_from)) if a.facts_from else {}
    commands = [l.rstrip('\n') for l in open(a.commands) if l.strip()]
    menu = {m['spec']: m for m in rpc(a.url, '/v1/opinion/specs')}
    if a.spec not in menu:
        sys.exit(f'{a.spec!r} is not on the menu: {sorted(menu)}')
    field = next((f for f in menu[a.spec]['fields'] if f['field'] == a.field), None)
    if not field or field['kind'] != 'choice':
        sys.exit(f'{a.field!r} is not a choice field of {a.spec!r}')
    options = field['options']
    describes = [f['field'] for f in menu[a.spec]['fields'][:menu[a.spec]['fields'].index(field)]]
    print(f'spec {a.spec}  question {a.field} over {options}  describes {describes}')
    print(f'{"command":34s} ' + ' '.join(f'{o[:7]:>7s}' for o in options) +
          f' {"mass":>7s} {"margin":>6s} {"miss ms":>8s} {"hit ms":>7s}' +
          (f' {"generative":>12s} {"esc ms":>7s} {"resumed":>8s}' if a.generative else ''))
    for command in commands:
        state = {'command': command}
        if command in facts:
            state['facts'] = facts[command]
        body = {'spec': a.spec, 'state': state, 'questions': [{'field': a.field}]}
        t0 = time.perf_counter()
        first = rpc(a.url, '/v1/opinion', body)
        miss_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        second = rpc(a.url, '/v1/opinion', body)
        hit_ms = (time.perf_counter() - t0) * 1000
        if 'http_error' in first:
            print(f'{command[:34]:34s} HTTP {first["http_error"]}: {first["body"]}')
            continue
        answer = first['answers'][0]
        probs = {o['option']: o['prob'] for o in answer['options']}
        line = (f'{command[:34]:34s} ' + ' '.join(f'{probs[o]:7.3f}' for o in options) +
                f' {answer["sequence_mass"]:7.3f} {answer["margin"]:6.2f} {miss_ms:8.0f} {hit_ms:7.0f}')
        if a.generative:
            t0 = time.perf_counter()
            rendered = state.get('facts', '') + f'Command:\n{command}'  # OpinionState::render
            gen = rpc(a.url, '/v1/adjudicate', {'input': rendered})
            gen_ms = (time.perf_counter() - t0) * 1000
            verdict = (gen.get('report') or {}).get(a.field, gen.get('report_error', 'error'))
            resumed = gen.get('resumed_tokens')
            line += f' {str(verdict):>12s} {gen_ms:7.0f} {str(resumed):>8s}'
        print(line)
        described = ', '.join(f'{d["field"]}={json.dumps(d["value"])}' for d in first['described'])
        cache = first['cache']
        print(f'    described: {described}')
        print(f'    cache: first {cache["state"]}/{cache["described"]}, second '
              f'{second["cache"]["state"]}/{second["cache"]["described"]}; '
              f'described_tokens {first["described_tokens"]}; identical read: '
              f'{first["answers"] == second["answers"]}')
        if a.json:
            print(json.dumps(first, indent=1))


if __name__ == '__main__':
    main()
