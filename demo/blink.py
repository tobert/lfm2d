#!/usr/bin/env python3
"""Blink: the System 1 gut-reaction demo. Type a command, see the model's
instant opinion before anything runs. Stdlib only; needs a daemon with
/v1/opinion on --url.

    python3 blink.py --url http://127.0.0.1:18171 --spec command-verdict-enum-v1

The story in one screen: fast distributional read (~50 ms warm), what the
model thinks the command DOES, and the gate a caller would apply. Ctrl-D exits.
"""
import argparse, json, sys, time, urllib.error, urllib.request

GREEN, RED, YELLOW, DIM, BOLD, OFF = '\033[32m', '\033[31m', '\033[33m', '\033[2m', '\033[1m', '\033[0m'


def rpc(url, path, payload):
    req = urllib.request.Request(url.rstrip('/') + path, json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def bar(p, width=18):
    filled = round(p * width)
    color = GREEN if p < 0.34 else YELLOW if p < 0.67 else RED
    return f'{color}{"█" * filled}{"░" * (width - filled)}{OFF} {p:5.1%}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://127.0.0.1:18171')
    ap.add_argument('--spec')
    ap.add_argument('--field', default='verdict')
    ap.add_argument('--ask-below', type=float, default=0.8,
                    help='P(top option) gate: below it, the caller should pause (System 2)')
    a = ap.parse_args()

    menu = json.load(urllib.request.urlopen(a.url.rstrip('/') + '/v1/opinion/specs', timeout=10))
    menu = {m['spec']: m for m in menu}
    spec = a.spec or sorted(menu)[0]
    entry = menu[spec]
    fields = entry['fields']
    field = next(f for f in fields if f['field'] == a.field and f['kind'] == 'choice')
    options = field['options']
    print(f'{BOLD}blink — system 1 reads {spec!r} live{OFF}')
    print(f'{DIM}verdicts: {", ".join(options)}  ·  type a command, press enter, nothing runs{OFF}\n')

    while True:
        try:
            command = input(f'{BOLD}❯ {OFF}')
        except EOFError:
            print()
            return
        if not command.strip():
            continue
        body = {'spec': spec, 'state': {'command': command}, 'questions': [{'field': a.field}]}
        t0 = time.perf_counter()
        try:
            resp = rpc(a.url, '/v1/opinion', body)
        except urllib.error.HTTPError as e:
            print(f'{RED}{e.code} {e.read().decode()[:200]}{OFF}')
            continue
        ms = (time.perf_counter() - t0) * 1000
        ans = resp['answers'][0]
        probs = {o['option']: o['prob'] for o in ans['options']}
        top = max(probs, key=probs.get)
        print(f'{DIM}it reads this as:{OFF}')
        for d in resp['described']:
            print(f'{DIM}  {d["field"]:8s} {json.dumps(d["value"])}{OFF}')
        print('  ' + '  '.join(f'{o} {bar(probs[o])}' for o in options))
        if top != 'allow':
            gate = (f'{YELLOW}{BOLD}top = {top} ({probs[top]:.2f}) → the gut says: pause, ask System 2{OFF}')
        elif probs['allow'] >= a.ask_below:
            gate = (f'{GREEN}{BOLD}allow {probs["allow"]:.2f} ≥ {a.ask_below} → run it, no second thought{OFF}')
        else:
            gate = (f'{YELLOW}{BOLD}allow leads but only {probs["allow"]:.2f} < {a.ask_below} → let it pass, log the hesitation{OFF}')
        print(f'  {gate}   {DIM}{ms:.0f} ms · mass {ans["sequence_mass"]:+.3f} · margin {ans["margin"]:.2f}{OFF}\n')


if __name__ == '__main__':
    main()
