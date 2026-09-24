#!/usr/bin/env python3
"""blink_anything: System 1 has no shell in it. Ask any multiple-choice
question a prompt spec can name — email routing, review moderation, whatever
the app invents — and watch the model describe the item and hesitate over the
options in well under a blink once warm. No gates here: the distribution IS
the product; the app decides what to do with it.

    python3 blink_anything.py --url http://127.0.0.1:18171 --spec email-triage-v1 --field verdict

Type an item (an email, a sentence), press enter, nothing runs. Ctrl-D exits.
"""
import argparse, json, sys, time, urllib.error, urllib.request

DIM, BOLD, OFF = '\033[2m', '\033[1m', '\033[0m'
PALETTE = ['\033[36m', '\033[33m', '\033[35m', '\033[32m', '\033[31m']


def bar(p, color, width=18):
    filled = round(p * width)
    return f'{color}{"█" * filled}{"░" * (width - filled)}{OFF} {p:5.1%}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://127.0.0.1:18171')
    ap.add_argument('--spec', required=True)
    ap.add_argument('--field', required=True)
    a = ap.parse_args()

    menu = {m['spec']: m for m in json.load(
        urllib.request.urlopen(a.url.rstrip('/') + '/v1/opinion/specs', timeout=10))}
    if a.spec not in menu:
        sys.exit(f'{a.spec!r} not on the menu: {sorted(menu)}')
    field = next((f for f in menu[a.spec]['fields'] if f['field'] == a.field), None)
    if not field or field['kind'] != 'choice':
        sys.exit(f'{a.field!r} is not a choice field of {a.spec!r}')
    options = field['options']
    print(f'{BOLD}blink_anything — one read primitive, any question{OFF}')
    print(f'{DIM}spec {a.spec}  ·  question {a.field} over {options}{OFF}\n')

    while True:
        try:
            item = input(f'{BOLD}❯ {OFF}')
        except EOFError:
            print()
            return
        if not item.strip():
            continue
        body = {'spec': a.spec, 'state': {'input': item}, 'questions': [{'field': a.field}]}
        t0 = time.perf_counter()
        req = urllib.request.Request(a.url.rstrip('/') + '/v1/opinion',
                                     json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                resp = json.load(r)
        except urllib.error.HTTPError as e:
            print(f'{e.code} {e.read().decode()[:200]}')
            continue
        ms = (time.perf_counter() - t0) * 1000
        ans = resp['answers'][0]
        probs = {o['option']: o['prob'] for o in ans['options']}
        for i, d in enumerate(resp['described']):
            print(f'{DIM}  {d["field"]:9s} {d["value"] if isinstance(d["value"], str) else json.dumps(d["value"])}{OFF}')
        print('  ' + '   '.join(f'{o} {bar(probs[o], PALETTE[i % len(PALETTE)])}'
                                for i, o in enumerate(options)))
        print(f'  {DIM}{resp["cache"]["described"]} · {ms:.0f} ms · mass {ans["sequence_mass"]:+.3f}'
              f' · margin {ans["margin"]:.2f}{OFF}\n')


if __name__ == '__main__':
    main()
