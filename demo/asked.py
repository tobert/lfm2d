#!/usr/bin/env python3
"""asked: was the model even asked? `prob` is renormalised over the options
the caller named; `sequence_mass` is the raw probability the model put on
them. Ask the whole menu and the mass is ~100% on everything — even "what
is the capital of France?" — because the grammar walks the model to the
slot: full mass proves the question was put, never that the input made
sense. Narrow the menu and the mass falls by whatever the omitted options
held, while `prob` renormalises the remainder into a confident-looking
answer. That is invariant 9 (docs/integration.md) in one screen.

    python3 asked.py --url http://127.0.0.1:18171 --spec command-verdict-enum-v1 --field verdict

Each line is an item; by default it is read over the full menu and then
with each option left out once. `item :: a,b` asks exactly that subset.
Nothing runs. Ctrl-D exits.
"""
import argparse, json, math, sys, time, urllib.error, urllib.request

GREEN, RED, YELLOW, DIM, BOLD, OFF = '\033[32m', '\033[31m', '\033[33m', '\033[2m', '\033[1m', '\033[0m'


def rpc(url, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url.rstrip('/') + path, data,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def meter(p, color, width=12):
    filled = round(p * width)
    return f'{color}{"█" * filled}{"░" * (width - filled)}{OFF}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://127.0.0.1:18171')
    ap.add_argument('--spec', required=True)
    ap.add_argument('--field', required=True)
    ap.add_argument('--unasked-below', type=float, default=0.5,
                    help='raw mass under which a read is called unasked (a demo choice, not a fitted gate)')
    a = ap.parse_args()

    menu = {m['spec']: m for m in rpc(a.url, '/v1/opinion/specs')}
    if a.spec not in menu:
        sys.exit(f'{a.spec!r} not on the menu: {sorted(menu)}')
    field = next((f for f in menu[a.spec]['fields'] if f['field'] == a.field), None)
    if not field or field['kind'] != 'choice':
        sys.exit(f'{a.field!r} is not a choice field of {a.spec!r}')
    options = field['options']
    print(f'{BOLD}asked — the renormalised answer beside the raw mass it stands on{OFF}')
    print(f'{DIM}spec {a.spec} · {a.field} over {options} · "item :: a,b" asks a subset{OFF}\n')

    while True:
        try:
            line = input(f'{BOLD}❯ {OFF}')
        except EOFError:
            print()
            return
        if not line.strip():
            continue
        item, sep, subset = line.rpartition(' :: ')
        if sep:
            asked = [o.strip() for o in subset.split(',') if o.strip()]
            menus = [options, [o for o in options if o in asked]]
            unknown = set(asked) - set(options)
            if unknown:
                print(f'{RED}not on the menu: {sorted(unknown)}{OFF}\n')
                continue
        else:
            item = line
            menus = [options] + [[o for o in options if o != left] for left in options]
        full_top, outcomes = None, []
        t0 = time.perf_counter()
        for i, opts in enumerate(menus):
            body = {'spec': a.spec, 'state': {'command': item},
                    'questions': [{'field': a.field, 'options': opts}]}
            try:
                resp = rpc(a.url, '/v1/opinion', body)
            except urllib.error.HTTPError as e:
                print(f'{RED}{e.code} {e.read().decode()[:200]}{OFF}')
                break
            ans = resp['answers'][0]
            outcomes.append(resp['cache']['described'])
            if i == 0:
                for d in resp['described']:
                    print(f'  {DIM}{d["field"]:9s} {json.dumps(d["value"])}{OFF}')
            mass = math.exp(ans['sequence_mass'])
            top = max(ans['options'], key=lambda o: o['prob'])
            if i == 0:
                full_top = top['option']
            dropped = [o for o in options if o not in opts]
            label = 'full menu' if not dropped else f'without {", ".join(dropped)}'
            cells = '  '.join(f'{o["option"]} {o["prob"]:5.1%}' for o in ans['options'])
            color = GREEN if mass >= a.unasked_below else RED
            note = ''
            if mass < a.unasked_below:
                held = 'the omitted options held the rest' if dropped else 'the model put it elsewhere'
                note = (f'{RED}{BOLD}unasked: {top["option"]} {top["prob"]:.0%} is renormalised noise'
                        f'{OFF}{DIM} ({held}){OFF}')
            elif top['option'] != full_top:
                note = f'{YELLOW}top moved {full_top} → {top["option"]}{OFF}'
            print(f'  {label:26s} mass {meter(mass, color)} {mass:6.1%}   {cells}   {note}')
        tally = ', '.join(f'{outcomes.count(k)} {k}' for k in dict.fromkeys(outcomes))
        print(f'  {DIM}{len(outcomes)} reads · {(time.perf_counter() - t0) * 1000:.0f} ms · '
              f'described cache: {tally}{OFF}\n')


if __name__ == '__main__':
    main()
