#!/usr/bin/env python3
"""Cascade: watch the two systems split the work. Every command gets a System 1
blink (~40 ms warm). When the blink hesitates, System 2 wakes and THINKS — but
it resumes from the very state the blink built, so thinking is a continuation,
not a restart. --cold prices the third act: the same thought from scratch,
which is what a System-2-only design pays for every single command.

    python3 cascade.py --url http://127.0.0.1:18171 --pass-option allow --cold

Type a command, press enter, nothing runs. Ctrl-D exits.
"""
import argparse, json, sys, threading, time, urllib.error, urllib.request

GREEN, RED, YELLOW, CYAN, DIM, BOLD, OFF = ('\033[32m', '\033[31m', '\033[33m',
                                            '\033[36m', '\033[2m', '\033[1m', '\033[0m')
SPIN = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']


def rpc(url, path, payload):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url.rstrip('/') + path, data,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def mini(probs, options):
    return ' '.join(f'{o[:6]}={probs[o]:.2f}' for o in options)


def lane(label, ms, color, scale):
    blocks = max(1, round(ms / scale))
    return f'  {label:12s} {color}' + '▇' * blocks + OFF + f' {ms:6.0f} ms'


class Spinner:
    def __init__(self, msg):
        self.stop = False
        self.msg = msg
        self.t = threading.Thread(target=self.run, daemon=True)

    def run(self):
        i = 0
        while not self.stop:
            sys.stdout.write(f'\r  {CYAN}{SPIN[i % len(SPIN)]} {self.msg}{OFF}')
            sys.stdout.flush()
            i += 1
            time.sleep(0.07)

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *a):
        self.stop = True
        self.t.join()
        sys.stdout.write('\r' + ' ' * 60 + '\r')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://127.0.0.1:18171')
    ap.add_argument('--spec', default='command-verdict-enum-v1')
    ap.add_argument('--field', default='verdict')
    ap.add_argument('--pass-option', required=True,
                    help='the option this system treats as pass-through (checked against the menu)')
    ap.add_argument('--ask-below', type=float, default=0.8)
    ap.add_argument('--cold', action='store_true', help='also price thinking from scratch')
    a = ap.parse_args()

    menu = {m['spec']: m for m in json.load(
        urllib.request.urlopen(a.url.rstrip('/') + '/v1/opinion/specs', timeout=10))}
    field = next(f for f in menu[a.spec]['fields'] if f['field'] == a.field)
    options = field['options']
    if a.pass_option not in options:
        sys.exit(f'{a.pass_option!r} is not an option of {a.spec!r}/{a.field} '
                 f'({options}) — read the menu, do not invent labels')
    describes = [f['field'] for f in menu[a.spec]['fields'][:menu[a.spec]['fields'].index(field)]]
    n_let, n_escal, t_system1, t_system2 = 0, 0, 0.0, 0.0
    print(f'{BOLD}cascade — system 1 blinks, system 2 thinks, you pay only for hesitation{OFF}')
    print(f'{DIM}gate: escalate unless P({a.pass_option}) ≥ {a.ask_below}  ·  nothing you type runs{OFF}\n')

    while True:
        try:
            command = input(f'{BOLD}❯ {OFF}')
        except EOFError:
            print()
            break
        if not command.strip():
            continue
        state = {'command': command}
        body = {'spec': a.spec, 'state': state, 'questions': [{'field': a.field}]}
        t0 = time.perf_counter()
        try:
            read = rpc(a.url, '/v1/opinion', body)
        except urllib.error.HTTPError as e:
            print(f'{RED}{e.code} {e.read().decode()[:200]}{OFF}')
            continue
        blink_ms = (time.perf_counter() - t0) * 1000
        ans = read['answers'][0]
        probs = {o['option']: o['prob'] for o in ans['options']}
        print(f'  {DIM}blink{OFF}  {mini(probs, options)}   '
              f'{DIM}{read["cache"]["described"]} · {blink_ms:.0f} ms{OFF}')
        if probs.get(a.pass_option, 0) >= a.ask_below:
            n_let += 1
            t_system1 += blink_ms
            print(f'  {GREEN}passed — no LLM woke up{OFF}\n')
            continue
        n_escal += 1
        t_system1 += blink_ms
        rendered = state.get('facts', '') + f'Command:\n{command}'
        t0 = time.perf_counter()
        with Spinner('system 2 is thinking…'):
            gen = rpc(a.url, '/v1/adjudicate', {'input': rendered})
        think_ms = (time.perf_counter() - t0) * 1000
        t_system2 += think_ms
        report = gen.get('report') or {}
        resumed = gen.get('resumed_tokens')
        verdict = report.get(a.field, '?')
        color = GREEN if verdict == a.pass_option else RED
        print(f'  {color}thought{OFF}  {BOLD}{verdict}{OFF}  '
              f'{DIM}({resumed} tokens carried from the blink, {think_ms:.0f} ms){OFF}')
        print(f'         {report.get("reason", "")}')
        if a.cold:
            t0 = time.perf_counter()
            with Spinner('pricing the same thought from scratch…'):
                rpc(a.url, '/v1/adjudicate', {'input': rendered, 'use_cache': False})
            cold_ms = (time.perf_counter() - t0) * 1000
            scale = max(cold_ms, 100) / 40
            print(lane('blink', blink_ms, CYAN, scale))
            print(lane('think (warm)', think_ms, YELLOW, scale))
            print(lane('think (cold)', cold_ms, RED, scale))
        print()

    all_cmd = n_let + n_escal
    print(f'{BOLD}session ledger{OFF}')
    print(f'  {n_let} passed on the blink alone ({t_system1 / all_cmd if all_cmd else 0:.0f} ms avg across all)')
    print(f'  {n_escal} escalated: blink + a warm thought (System 2 never re-read the command)')
    print(f'  system 1 time {t_system1:.0f} ms · system 2 time {t_system2:.0f} ms '
          f'({100 * t_system2 / (t_system1 + t_system2) if t_system1 + t_system2 else 0:.0f}% of thinking '
          f'bought only for the {n_escal} that hesitated)')


if __name__ == '__main__':
    main()
