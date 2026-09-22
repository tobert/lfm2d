#!/usr/bin/env python3
"""night_watch: System 1 on the fleet. A feed of cluster events (actor + command
lines, timestamps included) flows through the opinion read on a loop. Most
events pass in silence at blink speed; when the blink hesitates, System 2 wakes
and says why — resuming from the blink's own description, so the flag costs a
thought, not a re-read. --repeat loops forever, which is what makes it look
like the monitoring it is meant to be: the second pass over the feed reads at
described-cache speed, ~40 ms an event.

    python3 night_watch.py --url http://127.0.0.1:18171 --spec command-verdict-enum-v1 \
        --pass-option allow inputs/fleet_feed.txt --repeat

Feed format: `TIMESTAMP ACTOR COMMAND` (whitespace-separated; the command is
the whole rest of the line). The pass option is a request parameter, never a
hard-coded label: read the menu at runtime, pass whatever the consuming
system treats as its pass-through.
"""
import argparse, json, sys, time, urllib.error, urllib.request

GREEN, RED, YELLOW, CYAN, DIM, BOLD, OFF = ('\033[32m', '\033[31m', '\033[33m',
                                            '\033[36m', '\033[2m', '\033[1m', '\033[0m')


def rpc(url, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url.rstrip('/') + path, data,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def parse_feed(path):
    rows = []
    for line in open(path):
        parts = line.rstrip('\n').split(None, 2)
        if len(parts) < 3:
            continue
        ts, actor, command = parts
        rows.append((ts.split('T')[-1].rstrip('Z'), actor, command.strip()))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('feed')
    ap.add_argument('--url', default='http://127.0.0.1:18171')
    ap.add_argument('--spec', default='command-verdict-enum-v1')
    ap.add_argument('--field', default='verdict')
    ap.add_argument('--pass-option', required=True,
                    help='the option this system treats as pass-through')
    ap.add_argument('--ask-below', type=float, default=0.8)
    ap.add_argument('--speed', type=float, default=0.5,
                    help='seconds between events (their real work takes longer anyway)')
    ap.add_argument('--repeat', action='store_true', help='loop the feed forever')
    a = ap.parse_args()

    menu = {m['spec']: m for m in rpc(a.url, '/v1/opinion/specs')}
    if a.spec not in menu:
        sys.exit(f'{a.spec!r} not on the menu: {sorted(menu)}')
    entry = menu[a.spec]
    field = next((f for f in entry['fields'] if f['field'] == a.field), None)
    if not field or field['kind'] != 'choice':
        sys.exit(f'{a.field!r} is not a choice field of {a.spec!r}')
    if a.pass_option not in field['options']:
        sys.exit(f'{a.pass_option!r} is not an option of {a.spec!r}/{a.field} '
                 f'({field["options"]}) — read the menu, do not invent labels')
    rows = parse_feed(a.feed)

    print(f'{BOLD}night watch — {len(rows)} events on {a.spec!r}, '
          f'pass = {a.pass_option!r} ≥ {a.ask_below}{OFF}')
    print(f'{DIM}time      who          event{" " * 50}read   flag{OFF}')
    seen = flagged = 0
    total_ms = 0.0
    cycle = 0
    while True:
        cycle += 1
        if cycle > 1:
            print(f'{DIM}──── feed repeats (cycle {cycle}: the described cache is warm) ────{OFF}')
        for ts, actor, command in rows:
            body = {'spec': a.spec, 'state': {'command': command},
                    'questions': [{'field': a.field}]}
            t0 = time.perf_counter()
            try:
                read = rpc(a.url, '/v1/opinion', body)
            except urllib.error.HTTPError as e:
                print(f'{RED}{e.code} {e.read().decode()[:120]}{OFF}')
                continue
            ms = (time.perf_counter() - t0) * 1000
            seen += 1
            total_ms += ms
            probs = {o['option']: o['prob'] for o in read['answers'][0]['options']}
            p_pass = probs.get(a.pass_option, 0.0)
            shown = command if len(command) <= 52 else command[:49] + '...'
            if p_pass >= a.ask_below:
                print(f'{ts} {actor:11s} {shown:54s}{DIM}{ms:5.0f} ms{OFF} '
                      f'{GREEN}✓{OFF}     {DIM}{read["cache"]["described"]}{OFF}')
            else:
                flagged += 1
                print(f'{ts} {actor:11s} {shown:54s}{DIM}{ms:5.0f} ms{OFF} '
                      f'{YELLOW}{BOLD}⚑ {p_pass:.2f}{OFF}')
                t1 = time.perf_counter()
                gen = rpc(a.url, '/v1/adjudicate', {'input': f'Command:\n{command}'})
                think_ms = (time.perf_counter() - t1) * 1000
                report = gen.get('report') or {}
                verdict = report.get(a.field, '?')
                color = {a.pass_option: GREEN}.get(verdict, RED)
                print(f'{CYAN}           ↳ system 2 ({gen.get("resumed_tokens")} tokens carried, '
                      f'{think_ms:.0f} ms){OFF}')
                print(f'           {color}{BOLD}{verdict}{OFF}  {report.get("reason", "")}')
            time.sleep(a.speed)
        if not a.repeat:
            break
    print(f'\n{BOLD}shift summary{OFF}: {seen} events · {flagged} flagged '
          f'({100 * flagged / seen if seen else 0:.0f}%) · '
          f'avg read {total_ms / seen if seen else 0:.0f} ms')


if __name__ == '__main__':
    main()
