#!/usr/bin/env python3
"""The System 1 matinee in one terminal: three acts against one daemon,
title cards between them, and each act's REPL driven through a pty — so the
items echo like someone typed them, the spinners stay alive, and the next
fed line waits for the child's own prompt however long a row takes.

    python3 show.py --url http://127.0.0.1:18171     # pauses for enter between acts
    python3 show.py --auto                           # rehearsals: no pauses

Acts: 1 inbox (any question a schema can name) · 2 xray (the distribution
under every written field, and the exact bytes read) · 3 asked (the raw mass
beside the renormalised answer). Every act uses the email-triage-v1 spec;
the daemon needs it on the menu (--opinion-spec, or an upload). Ctrl-C ends
the show.
"""
import argparse, json, os, pty, select, subprocess, sys, threading, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
B, DIM, OFF = '\033[1m', '\033[2m', '\033[0m'
PROMPT = '\u276f'  # the '❯' the act REPLs print before reading a line


def run_feed(master, lines, stop):
    """Relay the child's output to the screen; on each prompt, type the next
    line. The pty echoes it, so it looks typed. After the last line, Ctrl-D."""
    idx, recent = 0, ''
    while True:
        r, _, _ = select.select([master], [], [], 1.0)
        if not r:
            if stop.is_set():
                return
            continue
        try:
            chunk = os.read(master, 4096)
        except OSError:
            return
        if not chunk:
            return
        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()
        recent = (recent + chunk.decode('utf-8', 'replace'))[-80:]
        if PROMPT in recent:
            recent = ''
            if idx >= len(lines):
                time.sleep(0.4)
                try:
                    os.write(master, b'\x04')  # EOF: the REPL exits
                except OSError:
                    pass
                return
            time.sleep(0.2)
            try:
                os.write(master, (lines[idx] + '\n').encode())
            except OSError:
                return
            idx += 1


def card(n, title, blurb):
    print(f'\n{B}╭─ act {n} · {title} ' + '─' * max(2, 62 - len(title)) + '╮')
    for row in blurb:
        print(f'{B}│{OFF} {DIM}{row}{OFF}')
    print(f'{B}╰' + '─' * 64 + '╯\n')


def run(act, stop):
    cmd, feed_lines = act
    if not feed_lines:
        subprocess.run(cmd, cwd=HERE)
        return
    master, slave = pty.openpty()
    env = {**os.environ, 'PYTHONUNBUFFERED': '1', 'COLUMNS': '96'}
    proc = subprocess.Popen(cmd, cwd=HERE, stdin=slave, stdout=slave,
                            stderr=slave, env=env, close_fds=True)
    os.close(slave)
    t = threading.Thread(target=run_feed, args=(master, feed_lines, stop), daemon=True)
    t.start()
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        stop.set()
        raise SystemExit(f'\n{B}that is the show.{OFF}')
    finally:
        try:
            os.close(master)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://127.0.0.1:18171')
    ap.add_argument('--auto', action='store_true', help='no enter between acts')
    ap.add_argument('--acts', default='123', help='subset of acts, e.g. --acts 13')
    a = ap.parse_args()

    try:
        menu = json.load(urllib.request.urlopen(a.url.rstrip('/') + '/v1/opinion/specs', timeout=5))
    except OSError:
        sys.exit(f'no daemon on {a.url} — start one first (demo/README.md)')
    if 'email-triage-v1' not in {m['spec'] for m in menu}:
        sys.exit(f'email-triage-v1 is not on {a.url}\'s menu: start the daemon with '
                 f'--opinion-spec demo/specs/email-triage-v1.json, or upload it')
    print(f'{B}the system 1 matinee{OFF}  {DIM}{len(menu)} specs on the menu: '
          f'{", ".join(sorted(m["spec"] for m in menu))}{OFF}')
    stop = threading.Event()

    def lines_of(path):
        return [l.rstrip('\n') for l in open(HERE / path) if l.strip()]

    acts = {
        '1': ('inbox — the verdict vocabulary is the app\u2019s',
              ['a support-routing spec written as a prop; same primitive'],
              ([sys.executable, 'blink_anything.py', '--url', a.url,
                '--spec', 'email-triage-v1', '--field', 'verdict'],
               lines_of('inputs/inbox.txt'))),
        '2': ('xray — what it wrote, and what it wrote it from',
              ['every choice field asked; each written value is a greedy pick',
               'then the exact bytes the read continued, checked by sha256'],
              ([sys.executable, 'xray.py', '--url', a.url, '--spec', 'email-triage-v1'],
               lines_of('inputs/xray.txt'))),
        '3': ('asked — was the model even asked?',
              ['the full menu always holds the mass: the grammar walked it there',
               'narrow the menu and prob renormalises whatever is left'],
              ([sys.executable, 'asked.py', '--url', a.url,
                '--spec', 'email-triage-v1', '--field', 'verdict'],
               lines_of('inputs/asked.txt'))),
    }
    for key in a.acts:
        if key not in acts:
            sys.exit(f'no act {key!r}: the acts are {", ".join(acts)}')
        title, blurb, act = acts[key]
        card(key, title, blurb)
        run(act, stop)
        if not a.auto and key != a.acts[-1]:
            try:
                input(f'{DIM}── next act (enter) ──{OFF}')
            except EOFError:
                pass
    print(f'\n{B}one 8B model · one resident prefix · any question a spec can name, and the numbers under each.{OFF}')


if __name__ == '__main__':
    main()
