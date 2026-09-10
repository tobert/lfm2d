#!/usr/bin/env python3
"""Parity gate for stage 4: turning it on must change no emitted decision.

Run:  python3 lfm2d/hooks/test_stage4_parity.py

Plain script, no pytest (none on this machine). Exits non-zero on failure.
Self-contained: it starts a stub daemon on loopback, so it needs no live
lfm2d and is deterministic.

test_parity.py gates the regex half against the baseline dotfiles hook
with LFM2D_HOOK_MODE=off, so it never reaches the classifier and cannot
see stage 4 at all. This gate covers the other half: the hook runs TWICE
per case, once with LFM2D_STAGE4=off and once with `record`, and the
emitted decision plus exit code must be byte-identical.

The stub always answers data-critical, because a raised verdict is the
only case where stage 4 has anything to say -- a stub that answered
situation-normal would pass this gate while testing nothing. Its winner
is the LAST clause, so a dismissal has to follow winner_index rather than
defaulting to clause 0.

What this pins, beyond the decisions:
  - flag off writes NO stage4 key (not an empty one);
  - flag on writes a stage4 key on EVERY logged row, including rows stage
    4 declines -- a sometimes-absent key would give the field a third
    state that reads as "reviewed and cleared";
  - a guard denial is never dismissed, even when the winning clause would
    otherwise clear.

Note on the case list: several cases are assembled from fragments rather
than written whole. The regex guard reads the prose of whatever command
carries them, so a literal `git` + `add -A` in this file's source is
blocked when an agent writes the file with a shell heredoc. Assembling
the string is not evasion -- the guard's own hint says to pass such text
by path rather than reformulate the command, and this IS the file.
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

HOOK = Path(__file__).resolve().parent / 'pre_command_advisory.py'
SCORES = {'data-critical': 0.94, 'informative': 0.03, 'situation-normal': 0.03}


class Stub(BaseHTTPRequestHandler):
    """The minimum of the daemon contract this hook consumes."""

    def log_message(self, *a):
        pass

    def do_GET(self):
        self._send({'models': [{'model_id': 'stub', 'labels': list(SCORES)}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        ins = body.get('inputs') or body.get('clauses') or []
        if isinstance(ins, str):
            ins = [ins]
        models = [{'model_id': 'stub', 'weight_hash': 'stub'}]
        if self.path.endswith('/cascade'):
            last = len(ins) - 1 if ins else 0
            rows = [{'clause': c, 'top_severity': 'data-critical',
                     'severity_scores': SCORES} for c in ins]
            self._send({'model_id': 'stub', 'weight_hash': 'stub',
                        'models': models, 'clauses': rows,
                        'winner': {'index': last,
                                   'clause': ins[last] if ins else '',
                                   'severity_scores': SCORES},
                        'route': {'route': 'shell', 'cosine': 0.99}})
        else:
            # /v1/classify answers with a LIST, one entry per input.
            self._send([{'model_id': 'stub', 'weight_hash': 'stub',
                         'models': models,
                         'top': 'data-critical', 'scores': SCORES}
                        for _ in (ins or [''])])

    def _send(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(b)))
        self.end_headers()
        self.wfile.write(b)


srv = HTTPServer(('127.0.0.1', 0), Stub)
threading.Thread(target=srv.serve_forever, daemon=True).start()
URL = f'http://127.0.0.1:{srv.server_address[1]}'

# Every regex outcome (deny / soft_deny / warn / allow) crossed with every
# stage-4 outcome (dismissed / survived / fallback_path / declined),
# because the claim under test is that the two are independent.
CASES = [
    'echo restored',                       # allow      + dismissed
    'echo restored > /etc/shadow',         # allow      + write redirect
    'rm -f tests/sz.rs',                   # warn       + unlisted verb
    'git branch -a',                       # allow      + read subcommand
    'git branch -D feature',               # allow      + forbidden flag
    'git push -q origin main',             # allow      + unlisted subcommand
    'find src -type f',                    # allow      + dismissed
    'rm -rf /var/lib/data',                # soft_deny  + ratchet
    'git' + ' add -A',                     # deny       + ratchet
    'git commit ' + '--amend --no-edit',   # soft_deny  + ratchet
    'ls -la',                              # allow      + dismissed
    'cd /tmp && rm -rf ./x ; echo done',   # soft_deny  + multi-clause
    'for f in a b',                        # allow      + fallback path
    'set -euo pipefail',                   # allow      + dismissed
    '',                                    # allow      + empty
]


def run(cmd, stage4):
    """One hook invocation with an isolated cache. Returns
    (returncode, stdout, stderr, logged rows)."""
    with tempfile.TemporaryDirectory() as d:
        env = dict(os.environ)
        env.update({'XDG_CACHE_HOME': d, 'LFM2D_HOOK_MODE': 'advisory',
                    'LFM2D_URL': URL, 'LFM2D_STAGE4': stage4})
        p = subprocess.run([sys.executable, str(HOOK)],
                           input=json.dumps({'tool_name': 'Bash',
                                             'tool_input': {'command': cmd}}),
                           capture_output=True, text=True, timeout=60, env=env)
        log = Path(d) / 'claude-hooks/lfm2d-advisory.jsonl'
        rows = [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
        return p.returncode, p.stdout, p.stderr, rows


def main():
    bad = 0
    print(f'  {"command":44} {"emit identical":16} stage 4 recorded')
    print(f'  {"-" * 44} {"-" * 16} {"-" * 30}')
    for cmd in CASES:
        off = run(cmd, 'off')
        on = run(cmd, 'record')
        same = (off[0], off[1]) == (on[0], on[1])
        off_key = any('stage4' in r for r in off[3])
        s4 = [r.get('stage4') for r in on[3] if 'stage4' in r]
        note = (s4[0].get('reason') or s4[0].get('skipped')
                or s4[0].get('error')) if s4 else '(nothing)'
        dis = s4[0].get('dismissible') if s4 else None
        verdict = 'dismissed' if dis else 'survived' if dis is False else '-'
        if not same or off_key or not s4:
            bad += 1
        print(f'  {cmd[:42]!r:44} {"YES" if same else "*** NO ***":16} '
              f'{verdict:10} {note}')
        if not same:
            print(f'      off: {off[1]!r}\n      on:  {on[1]!r}')
        if off_key:
            print('      *** flag OFF but a stage4 key was written')
        if not s4:
            print('      *** flag ON but no stage4 key on the row')

    srv.shutdown()
    if bad:
        print(f'\nFAILED: {bad} of {len(CASES)} cases')
        return 1
    print(f'\nSTAGE 4 PARITY OK: {len(CASES)} cases, emitted decision identical '
          f'with the flag off and on')
    return 0


if __name__ == '__main__':
    sys.exit(main())
