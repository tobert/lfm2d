#!/usr/bin/env python3
"""Verify and print the bash -> kaish side-by-side in bash_to_kaish.tsv.

Every kaish column must plan and every bash column must FAIL to plan;
exit 1 naming any row that flips. Amy ruled (2026-08-23) that kaish's
QUOTE-TO-JOIN RULE does not fold exceptions in for any shape in the
table -- "nothing adjacent is joined, quote to join" stays one rule
with no per-class carve-outs, so the fix for that grammar is to quote
the word. (An earlier revision carried a Group A "win condition" branch
for a lexer fold that was refused; see `canary-must-not-fire-on-success`
for why the branch existed at all.)

That ruling is about the JOIN rule, not a promise that kaish's bare-word
grammar is frozen. Separately, kaish keeps widening what it accepts
unquoted -- 0.17.2 plans `v=0.16.0` and `ping 10.0.0.1` bare, where
0.16.0 required quotes -- and Amy has acknowledged that as real,
sanctioned growth ("kaish got more support for forms of bare string
recently"), not a regression. A row whose bash side starts planning for
that reason is ABSORBED: it stops being a divergence and gets deleted
from the table, by hand, after a human confirms the cause (2026-09-21
re-derivation: `x=~/.cache/foo`, `v=0.16.0`, `ping 10.0.0.1`,
`.venv-train/bin/python x.py`, `git show HEAD:training/v9/x.py`). A row
whose bash side starts planning for any OTHER reason is still the
08-23 regression case. This script cannot tell the two apart by itself
-- it exits 1 either way, because the canary's job is to force that
call, not make it; only a human deletes an absorbed row from the table.

Either way the table is a claim about a specific kaish, and the claim
expires loudly (`commit-the-scorer`).

    .venv-train/bin/python training/v10/bash_to_kaish.py
"""
import json
import subprocess
import sys
from pathlib import Path

TSV = Path(__file__).resolve().with_name('bash_to_kaish.tsv')


def plan(src):
    r = subprocess.run(['kaish', '--plan-file', '-'], input=src, capture_output=True,
                       text=True, timeout=10)
    j = json.loads(r.stdout)
    if 'errors' in j:
        return None, j['errors'][0]['message']
    cmds = []
    for st in j['statements']:
        for c in st['plan']['commands']:
            args = ' '.join(a.get('plain', json.dumps(a)) for a in c['args'])
            red = ''.join(' ' + r['kind'] + (r['target'].get('plain', '?')
                                             if isinstance(r.get('target'), dict) else '')
                          for r in c['redirects'])
            cmds.append(f"{c['name']} {args}{red}".strip())
    return cmds, None


def main():
    ver = subprocess.run(['kaish', '--version'], capture_output=True, text=True).stdout.strip()
    print(f'verifying against {ver}\n')
    stale = []
    for line in TSV.read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        group, bash, kai, note = line.split('\t')
        b_cmds, b_err = plan(bash)
        k_cmds, k_err = plan(kai)
        if b_cmds is not None:
            stale.append(f'bash side now PLANS (group {group}; either a genuine '
                         f'kaish regression, or kaish absorbed this bare-word '
                         f'shape -- diagnose before deleting the row): {bash!r}')
        if k_cmds is None:
            stale.append(f'kaish side FAILS: {kai!r}: {k_err}')
        print(f'{group}  {bash}')
        print(f'   bash : ' + (f'FAIL {b_err[:50]}' if b_err else 'PLANS (!! regression)'))
        print(f'   kaish: {kai}')
        if k_cmds is None:
            shown = 'FAIL ' + k_err[:50]
        else:
            shown = ' | '.join(k_cmds) or '(assignment only, no command)'
        print(f'   plan : {shown}   -- {note}')
    if stale:
        print('\nSTALE ROWS (kaish changed; update the table or the plan):', file=sys.stderr)
        for s in stale:
            print('  ' + s, file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
