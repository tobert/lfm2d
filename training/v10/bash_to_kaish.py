#!/usr/bin/env python3
"""Verify and print the bash -> kaish side-by-side in bash_to_kaish.tsv.

Every kaish column must plan and every bash column must FAIL to plan;
exit 1 naming any row that flips. Amy ruled (2026-08-23) that kaish's
language does not change for any shape in the table -- the fix is to
quote the word -- so a bash side that starts planning is a kaish
regression, never a win. (An earlier revision carried a Group A "win
condition" branch for a lexer fold that was refused; see
`canary-must-not-fire-on-success` for why the branch existed at all.)

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
            stale.append(f'bash side now PLANS (group {group}; kaish regression, '
                         f'or the ruling changed): {bash!r}')
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
