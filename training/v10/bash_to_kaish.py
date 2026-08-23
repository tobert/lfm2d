#!/usr/bin/env python3
"""Verify and print the bash -> kaish side-by-side in bash_to_kaish.tsv.

Every kaish column must plan. For the bash column the invariant depends
on the group (kaish-lead, 2026-08-23 -- "a canary that fires on success
gets muted, and then it is not a canary any more"):

  B, C  bash must NEVER plan: these are shapes kaish deliberately does
        not speak (`[ ]` is the list literal, no subshells, ...). A bash
        side that starts planning here is a kaish regression -> exit 1.
  A     bash planning is the WIN condition: legal bash kaish wrongly
        rejects today. When the lexer fix lands these rows flip, and the
        script reports FIXED and exits 0 -- move them out of the table
        (or keep them as regression rows with group A-fixed).

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
    fixed = []
    for line in TSV.read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        group, bash, kai, note = line.split('\t')
        b_cmds, b_err = plan(bash)
        k_cmds, k_err = plan(kai)
        if b_cmds is not None:
            if group == 'A':
                fixed.append(f'A row now plans (kaish fixed it): {bash!r}')
            else:
                stale.append(f'bash side now PLANS for a shape kaish deliberately '
                             f'does not speak (group {group}): {bash!r}')
        if k_cmds is None:
            stale.append(f'kaish side FAILS: {kai!r}: {k_err}')
        print(f'{group}  {bash}')
        print(f'   bash : ' + (f'FAIL {b_err[:50]}' if b_err else ('PLANS (fixed)' if group == 'A' else 'PLANS (!! regression)')))
        print(f'   kaish: {kai}')
        if k_cmds is None:
            shown = 'FAIL ' + k_err[:50]
        else:
            shown = ' | '.join(k_cmds) or '(assignment only, no command)'
        print(f'   plan : {shown}   -- {note}')
    if fixed:
        print('\nFIXED IN KAISH (win condition; retire or re-tag these rows):')
        for f in fixed:
            print('  ' + f)
    if stale:
        print('\nSTALE ROWS (kaish changed; update the table or the plan):', file=sys.stderr)
        for s in stale:
            print('  ' + s, file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
