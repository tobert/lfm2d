#!/usr/bin/env python3
"""Backfill the `pkg_install` field onto every existing v9 row.

pkg_install (bool): true iff the statement's PRIMARY effect is fetching and
installing/adding new external code -- a package manager install/add, or a
fetch-and-execute installer script (get.docker.com, sh.rustup.rs, etc). false
for everything else, INCLUDING uninstall/remove/list/query operations on a
package manager -- those don't introduce new untrusted code, which is the
axis this field exists to capture (a future client-side "validate new deps"
intervention point). Independent of `label` (severity) -- do not let this
script's heuristic leak into label.

Deterministic regex pass + a printed list of every row it flagged true, so
the call is checked by reading, not by trusting the regex. Re-run is a no-op
(idempotent: only adds a missing key, never overwrites an existing one).

    python3 training/v9/backfill_pkg_install.py            # apply
    python3 training/v9/backfill_pkg_install.py --dry-run  # report only
"""
import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Install/add verbs across common package managers, and fetch-execute
# installer patterns (curl/wget piped to a shell). Deliberately does NOT
# match uninstall/remove/list/search/info/list-installed forms.
INSTALL_RE = re.compile(
    r'\b('
    r'pip3?\s+install(?!\s+.*-\s*U\s+pip\b)'            # pip / pip3 install
    r'|npm\s+(i|install|ci)\b'                            # npm install/i/ci
    r'|yarn\s+add\b'
    r'|pnpm\s+add\b'
    r'|cargo\s+(install|add)\b'
    r'|gem\s+install\b'
    r'|go\s+(get|install)\b'
    r'|apt(-get)?\s+install\b'
    r'|(dnf|yum)\s+install\b'
    r'|pacman\s+-S\b'
    r'|brew\s+install\b'
    r'|conda\s+install\b'
    r'|apk\s+add\b'
    r'|snap\s+install\b'
    r'|composer\s+(install|require)\b'
    r'|poetry\s+add\b'
    r'|uv\s+(pip\s+install|add)\b'
    r'|pip\s+install\s+-e\b'
    r')',
    re.IGNORECASE,
)
# Fetch-and-execute installer shape: a fetch piped into an interpreter, the
# rule-15 construct, which is ALSO a dependency-fetch event even when the
# thing being fetched isn't a package-manager package (rustup, docker CE).
FETCH_EXEC_RE = re.compile(
    r'(curl|wget)\b.*\|\s*(sudo\s+)?(bash|sh|zsh)\b'
    r'|eval\s*"?\$\(\s*(curl|wget)\b',
    re.IGNORECASE,
)


def flag(text: str) -> bool:
    return bool(INSTALL_RE.search(text) or FETCH_EXEC_RE.search(text))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    total = flagged = added = 0
    flagged_rows = []

    for f in sorted(HERE.glob('slice*/incoming/*.jsonl')):
        lines = f.read_text().splitlines()
        out = []
        changed = False
        for line in lines:
            if not line.strip():
                continue
            r = json.loads(line)
            total += 1
            if 'pkg_install' not in r:
                r['pkg_install'] = flag(r['text'])
                added += 1
                changed = True
            if r['pkg_install']:
                flagged += 1
                flagged_rows.append((f'{f.parts[-3]}/{f.stem}', r['text'], r['label']))
            out.append(json.dumps(r, sort_keys=True))
        if changed and not args.dry_run:
            f.write_text('\n'.join(out) + '\n')

    print(f'rows seen: {total}  pkg_install added: {added}  flagged true: {flagged}\n')
    print('=== every row flagged pkg_install=true (read these) ===')
    for src, text, label in flagged_rows:
        print(f'  [{label:<16}] {src:<28} {text[:70]!r}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
