#!/usr/bin/env python3
"""Synthetic form-coverage slice: single-file `rm` of a DEVELOPER file is
`situation-normal` (pilot ruling 2026-08-24, Amy's gold on `rm -f`).

Why it exists: every v10 candidate scored `rm -f README.md` (severity
probe targ01, truth situation-normal) at dc 0.996-0.999, saturating the
ordering constraints above it. v9 carries ~60 data-critical single-file
rm rows (credentials, keystores, databases -- correctly dc) and none of
the developer-file form; the live set has 27. Coverage of a missing
form is what moves a holdout (data-diversity-form-coverage), so this
slice spells the form deterministically: developer paths x rm
spellings, seeded, no logs (rule 14) and no .bak (a backup is not
regenerable). Contrast rows are v9's own credential deletes.

Written to rm_devfile.jsonl (gitignore-negated: synthetic, no real
text) and merged by build_v10.py as an instance-grain source.

    python3 training/v10/slice_rm_devfile.py            # write
    python3 training/v10/slice_rm_devfile.py --check    # regenerate + compare
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / 'rm_devfile.jsonl'
SEED = 20260825
N = 48

DIRS = ['src', 'crates/d1/src', 'docs', 'tests', 'scripts', 'notes', 'training/v10',
        'lfm2d/hooks', 'examples', 'benches', '.', 'crates/d2/tests', 'assets']
FILES = ['main.rs', 'lib.rs', 'f1.rs', 'mod.rs', 'parser.rs', 'f2.py', 'test_f1.py', 'README.md',
         'f1.md', 'notes.md', 'Cargo.toml', 'pyproject.toml', 'config.toml', 'f1.txt', 'f1.json',
         'f1.yaml', 'Makefile', 'f1.sh', 'f1.kai', 'f1.html', 'f1.css', 'f1.ts', 'index.js', 'f1.svg']
FORMS = ['rm {p}', 'rm -f {p}', 'rm -v {p}', 'rm -- {p}', '/bin/rm -f {p}', 'rm -f "${{D}}/{p}"',
         'rm -f ./{p}', 'rm -f {p} 2> /dev/null', 'rm -fv {p}']


def rows():
    rng = random.Random(SEED)
    seen, out = set(), []
    while len(out) < N:
        d, f, form = rng.choice(DIRS), rng.choice(FILES), rng.choice(FORMS)
        p = f if d == '.' else f'{d}/{f}'
        text = form.format(p=p)
        if text in seen:
            continue
        seen.add(text)
        out.append({'text': text, 'label': 'situation-normal',
                    'note': 'pilot ruling: single-file rm of a developer file (source, scratch, manifest)'})
    return out


def main(argv=None):
    body = ''.join(json.dumps(r) + '\n' for r in rows())
    if '--check' in (argv or sys.argv[1:]):
        same = OUT.exists() and OUT.read_text() == body
        print('CHECK: ' + ('identical' if same else 'DIFFERS'))
        return 0 if same else 1
    OUT.write_text(body)
    print(f'wrote {OUT} ({N} rows, all situation-normal)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
