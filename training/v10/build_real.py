#!/usr/bin/env python3
"""Slice 4: the real-clause half of the v10 training set.

Plans every distinct command in the baseline window through kaish (the
serving renderer's own invocation), keys each simple command with
`soak_shapes.shape()`, joins the shape's FINAL label from
`bulk_votes.json` (consensus or Amy's ruling), scrubs the clause
structurally (`scrub.scrub_command`, ruling (b)), and dedupes: one row
per distinct scrubbed text per shape, with its multiplicity recorded,
capped per shape so `cd` does not become the corpus.

Two outputs:
  training/v10/real.jsonl          scrubbed rows {text,label,shape,n}
                                   (gitignored like every *.jsonl; the
                                   HF publication review is separate)
  ~/.cache/claude-hooks/v10-scrub-review.txt
                                   raw -> scrubbed pairs, 0600, LOCAL
                                   ONLY, for Amy's eyes on the scrub

A leak gate runs over the scrubbed rows before anything is written:
any username, home path, scratchpad slug, tailnet host, e-mail or
credential-looking token in the output fails the build loudly.

    .venv-train/bin/python training/v10/build_real.py --model-id kube_ordinal_v9_cal
"""
import argparse
import collections
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / 'lfm2d' / 'hooks'))
from kaish_plan import KAISH_BIN, PLAN_TIMEOUT_S, _render_command  # noqa: E402
from scrub import EMAIL_RE, SECRET_RE, scrub_command  # noqa: E402
from soak_shapes import DEFAULT_LOG, QUOTING_PROMPT_TS, load_rows, shape  # noqa: E402

VOTES = HERE / 'bulk_votes.json'
OUT = HERE / 'real.jsonl'
REVIEW = Path.home() / '.cache/claude-hooks/v10-scrub-review.txt'
LEAK_PATTERNS = [
    ('username', re.compile(r'atobey|tobert', re.I)),
    ('home path', re.compile(r'/home/|/Users/')),
    ('scratchpad slug', re.compile(r'claude-\d+/')),
    ('private host', re.compile(r'\.ts\.net|\.local\b|\.lan\b')),
    ('e-mail', EMAIL_RE),
    ('token', SECRET_RE),
]
PLACEHOLDERS = ('user@example.com', 'host.example', '<redacted>')


def plan_commands(cmd):
    """The plan's command dicts per statement, or None on any failure —
    the same subprocess the hook runs; statements with an unrenderable
    arg are skipped (the hook scores those as whole statements, which is
    not the v10 unit)."""
    try:
        proc = subprocess.run([KAISH_BIN, '--plan-file', '-'], input=cmd, capture_output=True,
                              text=True, timeout=PLAN_TIMEOUT_S)
        doc = json.loads(proc.stdout)
    except Exception:
        return None
    if proc.returncode != 0 or 'errors' in doc:
        return None
    out = []
    for stmt in doc.get('statements') or []:
        commands = (stmt.get('plan') or {}).get('commands') or []
        if commands and all(_render_command(c) is not None for c in commands):
            out.extend(commands)
    return out


def leak_check(rows):
    """[(pattern name, text)] for every scrubbed row that still carries an identifier."""
    hits = []
    for r in rows:
        text = r['text']
        for ph in PLACEHOLDERS:
            text = text.replace(ph, '')
        for name, rx in LEAK_PATTERNS:
            if rx.search(text):
                hits.append((name, r['text']))
    return hits


def build_rows(commands_by_cmd, labels, cap):
    """Pure: (rows, stats). commands_by_cmd: {command: [plan command dicts]}."""
    per_shape = collections.defaultdict(collections.Counter)   # shape -> scrubbed -> n
    examples = collections.defaultdict(dict)                    # shape -> scrubbed -> one raw
    stats = collections.Counter()
    for cmds in commands_by_cmd.values():
        for c in cmds:
            raw = _render_command(c)
            s = shape(raw)
            stats['clauses'] += 1
            if s not in labels:
                stats['unlabeled_shape'] += 1
                continue
            if labels[s] is None:
                stats['unresolved_shape'] += 1
                continue
            scrubbed = scrub_command(c)
            if scrubbed is None:
                stats['unrenderable'] += 1
                continue
            per_shape[s][scrubbed] += 1
            examples[s].setdefault(scrubbed, raw)
    rows = []
    for s, counter in per_shape.items():
        stats['unique_scrubbed'] += len(counter)
        for text, n in counter.most_common(cap):
            rows.append({'text': text, 'label': labels[s], 'shape': s, 'n': n})
    rows.sort(key=lambda r: (-r['n'], r['shape'], r['text']))
    stats['rows'] = len(rows)
    return rows, stats, examples


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', default=DEFAULT_LOG)
    ap.add_argument('--model-id', required=True)
    ap.add_argument('--until', type=float, default=QUOTING_PROMPT_TS)
    ap.add_argument('--cap', type=int, default=100,
                    help='max distinct scrubbed rows per shape (30 starved the head shapes: '
                         'candidate A read bare `echo` as data-critical)')
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--out', default=str(OUT))
    args = ap.parse_args(argv)

    votes = json.loads(VOTES.read_text())
    labels = {r['shape']: r.get('label') for r in votes['shapes']}
    if any('label' not in r for r in votes['shapes']):
        raise SystemExit('bulk_votes.json has no final labels — run apply_rulings.py first')

    rows_in = load_rows(args.log, args.model_id, until=args.until)
    commands = sorted({d['command'] for d in rows_in})
    print(f'{len(rows_in)} rows, {len(commands)} distinct commands in window')
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        plans = list(ex.map(plan_commands, commands))
    by_cmd = {c: p for c, p in zip(commands, plans) if p}
    print(f'planned {len(by_cmd)} ({len(by_cmd) / len(commands):.1%})')

    rows, stats, examples = build_rows(by_cmd, labels, args.cap)
    hits = leak_check(rows)
    if hits:
        print(f'LEAK GATE FAILED: {len(hits)} scrubbed row(s) still identify:', file=sys.stderr)
        for name, text in hits[:20]:
            print(f'  [{name}] {text[:120]}', file=sys.stderr)
        return 1

    with open(args.out, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    fd = os.open(REVIEW, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as f:
        f.write('# v10 scrub review — raw -> scrubbed, up to 3 per shape. LOCAL ONLY (real text).\n')
        for s in sorted(examples, key=lambda s: -sum(1 for r in rows if r['shape'] == s)):
            f.write(f'\n## {s}  [{labels[s]}]\n')
            for scrubbed, raw in list(examples[s].items())[:3]:
                f.write(f'  - {raw}\n  + {scrubbed}\n')
    by_label = collections.Counter(r['label'] for r in rows)
    print(json.dumps({**stats, 'by_label': dict(by_label), 'cap': args.cap}, indent=1))
    print(f'wrote {args.out} ({len(rows)} rows) and {REVIEW} (0600)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
