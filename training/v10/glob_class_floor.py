#!/usr/bin/env python3
"""kaish-lead's question (2026-08-25): if kaish 0.17 drops glob character
classes (`[0-9]*` in argv becomes a parse error), how far does our plan
floor move? Measured against the pinned baseline window.

Counts, per row and per distinct command:
  1. rows carrying an unquoted `[...]` class in ARGV position -- not a
     `[ ]` / `[[ ]]` test (those are command names, and a test has
     whitespace inside the brackets) and not a `${arr[i]}` subscript
     (stripped with the `${...}` it lives in);
  2. of those, how many PLAN today (the rows 0.17 would lose) vs are
     already unplannable (cost nothing);
  3. the class spellings and the argv0s that carry them.

Planned rows are checked on the parser's own argv words (`plain`), so
quoting is exact. Unplannable rows are checked on raw text with quoted
spans and heredoc bodies stripped -- approximate, and reported as such.

    .venv-train/bin/python training/v10/glob_class_floor.py --model-id kube_ordinal_v9_cal
"""
import argparse
import collections
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_real import plan_commands  # noqa: E402
from soak_shapes import DEFAULT_LOG, QUOTING_PROMPT_TS, load_rows  # noqa: E402

QUOTED = re.compile(r"'[^']*'|\"[^\"]*\"")
DOLLAR = re.compile(r'\$\{[^}]*\}')
# A glob character class: a set or range of plain characters, optionally
# negated. JSON lists, Python slices (`[:40]`), Rust attributes
# (`[tokio::test]`) and `[[ ]]` tests do not match: no `:`, `,`, quotes,
# braces or whitespace inside.
CLASS = re.compile(r'\[[!^]?[A-Za-z0-9_.-]+\]')
HEREDOC = re.compile(r"<<-?\s*'?\"?(\w+)'?\"?[^\n]*\n(.*?\n)?\1\s*$", re.S | re.M)


def classes_in_word(plain):
    """Glob classes in one argv word. A word with whitespace is a program
    or message, not a glob; a word starting with `[`/`{` is data; quoted
    spans and `${...}` are stripped first."""
    if re.search(r'\s', plain) or plain[:1] in '[{':
        return []
    return CLASS.findall(DOLLAR.sub('', QUOTED.sub('""', plain)))


def classes_in_raw(text):
    """Approximate: classes in raw command text outside quotes/heredocs,
    word by word under the same rules."""
    t = DOLLAR.sub('', QUOTED.sub('""', HEREDOC.sub('', text)))
    out = []
    for w in t.split():
        if w[:1] not in '[{':
            out.extend(CLASS.findall(w))
    return out


def measure(commands, plans):
    """Pure. commands: [text]; plans: [command dicts | None] aligned."""
    out = collections.Counter()
    spellings, argv0s, examples = collections.Counter(), collections.Counter(), []
    per_command = {}
    for cmd, plan in zip(commands, plans):
        hits = []
        if plan:
            for c in plan:
                for a in c.get('args') or []:
                    for cls in classes_in_word(a.get('plain') or ''):
                        hits.append((c.get('name') or '?', cls))
            kind = 'planned'
        else:
            for cls in classes_in_raw(cmd):
                hits.append(('?', cls))
            kind = 'unplannable'
        per_command[cmd] = (kind, bool(hits))
        if hits:
            out[f'{kind}_with_class'] += 1
            for name, cls in hits:
                spellings[cls] += 1
                argv0s[name] += 1
            if len(examples) < 12 and kind == 'planned':
                examples.append(cmd[:100])
        out[kind] += 1
    return out, spellings, argv0s, examples, per_command


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', default=DEFAULT_LOG)
    ap.add_argument('--model-id', required=True)
    ap.add_argument('--until', type=float, default=QUOTING_PROMPT_TS)
    ap.add_argument('--jobs', type=int, default=8)
    args = ap.parse_args(argv)
    rows = load_rows(args.log, args.model_id, until=args.until)
    commands = sorted({d['command'] for d in rows})
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        plans = list(ex.map(plan_commands, commands))
    counts, spellings, argv0s, examples, per_command = measure(commands, plans)
    row_hits = collections.Counter()
    for d in rows:
        kind, hit = per_command[d['command']]
        if hit:
            row_hits[kind] += 1
    report = {
        'window': {'rows': len(rows), 'distinct_commands': len(commands), 'until': args.until},
        'distinct_commands': dict(counts),
        'rows_with_class': {'planned': row_hits['planned'], 'unplannable': row_hits['unplannable']},
        'floor_today': round(counts['planned'] / len(commands), 4),
        'floor_after_drop': round((counts['planned'] - counts['planned_with_class']) / len(commands), 4),
        'class_spellings': spellings.most_common(20),
        'argv0_carrying_a_class': argv0s.most_common(15),
    }
    print(json.dumps(report, indent=1))
    print('\nplanned examples (first 12, truncated):', file=sys.stderr)
    for e in examples:
        print('  ' + e, file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
