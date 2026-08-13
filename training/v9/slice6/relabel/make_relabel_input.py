#!/usr/bin/env python3
"""Render a blind relabel prompt for one slice-6 incoming file.

Blind means the family sees ONLY the text. The generator's proposed label,
its note, and its author tag are all stripped — a relabel that can see the
proposal is not an independent second opinion, it is a confirmation bias
harness.

    python3 make_relabel_input.py flagladder > rendered_flagladder.txt

Row ids are positional (`r001`...) and stable for a given input file, so the
scorer can join answers back without ever having shown a label.
"""
import sys
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
INCOMING = HERE.parent / 'incoming'
RUBRIC = HERE.parent.parent / 'labeler_prompt.txt'

HEADER = """You are labeling shell statements that an execution gate will see.

Assign each statement exactly one label measuring **data recoverability** —
what would happen to durable data if the statement were carried out.

{rubric}

## Your task

Label every statement below. Output **JSON Lines only** — one object per row,
no prose, no markdown fences:

{{"id": "<row id>", "label": "informative" | "situation-normal" | "data-critical", "why": "<one short sentence>"}}

Emit exactly one line per row, covering every id. Judge the statement exactly
as written, on a Linux workstation, assuming no backups beyond what the text
itself shows.

## Statements

"""


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    stem = sys.argv[1]
    src = INCOMING / f'{stem}.jsonl'
    rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]

    # Optional 1-based inclusive slice: make_relabel_input.py flagladder 1 26
    # Ids stay GLOBAL (r001..) so chunks join back without renumbering. A
    # 78-row ask made gemini-3.5-flash truncate at r032 mid-token and emit
    # visible reasoning; ~26 rows per call keeps the output well-formed.
    lo, hi = 1, len(rows)
    if len(sys.argv) >= 4:
        lo, hi = int(sys.argv[2]), int(sys.argv[3])
    sel = [(i, r) for i, r in enumerate(rows, 1) if lo <= i <= hi]

    # Rubric: labels + decision rules, up to the output-format section.
    rubric = RUBRIC.read_text()
    cut = rubric.find('## Output')
    if cut > 0:
        rubric = rubric[:cut].rstrip()

    out = [HEADER.format(rubric=rubric)]
    for i, r in sel:
        out.append(f'r{i:03d}: {r["text"]!r}')
    body = '\n'.join(out) + '\n'

    # Leak check: the proposed label and note must not reach the family. The
    # label VOCABULARY legitimately appears in the rubric, so check that no
    # row's own note text made it through instead.
    for _, r in sel:
        if r['note'] and r['note'] in body:
            print(f'REFUSING: generator note leaked into prompt: {r["note"][:60]!r}',
                  file=sys.stderr)
            sys.exit(2)

    sys.stdout.write(body)


if __name__ == '__main__':
    main()
