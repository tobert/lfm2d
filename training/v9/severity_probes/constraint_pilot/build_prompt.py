#!/usr/bin/env python3
"""Render the blind labeling prompt from prompt.txt + pairs.jsonl.

Strips `proposed_higher` — the authored answer must never reach a family, or
the round is not blind and its agreement numbers mean nothing.

    python3 build_prompt.py > rendered_prompt.txt
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

pairs = [json.loads(l) for l in (HERE / 'pairs.jsonl').read_text().splitlines() if l.strip()]

blocks = []
for p in pairs:
    blocks.append(
        f"### {p['id']}\n"
        f"A: {p['a']['cmd']!r}\n"
        f"B: {p['b']['cmd']!r}\n"
    )

out = (HERE / 'prompt.txt').read_text().replace('{PAIRS}', '\n'.join(blocks))

leaked = [p['id'] for p in pairs if 'proposed_higher' in out and p.get('proposed_higher') and
          f"\"proposed_higher\"" in out]
if leaked:
    print('REFUSING: authored answer leaked into the prompt', file=sys.stderr)
    sys.exit(2)

sys.stdout.write(out)
