#!/usr/bin/env python3
"""How far a rebuilt examiner input is from a replayed one, on a real run.

    replay_vs_rebuild.py RUN_DIR --prompt SPEC.json --slot verdict [--rows OUT]

A replay slices the bytes the run recorded. A rebuild renders them again from the
parsed report and the corpus text. This measures the gap between the two ON THE
SAME ROWS, which is the only way to say what the rebuild was costing: every
examiner number before the raw bytes were recorded came from a rebuild.

Two halves are compared separately, because they fork for different reasons:

  * THE PREFILL, rebuilt from `report`. The parse has lost the emission order
    (rows.jsonl stores keys sorted) and the model's own escaping.
  * THE USER TURN, rebuilt from `text` through build_facts, which reads the
    host's manual pages. A rebuild on another day or another machine renders
    whatever `man` says then.

Aggregates only. With --rows it writes the row NUMBERS that differ, one per line,
so the examiner can be pointed at exactly those rows; the corpus text stays in
the run directory.
"""
import argparse, hashlib, json
from collections import Counter
from pathlib import Path

from verdict_inputs import input_for_row


def why(replayed, rebuilt):
    """A short tag for how two prefills for one row differ, or None if they do not.

    The prefills end at an opening quote, so both are closed with a throwaway
    value before parsing.
    """
    if replayed == rebuilt:
        return None
    try:
        a, b = json.loads(replayed + 'x"}'), json.loads(rebuilt + 'x"}')
    except json.JSONDecodeError:
        return 'one side is not parseable'
    if list(a) != list(b):
        return 'different field order'
    if a != b:
        return 'different values'
    # Same report, different bytes: the examiner stands in front of different
    # tokens while every downstream field reads the same.
    return 'same parse, different bytes'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('run', type=Path)
    ap.add_argument('--prompt', type=Path, required=True)
    ap.add_argument('--slot', required=True)
    ap.add_argument('--rows', type=Path, help='write the differing row numbers here')
    a = ap.parse_args()

    summary = json.loads((a.run / 'summary.json').read_text())
    sha = hashlib.sha256(a.prompt.read_bytes()).hexdigest()
    if sha != summary['prompt_sha256']:
        raise SystemExit('%s is not the prompt this run used (sha %s, run recorded %s)'
                         % (a.prompt, sha[:12], summary['prompt_sha256'][:12]))
    order = json.loads(a.prompt.read_text())['output_schema']['required']

    cov = Counter()
    prefill_tags, turn_tags = Counter(), Counter()
    differing, escaped_in_replay, total = [], 0, 0
    for n, line in enumerate((a.run / 'rows.jsonl').read_text().splitlines()):
        r = json.loads(line)
        if r['outcome'] != 'answered':
            continue
        total += 1
        played = input_for_row(r, order, a.slot, reconstruct=False)
        built = input_for_row(r, order, a.slot, reconstruct=True, cov=cov)
        tag = why(played['assistant_prefill'], built['assistant_prefill'])
        prefill_tags[tag or 'identical'] += 1
        turn_tags['identical' if played['input'] == built['input'] else 'differs'] += 1
        if '\\' in played['assistant_prefill']:
            escaped_in_replay += 1
        if tag or played['input'] != built['input']:
            differing.append(n)

    print('answered rows      %d' % total)
    print('prefill            %s' % dict(prefill_tags))
    print('user turn          %s' % dict(turn_tags))
    print('replayed prefills carrying a backslash escape: %d' % escaped_in_replay)
    print('rows where either half differs: %d' % len(differing))
    if a.rows:
        a.rows.write_text(''.join('%d\n' % n for n in differing))
        print('wrote', a.rows)


if __name__ == '__main__':
    main()
