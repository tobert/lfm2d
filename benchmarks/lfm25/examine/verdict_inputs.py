#!/usr/bin/env python3
"""Turn a verdict_eval run back into examiner inputs, one per row.

    verdict_inputs.py RUN_DIR --prompt SPEC.json --slot verdict > inputs.jsonl

Each input is the row's exact user turn plus, as the assistant's prefill, the
fields the daemon generated BEFORE the named slot, ending at the opening quote
of that field. So the examiner stands where the model stood when it chose that
field's value, and its distribution there can be checked against the daemon's.

FIELD ORDER COMES FROM THE PROMPT SPEC, NOT FROM rows.jsonl. The daemon emits
fields in the schema's `required` order, but it stores the parsed report with its
keys sorted. The first version of this tool replayed the stored order, which put
`reason` -- written AFTER the verdict, to justify it -- in front of the verdict
slot on every row. Every number read from those inputs saw the justification
before the decision. The spec is checked against the sha the run recorded, so a
prompt edited since the run is an error here, not a quiet second fork.

Names are row numbers; the corpus text never becomes a file name.
"""
import argparse, hashlib, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'prompts'))
import verdict_eval as V
from collections import Counter


def prefill_for(report, order, slot):
    """The assistant text up to the opening quote of `slot`, in emission order."""
    if set(report) != set(order):
        raise ValueError('report fields %s are not the spec fields %s' % (sorted(report), sorted(order)))
    fields = ['%s: %s' % (json.dumps(k), json.dumps(report[k], ensure_ascii=False)) for k in order[:order.index(slot)]]
    return '{' + ', '.join(fields + ['%s: "' % json.dumps(slot)])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('run', type=Path, help='a verdict_eval run directory (holds rows.jsonl and summary.json)')
    ap.add_argument('--prompt', type=Path, required=True, help='the prompt spec the run used')
    ap.add_argument('--slot', required=True, help='the field whose value the examiner should stand in front of')
    a = ap.parse_args()

    summary = json.loads((a.run / 'summary.json').read_text())
    sha = hashlib.sha256(a.prompt.read_bytes()).hexdigest()
    if sha != summary['prompt_sha256']:
        raise SystemExit('%s is not the prompt this run used (sha %s, run recorded %s)' % (a.prompt, sha[:12], summary['prompt_sha256'][:12]))
    schema = json.loads(a.prompt.read_text())['output_schema']
    order = schema['required']
    if a.slot not in order:
        raise SystemExit('%r is not a field of this prompt: %s' % (a.slot, order))
    if schema['properties'][a.slot].get('type') != 'string':
        raise SystemExit('%r is not a string field, so it has no opening quote to stand at' % a.slot)

    cov = Counter()
    for n, line in enumerate((a.run / 'rows.jsonl').read_text().splitlines()):
        r = json.loads(line)
        if r['outcome'] != 'answered':
            continue
        if r.get('classifier') is not None:
            raise SystemExit('this run carried classifier evidence; its order is not stored in rows.jsonl')
        try:
            prefill = prefill_for(r['report'], order, a.slot)
        except ValueError as e:
            raise SystemExit(f'row {n}: {e}')
        print(json.dumps({'name': f'row-{n:04d}',
                          'input': V.render_input(r['text'], V.H.build_facts(r['text'], cov)),
                          'assistant_prefill': prefill}))


if __name__ == '__main__':
    main()
