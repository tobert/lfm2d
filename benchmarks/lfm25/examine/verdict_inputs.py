#!/usr/bin/env python3
"""Turn a verdict_eval run back into examiner inputs, one per row.

    verdict_inputs.py RUN_DIR --prompt SPEC.json --slot verdict > inputs.jsonl

Each input is the row's exact user turn plus, as the assistant's prefill, the
fields the daemon generated BEFORE the named slot, ending at the opening quote
of that field. So the examiner stands where the model stood when it chose that
field's value, and its distribution there can be checked against the daemon's.

A REPLAY, BY DEFAULT. Both halves are the bytes the run recorded: `input` as
verdict_eval sent it, and a slice of `output` as the model generated it. Nothing
is rebuilt, so nothing can fork.

The alternative, `--reconstruct`, rebuilds the prefill from the parsed report and
the user turn from the corpus text. It is kept only for runs recorded before
verdict_eval saved the raw bytes, and it is a second rendering in two ways:

  * The parsed report is not the generated text. rows.jsonl stores the report
    with its keys SORTED, while the daemon emits the schema's `required` order,
    so the first version of this tool put `reason` -- written AFTER the verdict,
    to justify it -- in front of the verdict slot on every one of 733 rows. That
    is fixed by ordering from the spec, but the escaping is not fixable: a model
    that writes `\\u00e9` and a json.dumps that writes the character parse to the
    same report and produce different bytes.
  * build_facts reads the host's manual pages. Re-rendering the user turn later,
    or on another machine, silently reads whatever `man` says then.

The spec is checked against the sha the run recorded, so a prompt edited since
the run is an error here, not a quiet second fork. Under replay the generated
field order is checked against the spec's too.

Names are row numbers; the corpus text never becomes a file name.
"""
import argparse, hashlib, json
from collections import Counter
from pathlib import Path

WHITESPACE = ' \t\r\n'


def _skip_ws(s, i):
    while i < len(s) and s[i] in WHITESPACE:
        i += 1
    return i


def _end_of_string(s, i):
    """Index just past the closing quote of the string opening at s[i]."""
    i += 1
    while i < len(s):
        if s[i] == '\\':
            i += 2
            continue
        if s[i] == '"':
            return i + 1
        i += 1
    raise ValueError('the generated text ends inside a string')


def _end_of_value(s, i):
    if s[i] == '"':
        return _end_of_string(s, i)
    if s[i] in '{[':
        # An output schema is a closed object of string and boolean fields, so
        # this cannot arise from a grammar-constrained report. It is refused
        # rather than walked because the bare-literal walk below would stop at
        # the nested object's own `}` and silently drop every field after it --
        # a scan that reports fewer fields than the text has is the one failure
        # this whole family of tools exists to prevent.
        raise ValueError('a structured value at offset %d is not a schema field' % i)
    j = i
    while j < len(s) and s[j] not in ',}' and s[j] not in WHITESPACE:
        j += 1
    if j == i:
        raise ValueError('no value at offset %d' % i)
    return j


def field_offsets(output):
    """Offsets into `output` of the first byte of each top-level field's value,
    in the order the model wrote them.

    A scan, not a search. The model writes about the fields it has just filled,
    so searching for `"verdict": "` can land inside an earlier value; walking the
    text with escapes honoured cannot. Nested objects are not expected -- an
    output schema is a closed object of string and boolean fields -- and a
    structured value is refused rather than walked past, because the bare-literal
    walk would take the nested object's own `}` for the end of the document and
    report fewer fields than the text holds.

    `\\uXXXX` needs no special case: it is skipped as a two-byte escape like any
    other, so an escape and a surrogate pair both walk correctly. WHITESPACE is
    JSON's four bytes, which is also exactly what the grammar emits; any other
    whitespace byte between tokens would be absorbed as value content rather than
    refused, and nothing in a constrained report can produce one.
    """
    i = _skip_ws(output, 0)
    if i >= len(output) or output[i] != '{':
        raise ValueError('the generated text does not open an object: %r' % output[:40])
    i = _skip_ws(output, i + 1)
    at = {}
    if i < len(output) and output[i] == '}':
        return at
    while True:
        if i >= len(output) or output[i] != '"':
            raise ValueError('expected a field name at offset %d, got %r' % (i, output[i:i + 12]))
        end = _end_of_string(output, i)
        key = json.loads(output[i:end])
        i = _skip_ws(output, end)
        if i >= len(output) or output[i] != ':':
            raise ValueError('expected `:` after the field name %r' % key)
        i = _skip_ws(output, i + 1)
        if i >= len(output):
            raise ValueError('the generated text ends after the field name %r' % key)
        if key in at:
            raise ValueError('the field %r appears twice' % key)
        at[key] = i
        i = _skip_ws(output, _end_of_value(output, i))
        if i < len(output) and output[i] == ',':
            i = _skip_ws(output, i + 1)
            continue
        if i < len(output) and output[i] == '}':
            return at
        raise ValueError('expected `,` or `}` after the field %r' % key)


def prefill_from_output(output, order, slot):
    """The generated text, sliced at the opening quote of `slot`."""
    at = field_offsets(output)
    if list(at) != list(order):
        raise ValueError('the daemon generated the fields %s; the spec says %s'
                         % (list(at), list(order)))
    if output[at[slot]] != '"':
        raise ValueError('%r is not a string in the generated text, so it has no '
                         'opening quote to stand at' % slot)
    return output[:at[slot] + 1]


def prefill_for(report, order, slot):
    """The assistant text up to the opening quote of `slot`, in emission order.

    A RE-RENDERING of the parsed report: see the module docstring. Correct in
    field order, and not byte-exact.
    """
    if set(report) != set(order):
        raise ValueError('report fields %s are not the spec fields %s' % (sorted(report), sorted(order)))
    fields = ['%s: %s' % (json.dumps(k), json.dumps(report[k], ensure_ascii=False)) for k in order[:order.index(slot)]]
    return '{' + ', '.join(fields + ['%s: "' % json.dumps(slot)])


def input_for_row(row, order, slot, reconstruct, cov=None):
    """One examiner input: the user turn and the prefill the slot stands at."""
    if reconstruct:
        if row.get('classifier') is not None:
            raise ValueError('this row carried classifier evidence, whose order is not '
                             'stored in rows.jsonl, so its user turn cannot be rebuilt; '
                             'replay it instead')
        # Imported here, not at the top: a replay must not need holdout_eval, a
        # kaish binary or the host's manual pages to run at all.
        import sys
        prompts = str(Path(__file__).resolve().parents[1] / 'prompts')
        if prompts not in sys.path:
            sys.path.insert(0, prompts)
        import verdict_eval as V
        return {'input': V.render_input(row['text'], V.H.build_facts(row['text'], cov)),
                'assistant_prefill': prefill_for(row['report'], order, slot)}
    missing = [k for k in ('input', 'output') if k not in row]
    if missing:
        raise ValueError('this row has no %s, so there are no bytes to replay: the run '
                         'predates verdict_eval recording what it sent and got. Re-run '
                         'it, or pass --reconstruct to rebuild the prefill from the '
                         'parsed report -- a second rendering, which cannot reproduce '
                         "the model's own escaping." % ' or '.join(missing))
    return {'input': row['input'],
            'assistant_prefill': prefill_from_output(row['output'], order, slot)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('run', type=Path, help='a verdict_eval run directory (holds rows.jsonl and summary.json)')
    ap.add_argument('--prompt', type=Path, required=True, help='the prompt spec the run used')
    ap.add_argument('--slot', required=True, help='the field whose value the examiner should stand in front of')
    ap.add_argument('--reconstruct', action='store_true',
                    help='rebuild the bytes instead of replaying them; for runs recorded '
                         'before the raw text was saved (see the module docstring)')
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
        try:
            built = input_for_row(r, order, a.slot, a.reconstruct, cov)
        except ValueError as e:
            raise SystemExit(f'row {n}: {e}')
        print(json.dumps(dict(name=f'row-{n:04d}', **built)))


if __name__ == '__main__':
    main()
