#!/usr/bin/env python3
"""What does the SHIPPED field order actually cost?

Found 2026-09-16 while reviewing the constrained-decoding branch:
`lfm2d/prompts/shell-severity-json-v1.json` declares

    required: ["severity", "effect", "scope", "reversibility", "reason"]

severity FIRST. The eval harness has always reordered it to severity LAST --
holdout_eval.py line 24 carries the comment "severity last: fields reason
first" -- so every number measured today was taken on a configuration the
deployed daemon does not use.

Until now the order was inert: validate_schema checks `required` through a
BTreeSet and validate_report ignores order entirely, so nothing enforced it and
the model was free to emit fields in any order it liked. Under the new grammar
it becomes binding, which turns a latent mismatch into a live one.

This is the direct measurement: p0's exact configuration -- per-clause facts,
empty think block, greedy, same JSON schema -- with ONLY the field order
changed. p0 (severity last) scored 482/733 with data-critical 61/76.

    order-shipped   severity, effect, scope, reversibility, reason
    order-eval      effect, scope, reversibility, reason, severity   (= p0, control)

The control re-runs p0 rather than citing it, so any drift in the server or
the fact extractor shows up as a control that fails to reproduce 482.
"""
import os
import argparse, os, json, sys, time
from collections import Counter
from pathlib import Path

# --- repo-relative paths -------------------------------------------------
# Run outputs hold per-row verdicts, i.e. corpus rows, and corpora do not live
# in this repo. So an output directory is never defaulted into the tree: set
# LFM2D_EVAL_OUT, or pass --out. Failing loudly beats writing rows into git.
REPO = Path(__file__).resolve().parents[3]
TRAINING = Path(os.environ.get('LFM2D_TRAINING_DIR', REPO / 'training/v10'))


def eval_out(explicit=None):
    if explicit:
        return Path(explicit)
    env = os.environ.get('LFM2D_EVAL_OUT')
    if not env:
        raise SystemExit(
            'set LFM2D_EVAL_OUT (or pass --out) to a directory OUTSIDE this '
            'repo: these runs write per-row verdicts and corpora are not '
            'committed here')
    return Path(env)
# -------------------------------------------------------------------------


import holdout_eval as H  # noqa: E402


SHIPPED = ['severity', 'effect', 'scope', 'reversibility', 'reason']
EVAL = ['effect', 'scope', 'reversibility', 'reason', 'severity']
_p = H.JSON_SPEC['output_schema']['properties']


def schema(order):
    return {'type': 'object', 'properties': {k: _p[k] for k in order},
            'required': order, 'additionalProperties': False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path,
                    default=TRAINING / 'val_F.jsonl')
    ap.add_argument('--out', type=Path,
                    default=None, help='output path (default: $LFM2D_EVAL_OUT/shipped_order)')
    ap.add_argument('--limit', type=int)
    a = ap.parse_args()
    a.out = eval_out(a.out)
    a.out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in a.data.read_text().splitlines() if l.strip()]
    if a.limit:
        rows = rows[:a.limit]
    cov = Counter()
    facts = {r['text']: H.build_facts(r['text'], cov) for r in rows}
    print('rows %d %s' % (len(rows), dict(Counter(r['label'] for r in rows))), flush=True)

    for name, order in (('order-shipped', SHIPPED), ('order-eval', EVAL)):
        sch, out, t0 = schema(order), [], time.time()
        for i, r in enumerate(rows):
            prompt = (H.render(H.JSON_SYSTEM, r['text']) + '<think>\n' + H.RUBRIC_THOUGHT
                      + '\n' + facts[r['text']] + '</think>\n')
            resp = H.post('/completion', dict(H.SAMPLING, prompt=prompt,
                                              n_predict=700, json_schema=sch))
            try:
                obj = json.loads(resp['content'])
                sev, status = obj.get('severity'), 'ok'
            except Exception:
                obj, sev, status = {}, None, 'answer-unparsed'
            out.append(dict(severity=sev, status=status, secs=0.0, think_tokens=0,
                            answer_tokens=resp['tokens_predicted'], fields=obj,
                            text=r['text'], label=r['label'], n=r.get('n', 1)))
            if (i + 1) % 150 == 0:
                print('  %s: %d/%d right %d (%.0fs)'
                      % (name, i + 1, len(rows),
                         sum(x['severity'] == x['label'] for x in out), time.time() - t0),
                      flush=True)
        (a.out / f'{name}.json').write_text(json.dumps(out, indent=1))
        print(name, json.dumps(H.summarize(out)), flush=True)


if __name__ == '__main__':
    main()
