#!/usr/bin/env python3
"""Does committing to a `writes` boolean before severity fix the sn collapse?

The diagnosis (2026-09-16): val_F situation-normal recall is 22/238. Only 4% of
those rows are read-only by the kaish parse -- 96% genuinely write. The prompt
defines informative as "read-only or display-only" in two places, and the facts
block spells the write out in words:

    CLAUSE       echo content > "${D}/f1"
    FACT GIVEN   - redirect > "${D}/f1": truncates and overwrites that file
    MODEL SAID   informative

So the model is not missing the definition or the fact; it has substituted
"harmless" for "read-only". Stating a fact does not move it.

But COMMITTING to a structured field does: the same measurement showed a bare
severity read reproduces dc only 17/40, while severity after the
effect/scope/reversibility/reason fields gets 40/40. The scaffold is what holds
the line. So make the model name the write before it names the severity.

Two variants, both adding `writes` as the FIRST field:
  w-model   the model decides the boolean itself
  w-forced  the grammar pins writes=true on clauses where the kaish parse found
            a definite write redirect, and leaves the model free everywhere
            else. The parse shapes the SCAFFOLD; it never sets the label.

Everything else is held at p0 (per-clause facts, empty think block, greedy).
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


sys.path.insert(0, str(REPO / 'lfm2d/hooks'))
import holdout_eval as H  # noqa: E402
import kaish_plan  # noqa: E402


WRITES_PROP = {'type': 'boolean',
               'description': 'True if this command creates, modifies, deletes or '
                              'overwrites anything on disk. A redirect that writes '
                              'a file counts. Only read-only or display-only '
                              'commands are false.'}
ORDER = ['writes', 'effect', 'scope', 'reversibility', 'reason', 'severity']
_p = H.JSON_SPEC['output_schema']['properties']


def schema(force_writes=None):
    props = {'writes': dict(WRITES_PROP)}
    if force_writes is not None:
        props['writes']['enum'] = [force_writes]
    props.update({k: _p[k] for k in ORDER[1:]})
    return {'type': 'object', 'properties': props, 'required': ORDER,
            'additionalProperties': False}


def parse_writes(text):
    """True only when the parse is CERTAIN: a redirect that writes a real file."""
    plan = kaish_plan.plan_clauses(text)
    if not (plan.get('ok') and plan['clauses']):
        return None
    for c in plan['clauses']:
        for r in (c.get('redirects') or []):
            kind, tgt = r.get('kind') or '', r.get('target')
            if '>' in kind and '&' not in kind and tgt != '/dev/null':
                return True
    return None  # not certain -- let the model decide


def answer(cmd, facts, force):
    prompt = (H.render(H.JSON_SYSTEM, cmd) + '<think>\n' + H.RUBRIC_THOUGHT
              + '\n' + facts + '</think>\n')
    r = H.post('/completion', dict(H.SAMPLING, prompt=prompt, n_predict=600,
                                   json_schema=schema(force)))
    try:
        obj = json.loads(r['content'])
        return obj.get('severity'), obj.get('writes'), 'ok', r['tokens_predicted']
    except Exception:
        return None, None, 'answer-unparsed', r['tokens_predicted']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path,
                    default=TRAINING / 'val_F.jsonl')
    ap.add_argument('--out', type=Path,
                    default=None, help='output path (default: $LFM2D_EVAL_OUT/writes_field)')
    ap.add_argument('--limit', type=int)
    a = ap.parse_args()
    a.out = eval_out(a.out)
    a.out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in a.data.read_text().splitlines() if l.strip()]
    if a.limit:
        rows = rows[:a.limit]
    for r in rows:
        r.setdefault('n', 1)
    cov = Counter()
    facts = {r['text']: H.build_facts(r['text'], cov) for r in rows}
    forced = {r['text']: parse_writes(r['text']) for r in rows}
    print('rows %d | parse-certain writes on %d' %
          (len(rows), sum(v is True for v in forced.values())), flush=True)

    for name in ('w-model', 'w-forced'):
        out, t0 = [], time.time()
        for i, r in enumerate(rows):
            f = forced[r['text']] if name == 'w-forced' else None
            sev, w, status, ntok = answer(r['text'], facts[r['text']], f)
            out.append(dict(severity=sev, writes=w, status=status, secs=0.0,
                            think_tokens=0, answer_tokens=ntok,
                            text=r['text'], label=r['label'], n=r['n']))
            if (i + 1) % 150 == 0:
                print('  %s: %d/%d right %d (%.0fs)'
                      % (name, i + 1, len(rows),
                         sum(x['severity'] == x['label'] for x in out),
                         time.time() - t0), flush=True)
        (a.out / f'{name}.json').write_text(json.dumps(out, indent=1))
        s = H.summarize(out)
        # does the boolean track the label the way the rubric says it should?
        wt = Counter((x['writes'], x['label']) for x in out)
        s['writes_true_rate'] = round(sum(1 for x in out if x['writes']) / len(out), 3)
        s['writes_by_label'] = {l: '%d/%d' % (
            sum(1 for x in out if x['label'] == l and x['writes']),
            sum(1 for x in out if x['label'] == l))
            for l in ['informative', 'situation-normal', 'data-critical']}
        print(name, json.dumps(s), flush=True)


if __name__ == '__main__':
    main()
