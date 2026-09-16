#!/usr/bin/env python3
"""Does a cached static domain preamble shift the LFM2.5 adjudicator? val_F sweep.

Amy, 2026-09-15: "what if we try caching a prompt that has done a lot more
thinking about kaish and shell commands, would that maybe shift?"

Measured first (2026-09-16, llama.cpp b9820): prefix caching DOES hit on this
model now -- 4544/5060 tokens cached, cold-vs-warm output byte-identical, a
50k-token preamble costs 0.62s warm vs 20.1s cold. So the preamble is nearly
free at inference time and the only open question is whether it helps or hurts
the judgement.

deepseek-flash and glm-5.2 both predicted it HURTS, for two stated mechanisms:
  (1) crowding -- only 6 of 24 layers are full attention, the rest carry a
      fixed-size conv/recurrent state that the preamble fills before the clause
      arrives. Prediction: dc recall declines monotonically with preamble length.
  (2) the repetition penalty is applied over full history (repeat_last_n=-1), so
      preamble tokens penalise the very words the answer needs.

Mechanism (2) is a confound, not a property of preambles, so the sweep runs the
extremes at both repeat_last_n=-1 (matching the 09-15 baseline) and =64 (the
llama.cpp default). That separates "preamble hurts" from "penalty over preamble
hurts".

Held fixed from the winning 09-15 variant (json-facts-nothink): per-clause
label-blind facts, empty think block, answer under the JSON grammar, greedy.

Eval set is val_F (733 rows, 76 data-critical) -- the classifier's own val
split, which the LLM has never seen, so it is fair to the LLM and comparable to
the classifier's 93.3%. The 84-row shape holdout is deliberately NOT reused: we
have looked at it, and its 12 dc rows make anything under 3 rows noise.

Prints aggregates only; raw rows go to --out, outside the repo.
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


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import holdout_eval as H  # noqa: E402
import preamble as P  # noqa: E402


# (name, preamble budget in chars or None, repeat_last_n)
VARIANTS = [
    ('p0',          None,   -1),   # reproduce the 09-15 winner on val_F
    ('p28k',        128000, -1),   # the biggest preamble, baseline sampling
    ('p8k',         32000,  -1),
    ('p2k',         8000,   -1),
    ('p0-pen64',    None,   64),   # penalty control, no preamble
    ('p28k-pen64',  128000, 64),   # penalty control, biggest preamble
]


def system_for(budget):
    if budget is None:
        return H.JSON_SYSTEM, 0
    body, n, _ = P.build(budget)
    # Preamble FIRST so it is the cached prefix shared by every row.
    return body + '\n' + H.JSON_SYSTEM, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path,
                    default=TRAINING / 'val_F.jsonl')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--variants', default=','.join(v[0] for v in VARIANTS))
    ap.add_argument('--limit', type=int)
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in a.data.read_text().splitlines() if l.strip()]
    if a.limit:
        rows = rows[:a.limit]
    for r in rows:
        r.setdefault('n', 1)  # val_F carries no frequency weight
    print('rows %d  labels %s' % (len(rows), dict(Counter(r['label'] for r in rows))), flush=True)

    cov = Counter()
    facts = {r['text']: H.build_facts(r['text'], cov) for r in rows}
    print('facts coverage', dict(cov), flush=True)

    want = a.variants.split(',')
    summaries = {}
    spath = a.out / 'summary.json'
    if spath.exists():
        summaries = json.loads(spath.read_text()).get('variants', {})

    for name, budget, pen in VARIANTS:
        if name not in want:
            continue
        system, nverbs = system_for(budget)
        sampling = dict(H.SAMPLING, repeat_last_n=pen)
        t0 = time.time()
        out = []
        for i, r in enumerate(rows):
            cmd = r['text']
            res = H.two_phase(system, cmd,
                              '<think>\n' + H.RUBRIC_THOUGHT + '\n' + facts[cmd],
                              'json', think=False, sampling=sampling)
            out.append(dict(res, text=cmd, label=r['label'], n=r['n']))
            if (i + 1) % 100 == 0:
                right = sum(x['severity'] == x['label'] for x in out)
                print('  %s: %d/%d right %d (%.0fs)' % (name, i + 1, len(rows), right,
                                                        time.time() - t0), flush=True)
        (a.out / f'{name}.json').write_text(json.dumps(out, indent=1))
        s = H.summarize(out)
        s['preamble_verbs'] = nverbs
        s['repeat_last_n'] = pen
        summaries[name] = s
        print(name, json.dumps(s), flush=True)
        spath.write_text(json.dumps(dict(facts_coverage=cov, variants=summaries), indent=1))


if __name__ == '__main__':
    main()
