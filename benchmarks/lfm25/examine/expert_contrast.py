#!/usr/bin/env python3
"""Which experts does the model use differently when it gets a row right?

    expert_contrast.py --rows RUN/rows.jsonl --examined BATCH_DIR --out DIR

Joins a verdict_eval run with a batch examination of the same rows (made by
verdict_inputs.py + `lfm25-examine --inputs-file`), then contrasts expert usage
between two groups of rows, per expert layer and expert:

  caught vs missed   gold-ask rows the model flagged, against those it allowed
  ask vs allow       every row by the verdict the model gave

Usage is read in two windows: the answer slot alone (the last token, where the
verdict is chosen) and every recorded token (the input and the model's fields).

THIS IS CORRELATION. Rows the model catches differ in content from rows it
misses -- system paths against project paths, for one -- and content sets about
half of routing by itself. A cell here is a CANDIDATE for the intervention
experiment, which is the only thing that can say an expert carries a verdict.

With 704 cells and a few dozen rows, large differences appear by chance. So
every contrast comes with a permutation null: group labels are shuffled, the
LARGEST absolute difference anywhere on the map is kept, and a cell is marked
only if it beats the 95th percentile of that maximum (family-wise, not per cell).

THE LEVER. For the cell the model uses most MORE in flagged rows, it also asks
how far that expert was from being chosen in the rows where it was not: the
amount its selection score would have to rise to enter the top k, per row. That
is the size of bias nudge that would flip that one routing choice -- and, read on
the rows we do NOT want to disturb, how unselective a static nudge would be.
Entering the top k is not changing the verdict: later layers route again.
Needs the batch to have been examined with --router-logits.

Also checks the examiner against the daemon: the verdict distribution the
examiner reads at the slot beside the one the daemon recorded when it answered.
Prints aggregates only.
"""
import argparse, json, math, random
from pathlib import Path

FLAGGED = ('ask', 'review')


def load(rows_path, examined_dir):
    rows = [json.loads(l) for l in rows_path.read_text().splitlines()]
    exams = {}
    with (examined_dir / 'examinations.jsonl').open() as f:
        for line in f:
            rec = json.loads(line)
            exams[int(rec['name'].split('-')[1])] = rec['examination']
    joined = [(r, exams[n]) for n, r in enumerate(rows) if n in exams and r['outcome'] == 'answered']
    if not joined:
        raise SystemExit('no row of the run has an examination')
    return joined


def usage(exam, window):
    """{(layer index, expert)} used in the window, and the layer list."""
    cells = set()
    for li, r in enumerate(exam['routing']):
        positions = r['experts'][-1:] if window == 'slot' else r['experts']
        for chosen in positions:
            for e in chosen:
                cells.add((li, e))
    return cells


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def nudge_needed(routing, expert):
    """At the answer slot: how much `expert`'s selection score must rise to be
    chosen. 0 if it already is. Also the full picture of that decision."""
    logits, bias = routing['logits'][-1], routing['selection_bias']
    k = len(routing['experts'][-1])
    score = [sigmoid(l) for l in logits]
    select = [s + b for s, b in zip(score, bias)]
    others = sorted((v for i, v in enumerate(select) if i != expert), reverse=True)
    need = max(0.0, others[k - 1] - select[expert])
    return need, {'score': [round(v, 4) for v in score], 'bias': [round(v, 4) for v in bias],
                  'chosen': routing['experts'][-1]}


def contrast(a, b, shape, perms, rng):
    """Per-cell usage rate in group a minus group b, with a max-|diff| null."""
    n_layers, n_experts = shape

    def rates(group):
        m = [[0.0] * n_experts for _ in range(n_layers)]
        for cells in group:
            for (l, e) in cells:
                m[l][e] += 1
        return [[v / len(group) for v in row] for row in m]

    def diff(x, y):
        rx, ry = rates(x), rates(y)
        return [[rx[l][e] - ry[l][e] for e in range(n_experts)] for l in range(n_layers)]

    observed = diff(a, b)
    pooled, maxima = a + b, []
    for _ in range(perms):
        rng.shuffle(pooled)
        d = diff(pooled[:len(a)], pooled[len(a):])
        maxima.append(max(abs(v) for row in d for v in row))
    maxima.sort()
    return observed, rates(a), rates(b), maxima[int(0.95 * (len(maxima) - 1))]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--rows', type=Path, required=True)
    ap.add_argument('--examined', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True, help='directory outside this repo')
    ap.add_argument('--perms', type=int, default=2000)
    a = ap.parse_args()
    rng = random.Random(20260917)
    joined = load(a.rows, a.examined)
    first = joined[0][1]
    layers = [r['layer'] for r in first['routing']]
    shape = (len(layers), first['routing'][0]['n_experts'])
    print('rows joined %d, expert layers %d, experts %d' % (len(joined), *shape))

    # --- examiner against daemon, at the slot
    agree = total = 0
    gaps = []
    for r, e in joined:
        if not r.get('verdict_top') or not e['top']:
            continue
        theirs = {t.strip(): lp for t, lp in r['verdict_top']}
        ours = {(t['piece'] or '').replace('Ġ', ' ').strip(): t['logprob'] for t in e['top'][0]['by_depth'][-1]}
        total += 1
        agree += max(theirs, key=theirs.get) == max(ours, key=ours.get)
        gaps += [abs(theirs[k] - ours[k]) for k in theirs if k in ours and theirs[k] > -4]
    gaps.sort()
    check = {'rows': total, 'same_top_token': agree,
             'abs_logprob_gap_p50': round(gaps[len(gaps) // 2], 4), 'abs_logprob_gap_p95': round(gaps[int(.95 * len(gaps))], 4)}
    print('examiner vs daemon at the verdict slot:', check)

    groups = {
        'caught vs missed': ([e for r, e in joined if r['gold'] == 'ask' and r['verdict'] in FLAGGED],
                             [e for r, e in joined if r['gold'] == 'ask' and r['verdict'] not in FLAGGED]),
        'ask vs allow': ([e for r, e in joined if r['verdict'] in FLAGGED],
                         [e for r, e in joined if r['verdict'] not in FLAGGED]),
    }
    out = {'layers': layers, 'n_experts': shape[1], 'layer_kinds': first['layers'],
           'examiner_vs_daemon': check, 'contrasts': []}
    for name, (ga, gb) in groups.items():
        for window in ('slot', 'all'):
            ua, ub = [usage(e, window) for e in ga], [usage(e, window) for e in gb]
            d, ra, rb, bar = contrast(ua, ub, shape, a.perms, rng)
            hits = sorted(((abs(d[l][x]), l, x) for l in range(shape[0]) for x in range(shape[1]) if abs(d[l][x]) > bar), reverse=True)
            print('\n%s | window=%s | n=%d vs %d | family-wise 95%% bar on |diff| = %.3f | cells over it: %d of %d'
                  % (name, window, len(ga), len(gb), bar, len(hits), shape[0] * shape[1]))
            for _, l, x in hits[:10]:
                print('   layer %2d expert %2d   rate %.2f vs %.2f   diff %+.2f' % (layers[l], x, ra[l][x], rb[l][x], d[l][x]))
            out['contrasts'].append({'name': name, 'window': window, 'n': [len(ga), len(gb)], 'bar': round(bar, 4),
                                     'diff': [[round(v, 4) for v in row] for row in d],
                                     'rate_a': [[round(v, 4) for v in row] for row in ra],
                                     'rate_b': [[round(v, 4) for v in row] for row in rb]})
    # --- the lever: the cell used most MORE in flagged rows, at the slot
    slot = next(c for c in out['contrasts'] if c['name'] == 'ask vs allow' and c['window'] == 'slot')
    li, up = max(((l, x) for l in range(shape[0]) for x in range(shape[1])), key=lambda c: slot['diff'][c[0]][c[1]])
    _, down = min(((li, x) for x in range(shape[1])), key=lambda c: slot['diff'][c[0]][c[1]])
    if 'logits' in first['routing'][li]:
        sets = {'missed': lambda r: r['gold'] == 'ask' and r['verdict'] not in FLAGGED,
                'allowed': lambda r: r['gold'] == 'allow' and r['verdict'] not in FLAGGED}
        needs = {k: sorted(nudge_needed(e['routing'][li], up)[0] for r, e in joined if keep(r)) for k, keep in sets.items()}
        # One real decision to draw: the missed row whose nudge is the median
        # among missed rows where the expert was NOT chosen.
        pool = sorted(((nudge_needed(e['routing'][li], up)[0], n) for n, (r, e) in enumerate(joined)
                       if sets['missed'](r) and nudge_needed(e['routing'][li], up)[0] > 0))
        need, n = pool[len(pool) // 2]
        _, picture = nudge_needed(joined[n][1]['routing'][li], up)
        def margin(routing):
            # last chosen expert's selection score minus the best one left out
            select = [sigmoid(l) + b for l, b in zip(routing['logits'][-1], routing['selection_bias'])]
            chosen = routing['experts'][-1]
            return min(select[c] for c in chosen) - max(v for i, v in enumerate(select) if i not in chosen)
        all_margins = sorted(margin(e['routing'][li]) for _, e in joined)
        out['lever'] = {'layer': layers[li], 'expert_up': up, 'expert_down': down,
                        'needs': {k: [round(v, 4) for v in vs] for k, vs in needs.items()},
                        'example': dict(picture, need=round(need, 4)),
                        'slot_margin_p50': round(all_margins[len(all_margins) // 2], 4)}
        for k, vs in needs.items():
            print('lever L%d e%d | %-8s n=%d | already chosen %d | nudge needed p50 %.3f p90 %.3f | within 0.05: %d'
                  % (layers[li], up, k, len(vs), sum(v == 0 for v in vs), vs[len(vs) // 2], vs[int(.9 * (len(vs) - 1))], sum(v <= .05 for v in vs)))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'expert_contrast.json').write_text(json.dumps(out) + '\n')
    print('\nwrote', a.out / 'expert_contrast.json')


if __name__ == '__main__':
    main()
