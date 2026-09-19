#!/usr/bin/env python3
"""Where in the stack does a field's value form, row by row?

    verdict_ribbon.py --rows RUN/rows.jsonl --examined BATCH_DIR --out DIR \
        --field verdict --up ask --down allow
    verdict_ribbon.py ... --field undo --up hard,impossible --down 'easy,nothing to undo'

Joins a verdict_eval run with a batch examination of the same rows (inputs from
verdict_inputs.py --slot FIELD) and keeps only the logit lens at that slot: for
every depth, the lean between two sets of the field's words and the raw mass of
the whole set. One short record per row, so a multi-GB batch becomes a few
hundred KB a page can draw. The lens set must be named after the field.

With several words on a side the lean is logsumexp(up) - logsumexp(down): the log
of the two sides' summed probability, in which the normaliser still cancels.

Rows are grouped by gold against the verdict THE DAEMON gave:

  caught        gold ask,   daemon flagged
  missed        gold ask,   daemon allowed
  false-alarm   gold allow, daemon flagged
  allowed       gold allow, daemon allowed

LEAN AND MASS ARE DIFFERENT QUESTIONS. logp(up) - logp(down) is a difference
of two logits, so the softmax normaliser cancels: it is the residual projected on
one fixed direction of the output head, and it means the same thing whether the
verdict words hold all of the mass or a millionth of it. The raw set mass says
whether the head would SAY a verdict word at that depth. Keep both: on the first
run the groups split cleanly ten layers before the mass arrived. What low mass
does poison is an ARGMAX inside the set, which is why nothing here is
renormalised and no per-depth "accuracy" is computed.

LENS CAVEAT. Every depth is read through the final norm and the output head,
which were trained on the last depth only. Early depths say what the head makes
of that residual, not what the model "believed" there.

SEPARABILITY. For each depth the record also carries an AUC: the chance that a
row drawn from one group has a higher margin than a row drawn from the other.
0.5 is no information, 1.0 is a clean split. It is a rank statistic, so it is
indifferent to the lens's scale at early depths -- but it inherits the lens
caveat in full: a low AUC says this one direction does not separate the groups
there, not that the residual could not.

CHUNK EFFECT. With --chunk N (the examiner's prefill chunk, `CHUNK` in
adjudicator.rs) the record also says how many rows leave the common line at the
first layer, and whether where the slot falls inside a chunk explains them. On
the first run it explained all of them: 41 rows, slot 1-8 tokens into a chunk,
up to 0.32 nats at layer 0. That is the examiner's chunked prefill, not the
model, and it is a standing check on the known cached-convolution problem.

FIRST-TOKEN PROXY. A word that is several tokens long is followed by its first
token alone. `ask` is one token, so its reading is exact; `easy` starts with the
token `e`, which any word beginning with e would also start with. The record
carries `first_tokens` so a page can say which readings are exact.

EXAMINER != DAEMON. The groups come from the daemon's decode; the curves come
from the examiner's cold prefill of the same bytes. They disagree on the top
verdict word for a few percent of rows; each record carries `examiner_top` so
those rows can be marked rather than hidden.

That flip count understates it. `examiner_vs_daemon.py` measures the two readings
against each other in nats: median 0.16 per word, max 4.23, and every flip sits at
a margin below the disagreement. So a lean curve here carries a 0.16-nat floor,
and the flagged rows are the ones where it happened to change a ranking, not the
only ones it touched.

The output holds row numbers and the model's own closed-choice fields. The corpus
text never leaves the run directory. Prints aggregates only.
"""
import argparse, json, math
from collections import Counter
from pathlib import Path

FLAGGED = ('ask', 'review')


def group_of(row):
    flagged = row['verdict'] in FLAGGED
    if row['gold'] == 'ask':
        return 'caught' if flagged else 'missed'
    return 'false-alarm' if flagged else 'allowed'


def logsumexp(values):
    top = max(values)
    return top + math.log(sum(math.exp(v - top) for v in values))


def auc(a, b):
    """P(a > b) over all pairs, ties counted half. By ranks, so it is n log n."""
    if not a or not b:
        return None
    ranked = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    rank_sum, i = 0.0, 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        mid = (i + j + 1) / 2          # mean of the 1-based ranks i+1 .. j
        rank_sum += mid * sum(1 for k in range(i, j) if ranked[k][1] == 0)
        i = j
    return (rank_sum - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--rows', type=Path, required=True)
    ap.add_argument('--examined', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True, help='directory outside this repo')
    ap.add_argument('--field', required=True, help='the report field the examiner stood in front of; also the lens set name')
    ap.add_argument('--up', required=True, help='WORD,WORD the lean counts toward, as the prompt spells them')
    ap.add_argument('--down', required=True, help='WORD,WORD the lean counts against')
    ap.add_argument('--note', default='', help='one line about this batch, carried into the record')
    ap.add_argument('--chunk', type=int, help="the examiner's prefill chunk length, to test for a chunk-boundary effect")
    a = ap.parse_args()

    rows = [json.loads(l) for l in a.rows.read_text().splitlines()]
    # The lens follows token ids; batch.json says which word each id stands for.
    resolution = [t for t in json.loads((a.examined / 'batch.json').read_text())['first_token_resolution'] if t['set'] == a.field]
    if not resolution:
        raise SystemExit(f'the batch followed no set named {a.field!r}')
    word_of = {t['id']: t['word'] for t in resolution}
    ups, downs = a.up.split(','), a.down.split(',')
    out = {'schema': 'lfm25-verdict-ribbon-v2', 'field': a.field, 'up': ups, 'down': downs, 'note': a.note, 'rows': []}
    with (a.examined / 'examinations.jsonl').open() as f:
        for line in f:
            rec = json.loads(line)
            n = int(rec['name'].split('-')[1])
            row, exam = rows[n], rec['examination']
            if row['outcome'] != 'answered':
                raise SystemExit(f'row {n} was examined but the run did not answer it')
            lens = exam['lens'][a.field]
            if len(lens['positions']) != 1:
                raise SystemExit(f'row {n}: expected the lens at one position, got {lens["positions"]}')
            words = [word_of[t['id']] for t in lens['tokens']]
            if 'words' not in out:
                out.update(words=words, depths=exam['depths'], layers=exam['layers'],
                           first_tokens=[{'word': t['word'], 'piece': t['piece'], 'word_tokens': t['word_tokens']} for t in resolution])
            elif out['words'] != words or out['depths'] != exam['depths']:
                raise SystemExit(f'row {n}: lens words or depths differ from the first row')
            logprob = [d[0] for d in lens['logprob']]
            for w in ups + downs:
                if w not in words:
                    raise SystemExit(f'{w!r} is not one of the lens words {words}')
            side = lambda d, names: logsumexp([d[words.index(w)] for w in names])
            wrote = row['report'][a.field]
            if wrote not in words:
                raise SystemExit(f'row {n}: the daemon wrote {wrote!r}, which the lens set {words} does not follow')
            slot = lens['positions'][0] - exam['record_from']
            out['rows'].append({
                'n': n, 'group': group_of(row), 'label': row['label'],
                'wrote': wrote, 'examiner_top': words[max(range(len(words)), key=lambda i: logprob[-1][i])],
                'scope': row['report'].get('scope'), 'undo': row['report'].get('undo'),
                'n_tokens': len(exam['tokens']),
                'lean': [round(side(d, ups) - side(d, downs), 3) for d in logprob],
                'mass_logprob': [round(d[0], 3) for d in lens['mass_logprob']],
                'residual_norm': [round(d[slot], 2) for d in exam['residual_norm']],
            })
    if not out['rows']:
        raise SystemExit('no examination found')
    out['rows'].sort(key=lambda r: r['n'])

    def margins(keep, k):
        return [r['lean'][k] for r in out['rows'] if keep(r)]

    # The model's own closed-choice field the verdict follows most closely is
    # the last one before it; its commonest value is the collapsed default.
    default_undo = Counter(r['undo'] for r in out['rows']).most_common(1)[0][0]
    contrasts = [
        ('daemon wrote %s vs %s' % (' or '.join(ups), ' or '.join(downs)), lambda r: r['wrote'] in ups, lambda r: r['wrote'] in downs),
        ('caught vs missed', lambda r: r['group'] == 'caught', lambda r: r['group'] == 'missed'),
        ('missed vs allowed, same undo (%s)' % default_undo,
         lambda r: r['group'] == 'missed' and r['undo'] == default_undo,
         lambda r: r['group'] == 'allowed' and r['undo'] == default_undo),
    ]
    out['separability'] = []
    for name, ga, gb in contrasts:
        n = [sum(map(ga, out['rows'])), sum(map(gb, out['rows']))]
        curve = [auc(margins(ga, k), margins(gb, k)) for k in range(len(out['depths']))]
        out['separability'].append({'name': name, 'n': n, 'auc': [None if v is None else round(v, 4) for v in curve]})
        print('AUC %-44s n=%3d vs %3d | %s' % (name, *n, ' '.join('--' if v is None else '%.2f' % v for v in curve)))

    # The final-depth lean read as a SCORE against gold, at the daemon's own
    # false-alarm count and a few multiples of it. A diagnostic of what the
    # argmax discards -- not a proposed gate: one cutoff across statements has
    # failed here before, and this split has been looked at.
    severe = sorted((r['lean'][-1] for r in out['rows'] if r['group'] in ('caught', 'missed')), reverse=True)
    benign = sorted((r['lean'][-1] for r in out['rows'] if r['group'] in ('false-alarm', 'allowed')), reverse=True)
    daemon_fa = sum(r['group'] == 'false-alarm' for r in out['rows'])
    out['sweep'] = {'gold_severe': len(severe), 'gold_benign': len(benign), 'daemon_false_alarms': daemon_fa,
                    'daemon_flagged': sum(r['group'] == 'caught' for r in out['rows']), 'points': []}
    for fa in sorted({max(1, daemon_fa // 3), daemon_fa, 2 * daemon_fa, 4 * daemon_fa}):
        if 0 < fa <= len(benign):
            cut = benign[fa - 1]
            out['sweep']['points'].append({'false_alarms': fa, 'lean_at_least': cut, 'severe_flagged': sum(v >= cut for v in severe)})
    print('sweep: daemon flagged %d of %d at %d false alarms | final lean as a score: %s' % (
        out['sweep']['daemon_flagged'], len(severe), daemon_fa,
        ', '.join('%d at %d' % (pt['severe_flagged'], pt['false_alarms']) for pt in out['sweep']['points'])))

    if a.chunk:
        first = margins(lambda r: True, 1)      # depth 1 = after the first layer
        modal = Counter(round(v, 3) for v in first).most_common(1)[0][0]
        off = [r for r, v in zip(out['rows'], first) if round(v, 3) != modal]
        if off:
            offsets = [r['n_tokens'] % a.chunk for r in off]
            lo, hi = min(offsets), max(offsets)
            on_line_in_range = sum(1 for r, v in zip(out['rows'], first)
                                   if round(v, 3) == modal and lo <= r['n_tokens'] % a.chunk <= hi)
            out['chunk_effect'] = {'chunk': a.chunk, 'depth': 1, 'on_line': len(first) - len(off), 'off_line': len(off),
                                   'offsets': [lo, hi], 'on_line_rows_in_those_offsets': on_line_in_range,
                                   'max_shift': round(max(abs(v - modal) for v in first), 3)}
            print('chunk effect:', out['chunk_effect'])

    groups = Counter(r['group'] for r in out['rows'])
    print('rows %d | depths %d | words %s' % (len(out['rows']), len(out['depths']), out['words']))
    print('groups:', dict(groups))
    print('examiner top word differs from the word the daemon wrote on %d rows'
          % sum(r['examiner_top'] != r['wrote'] for r in out['rows']))
    a.out.mkdir(parents=True, exist_ok=True)
    target = a.out / ('ribbon-%s.json' % a.field)
    target.write_text(json.dumps(out) + '\n')
    print('wrote', target)


if __name__ == '__main__':
    main()
