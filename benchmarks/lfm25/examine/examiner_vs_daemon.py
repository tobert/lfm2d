#!/usr/bin/env python3
"""Why the examiner and the daemon disagree at an answer slot, in nats.

    examiner_vs_daemon.py --rows RUN/rows.jsonl --examined EX --field verdict

The examiner stands where the daemon stood and reads the same slot, so the two
should agree exactly: same weights, same bytes, greedy. They do not, on a small
percentage of rows, and counting the flips says nothing about why. This measures
the distance between the two distributions instead.

Both sides are a log_softmax over the full vocabulary, so they are directly
comparable: the daemon's raw top-k at the slot (raw = before the grammar mask and
before the repetition penalty) against the examiner's lens at its FINAL depth,
which is the model's own output distribution (examine.rs pins it to a reference).

WHAT A DISAGREEMENT MEANS depends on the margin it happened at, so the flip count
is printed against the size of the disagreement rather than on its own. The two
paths are not the same arithmetic, and the reason is the CACHE SCHEDULE: a run is
warm, so its chunk boundaries sit at `cached_tokens + 128k`, while the examiner is
cold from zero. Measured with `verdict_eval.py --no-cache`, the daemon differs from
ITSELF by the same amount on the same bytes -- p50 0.145 nats, 13 of 40 rows
generating identical text, one verdict moved. So this tool measures a real property
of the stack, not a defect in the examiner.

MEASURED, 733 val_F rows at the verdict slot (2026-09-19, ROCm, the 09-17 batch):
per-word |delta| p50 **0.16 nats**, p95 1.25, max 4.23, and 11 flips (1.5%), every
one of them at a daemon margin below 0.57. So the flips are not the finding -- the
0.16 is. Every lens number read through the examiner inherits it, and `--chunk 128`
is FLAT across the slot's offset into a prefill chunk (p50 0.34-0.48 in every
bucket), so it is not the layer-0 cached-convolution effect that
`verdict_ribbon.py --chunk` tracks -- the misalignment is a whole-schedule shift
(prefix 327, `327 % 128 = 71`), not a boundary artifact.

Aggregates and row numbers only.
"""
import argparse, json
from pathlib import Path


def compare(daemon_top, examiner, piece_of):
    """One row: {word: (daemon_logprob, examiner_logprob)} plus the two argmaxes.

    `daemon_top` is verdict_eval's [[piece, logprob], ...] and `piece_of` maps each
    followed word to the first-token piece the batch resolved it to, so the two
    sides are matched on that piece EXACTLY. A prefix match would be ambiguous --
    `a` is the start of both `allow` and `ask` -- and guessing which word the
    daemon meant is exactly the kind of quiet reconstruction this family of tools
    exists to stop. A word the daemon's top-k never reached is absent rather than
    imputed.
    """
    if len(set(piece_of.values())) != len(piece_of):
        raise ValueError('two followed words share a first token: %r' % piece_of)
    word_of_piece = {p: w for w, p in piece_of.items()}
    seen = {}
    for piece, logprob in daemon_top:
        word = word_of_piece.get(piece)
        if word is not None and word not in seen:
            seen[word] = logprob
    both = {w: (seen[w], examiner[w]) for w in piece_of if w in seen and w in examiner}
    d_top = max(seen, key=seen.get) if seen else None
    e_top = max(examiner, key=examiner.get) if examiner else None
    return both, d_top, e_top


def margin(values):
    """The winner's lead over the runner-up, in nats. None if there is no pair."""
    ranked = sorted(values, reverse=True)
    return None if len(ranked) < 2 else ranked[0] - ranked[1]


def quantile(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--rows', type=Path, required=True)
    ap.add_argument('--examined', type=Path, required=True, help='a directory holding examinations.jsonl and batch.json')
    ap.add_argument('--field', required=True, help='the lens set name, and the report field it stands in front of')
    ap.add_argument('--out', type=Path, help='write the per-row record here')
    ap.add_argument('--chunk', type=int,
                    help="the examiner's prefill chunk length: also report the delta by the "
                         "slot's offset into a chunk, which separates a boundary effect from a "
                         'flat path difference')
    a = ap.parse_args()

    rows = [json.loads(l) for l in a.rows.read_text().splitlines()]
    resolution = [t for t in json.loads((a.examined / 'batch.json').read_text())['first_token_resolution']
                  if t['set'] == a.field]
    if not resolution:
        raise SystemExit(f'the batch followed no set named {a.field!r}')
    word_of = {t['id']: t['word'] for t in resolution}
    piece_of = {t['word']: t['piece'] for t in resolution}

    per_row, deltas, flips = [], [], []
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
            final = dict(zip(words, lens['logprob'][-1][0]))
            if row[a.field] is None or row.get('%s_top' % a.field) is None:
                continue
            both, d_top, e_top = compare(row['%s_top' % a.field], final,
                                         {w: piece_of[w] for w in words})
            if d_top != row[a.field]:
                raise SystemExit('row %d: the daemon wrote %r but its own raw top-k ranks %r first; '
                                 'the mask or the penalty moved this choice, so the comparison below '
                                 'is not apples to apples' % (n, row[a.field], d_top))
            delta = {w: e - d for w, (d, e) in both.items()}
            rec = {'n': n, 'wrote': row[a.field], 'examiner_top': e_top,
                   'n_tokens': len(exam['tokens']),
                   # The daemon's own chunk boundaries sit at cached_tokens + 128k,
                   # not at 128k, so its offset is the examiner's shifted by the
                   # resident prefix. Recorded per row because the shift is only a
                   # constant while every row shares one prefix.
                   'cached_tokens': row.get('cached_tokens'),
                   'delta': {w: round(v, 4) for w, v in delta.items()},
                   'spread': None if len(delta) < 2 else round(max(delta.values()) - min(delta.values()), 4),
                   'daemon_margin': None if len(both) < 2 else round(margin([d for d, _ in both.values()]), 4)}
            per_row.append(rec)
            deltas += [abs(e - d) for d, e in both.values()]
            if e_top != row[a.field]:
                flips.append(rec)

    # The yardstick a flip has to clear is the SPREAD of the deltas within its own
    # row -- what separates two words is what can reorder them -- and it is
    # estimated on the rows that did NOT flip, so the flips are not helping to
    # excuse themselves. Using a pooled per-word p99 over all rows does both
    # wrong: it compares the wrong quantity, and the flipped rows are a large part
    # of the top percentile they are then measured against.
    steady = [r['spread'] for r in per_row if r['examiner_top'] == r['wrote'] and r['spread'] is not None]
    noise = quantile(steady, 0.99) or 0.
    unmeasurable = [r for r in flips if r['daemon_margin'] is None]
    explained = [r for r in flips if r['daemon_margin'] is not None and r['daemon_margin'] <= noise]
    wider = [r for r in flips if r['daemon_margin'] is not None and r['daemon_margin'] > noise]
    print('rows compared            %d' % len(per_row))
    print('per-word |delta| nats    p50 %.4f  p95 %.4f  p99 %.4f  max %.4f'
          % tuple(quantile(deltas, q) or 0. for q in (.5, .95, .99, 1.)))
    print('within-row delta spread  p50 %.4f  p99 %.4f   (on the %d rows that agree)'
          % (quantile(steady, .5) or 0., noise, len(steady)))
    print('argmax disagreements     %d (%.2f%%)'
          % (len(flips), 100. * len(flips) / max(1, len(per_row))))
    print('  margin below that p99  %d  <- the paths disagree by more than the row does'
          % len(explained))
    print('  margin above it        %d  <- not accounted for by the path difference'
          % len(wider))
    if unmeasurable:
        print('  margin unmeasurable    %d  <- fewer than two words matched; says nothing either way'
              % len(unmeasurable))
    if flips:
        ms = [r['daemon_margin'] for r in flips if r['daemon_margin'] is not None]
        if ms:
            print('  their daemon margins   p50 %.4f  max %.4f' % (quantile(ms, .5) or 0., max(ms)))
        print('  rows                   %s' % [r['n'] for r in flips])
    if a.chunk:
        # A boundary effect concentrates at low offsets; a path difference does not
        # care where the slot fell. Bucketed by the DAEMON's offset where the row
        # records its prefix, since that is the schedule whose boundaries are in
        # question; the examiner's own offset differs by cached_tokens mod chunk.
        # Width 8 because the known conv-head effect is a 1-8 token window, which a
        # sixteenth of a chunk cannot resolve.
        offsets = {r['cached_tokens'] for r in per_row}
        shifted = len(offsets) == 1 and None not in offsets
        buckets, width = {}, 8
        for r in per_row:
            worst = max((abs(v) for v in r['delta'].values()), default=None)
            if worst is None:
                continue
            base = r['n_tokens'] - (r['cached_tokens'] or 0)
            buckets.setdefault((base % a.chunk) // width * width, []).append(worst)
        print("per-row max |delta| by the slot's offset into a %d-token chunk%s:"
              % (a.chunk, ", the daemon's" if shifted else " (prefix varies or is unrecorded)"))
        for lo in sorted(buckets):
            v = buckets[lo]
            print('  %4d-%4d  n=%3d  p50 %.3f  p90 %.3f  max %.3f'
                  % (lo, lo + width - 1, len(v), quantile(v, .5), quantile(v, .9), max(v)))
        if shifted:
            # A single shared prefix makes the daemon's residues a relabelling of
            # the examiner's, so flatness in one is flatness in the other. It is
            # worth saying which, because prefix mod chunk != 0 means even the
            # resident prefix is built with a different last-chunk shape on the
            # two sides.
            print('  prefix %d, so prefix %% %d = %d: the two schedules are misaligned by that much'
                  % (next(iter(offsets)), a.chunk, next(iter(offsets)) % a.chunk))
    if a.out:
        a.out.write_text(json.dumps({'schema': 'lfm25-examiner-vs-daemon-v1', 'field': a.field,
                                     'noise_p99': noise, 'rows': per_row}) + '\n')
        print('wrote', a.out)


if __name__ == '__main__':
    main()
