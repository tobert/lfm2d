#!/usr/bin/env python3
"""Read a cached expert map: which knockouts move the verdict, and which flip it.

    knockout_report.py MAP.json --up ask --down allow [--contrast CONTRAST.json] [--out DIR]

The map (from `lfm25-expert-map`) holds, per probe, the change in each followed
token's log-probability when one chosen expert is knocked out at the answer slot.
This reads it as ONE number per knockout: the shift in the margin
logp(up) - logp(down). Negative means the expert was pushing toward `up`.

Per (layer, expert) it reports the mean shift over the probes where the router
chose it, and FLIPS: probes whose top followed token changed. A flip is the
concrete event; a mean shift of 0.1 nats that never flips anything is decoration.

With --contrast, it lines the causal map up against the correlational one
(expert_contrast.py): do the experts that TRAVEL with a verdict also CARRY it?

Prints aggregates. Probe names are row numbers, never corpus text.
"""
import argparse, json
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('map', type=Path)
    ap.add_argument('--up', required=True, help='followed word the margin counts toward')
    ap.add_argument('--down', required=True, help='followed word the margin counts against')
    ap.add_argument('--contrast', type=Path, help='expert_contrast.json to compare with')
    ap.add_argument('--out', type=Path, help='directory for knockout_report.json, outside this repo')
    a = ap.parse_args()
    rec = json.loads(a.map.read_text())
    words = [f['word'] for f in rec['follow']]
    up, down = words.index(a.up), words.index(a.down)
    m = rec['map']
    print('map %s | %s | backend %s | %d probes | %.0fs'
          % (rec['identity']['key'][:16], rec['identity']['prompt'], rec['identity']['backend'], len(m['probes']), rec['seconds']))

    cells = defaultdict(lambda: {'n': 0, 'shift': 0.0, 'flips': 0, 'by_group': defaultdict(lambda: [0, 0.0, 0])})
    groups = defaultdict(int)
    for p in m['probes']:
        g = p['group'] or 'ungrouped'
        groups[g] += 1
        top = max(range(len(words)), key=lambda i: p['baseline'][i])
        for k in p['knockouts']:
            after = [b + d for b, d in zip(p['baseline'], k['delta'])]
            shift = k['delta'][up] - k['delta'][down]
            flipped = max(range(len(words)), key=lambda i: after[i]) != top
            c = cells[(k['layer'], k['expert'])]
            c['n'] += 1; c['shift'] += shift; c['flips'] += flipped
            bg = c['by_group'][g]
            bg[0] += 1; bg[1] += shift; bg[2] += flipped
    print('groups', dict(groups))
    total_flips = sum(c['flips'] for c in cells.values())
    total = sum(c['n'] for c in cells.values())
    print('knockouts %d | verdict flips %d (%.1f%%)' % (total, total_flips, 100 * total_flips / total))

    ranked = sorted(cells.items(), key=lambda kv: -abs(kv[1]['shift'] / kv[1]['n']) * (kv[1]['n'] >= 8))
    print('\nlargest mean shift in logp(%s) - logp(%s), cells chosen in >= 8 probes:' % (a.up, a.down))
    print('  layer expert   n   mean shift   flips   by group: n / mean shift / flips')
    for (layer, expert), c in ranked[:14]:
        by = '  '.join('%s %d/%+.2f/%d' % (g, v[0], v[1] / v[0], v[2]) for g, v in sorted(c['by_group'].items()))
        print('  %5d %6d %3d   %+10.3f   %5d   %s' % (layer, expert, c['n'], c['shift'] / c['n'], c['flips'], by))

    by_layer = defaultdict(lambda: [0, 0.0, 0])
    for (layer, _), c in cells.items():
        by_layer[layer][0] += c['n']; by_layer[layer][1] += sum(abs(v[1]) for v in c['by_group'].values()); by_layer[layer][2] += c['flips']
    print('\nflips by layer:', '  '.join('%d:%d' % (l, v[2]) for l, v in sorted(by_layer.items())))

    out = {'identity': rec['identity'], 'up': a.up, 'down': a.down, 'layers': m['layers'], 'groups': dict(groups),
           'cells': [{'layer': l, 'expert': e, 'n': c['n'], 'mean_shift': round(c['shift'] / c['n'], 4), 'flips': c['flips'],
                      'by_group': {g: {'n': v[0], 'mean_shift': round(v[1] / v[0], 4), 'flips': v[2]} for g, v in c['by_group'].items()}}
                     for (l, e), c in sorted(cells.items())]}
    if a.contrast:
        con = json.loads(a.contrast.read_text())
        slot = next(c for c in con['contrasts'] if c['name'] == 'ask vs allow' and c['window'] == 'slot')
        ringed = [(con['layers'][i], x, slot['diff'][i][x]) for i in range(len(con['layers']))
                  for x in range(con['n_experts']) if abs(slot['diff'][i][x]) > slot['bar']]
        print('\nthe %d cells the contrast ringed, read causally (usage diff -> knockout mean shift, flips):' % len(ringed))
        agree = 0
        for layer, x, diff in sorted(ringed, key=lambda t: -abs(t[2])):
            c = cells.get((layer, x))
            if not c:
                print('  layer %2d expert %2d  diff %+.2f -> never chosen in these probes' % (layer, x, diff)); continue
            shift = c['shift'] / c['n']
            # used MORE in flagged rows and knocking it out LOWERS ask: travels with and carries
            same = (diff > 0) == (shift < 0)
            agree += same
            print('  layer %2d expert %2d  diff %+.2f -> shift %+.3f  flips %d of %d  %s'
                  % (layer, x, diff, shift, c['flips'], c['n'], 'same direction' if same else 'OPPOSITE'))
        print('  same direction: %d of %d' % (agree, len(ringed)))
        out['ringed'] = [{'layer': l, 'expert': x, 'usage_diff': d} for l, x, d in ringed]
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / 'knockout_report.json').write_text(json.dumps(out) + '\n')
        print('\nwrote', a.out / 'knockout_report.json')


if __name__ == '__main__':
    main()
