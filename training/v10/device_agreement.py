#!/usr/bin/env python3
"""Device agreement — does the severity head's VERDICT survive a device change?

Amy, 2026-09-12: "alright let's do #1" — the calibration the signoff made a
precondition for any GPU rollout of the safety classifier ("No GPU rollout
of safety classifier until device-specific calibration is evaluated").

lfm2d/tests/device_real.rs already proves the arithmetic agrees (every
score within 0.005 on a handful of fixtures). That is not the question a
safety head answers. On 2026-08-16 the same checkpoint on CPU vs ROCm
flipped 2 of 60 real verdicts (memory shadow-scorer-cpu-gpu-verdict-drift),
and this head has no global cutoff to hide a flip behind: rows near an
argmax boundary land on either side of it. So this measures VERDICTS, on
the real advisory traffic and on the v10 gate's own legs:

  collect  score every unique clause of the chosen row sets, plus the
           severity and benign probes, against ONE running daemon. The
           device is read from that daemon's own startup line, never taken
           from a flag. Scores are keyed by sha256(text): no clause text
           is ever written (training corpora never live in the repo, and
           these runs are local, 0600).
  compare  two collections of the SAME head: clause top-label flips,
           the hook's row decisions (fired / cascade winner / top), and
           per-side gate inputs in the exact formats score_probes.py
           (--results) and passthrough_gate.py (--benign-run,
           --soak-rows) already read — so each device's gate verdict comes
           from the committed gate scorers, not from a re-implementation.

Both daemons are the same binary and weights; only --device / --threads
differ. A CPU run at a different thread count is the control: it tells a
device effect apart from "any change to the reduction order".

Run with .venv-train/bin/python — row verdicts reuse clause_replay.py's
aggregation, which imports torch at module load.

Prints aggregates only (counts, label pairs, magnitudes), never a clause.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEVERITY_DIR = HERE.parent / 'v9' / 'severity_probes'
sys.path.insert(0, str(SEVERITY_DIR))
import clause_replay as cr  # noqa: E402  (imports torch; see module docstring)

SEVERITY_PROBES = SEVERITY_DIR / 'probes.jsonl'
BENIGN_PROBES = HERE / 'benign_probes.jsonl'
DEFAULT_LOG = Path('~/.cache/claude-hooks/lfm2d-advisory.jsonl').expanduser()
DEFAULT_OUT = Path('~/.cache/claude-hooks/device-agreement').expanduser()
# The v10 gate's soak window: v9_cal rows before the quoting-prompt change
# (training/v10/README.md pipeline, passthrough_gate --until).
GATE_LIVE_MODEL = 'kube_ordinal_v9_cal'
GATE_UNTIL = 1787497523
LIVE_MODEL = 'kube_ordinal_v10'
ANSI = re.compile(r'\x1b\[[0-9;]*m')
FIELD = re.compile(r'(\w+)=("[^"]*"|\S+)')


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def top_label(scores: dict, labels: list) -> str:
    """Argmax in checkpoint label order; exact ties go to the EARLIER label."""
    best = labels[0]
    for label in labels[1:]:
        if scores[label] > scores[best]:
            best = label
    return best


def _margin(scores: dict, labels: list) -> float:
    ranked = sorted((scores[l] for l in labels), reverse=True)
    return ranked[0] - ranked[1]


def _quantile(values: list, q: float) -> float:
    """Nearest-rank quantile; values must be non-empty."""
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


MARGIN_THRESHOLDS = (1e-4, 1e-3, 1e-2, 5e-2)


def compare_clauses(a: dict, b: dict, labels: list) -> dict:
    """Clause-level agreement between two {text_key: {label: p}} runs."""
    if a.keys() != b.keys():
        raise ValueError(f'runs scored different clause sets: {len(a.keys() - b.keys())} '
                         f'only in a, {len(b.keys() - a.keys())} only in b')
    flips = Counter()
    max_abs = {l: 0.0 for l in labels}
    dc_abs = []
    flip_margin_max = None
    # Headroom: a flip needs a clause whose top-two margin is smaller than
    # the device difference. Zero flips over clauses that all sit far from a
    # boundary would prove nothing, so report how close the closest calls are.
    min_margin = None
    margin_below = {f'{th:g}': 0 for th in MARGIN_THRESHOLDS}
    for k in a:
        sa, sb = a[k], b[k]
        margin_a = _margin(sa, labels)
        min_margin = margin_a if min_margin is None else min(min_margin, margin_a)
        for th in MARGIN_THRESHOLDS:
            if margin_a < th:
                margin_below[f'{th:g}'] += 1
        for l in labels:
            max_abs[l] = max(max_abs[l], abs(sa[l] - sb[l]))
        dc_abs.append(abs(sa['data-critical'] - sb['data-critical']))
        ta, tb = top_label(sa, labels), top_label(sb, labels)
        if ta != tb:
            flips[f'{ta}->{tb}'] += 1
            flip_margin_max = margin_a if flip_margin_max is None else max(flip_margin_max, margin_a)
    return {
        'clauses': len(a),
        'top_flips': sum(flips.values()),
        'flip_pairs': dict(flips),
        'max_abs_diff': max_abs,
        'dc_abs_diff_p50': _quantile(dc_abs, 0.5) if dc_abs else None,
        'dc_abs_diff_p99': _quantile(dc_abs, 0.99) if dc_abs else None,
        'flip_margin_max': flip_margin_max,
        'min_margin': min_margin,
        'margin_below': margin_below,
    }


def _row_verdict(row, scores, labels, weights):
    endpoint, texts = cr.row_inputs(row)
    probs = [[scores[text_key(t)][l] for l in labels] for t in texts]
    return endpoint, cr.aggregate_row(endpoint, probs, labels, weights)


def row_agreement(rows, a, b, labels, severe_order) -> dict:
    """The decisions the hook makes, compared row by row. An unscored
    clause raises KeyError: a row silently skipped is a row not measured."""
    weights = cr.severity_weights(labels, severe_order)
    out = {'rows': 0, 'fired_disagree': 0, 'top_disagree': 0,
           'cascade_rows': 0, 'winner_disagree': 0}
    for row in rows:
        endpoint, va = _row_verdict(row, a, labels, weights)
        _, vb = _row_verdict(row, b, labels, weights)
        out['rows'] += 1
        out['fired_disagree'] += va['fired'] != vb['fired']
        out['top_disagree'] += va['top'] != vb['top']
        if endpoint == 'cascade':
            out['cascade_rows'] += 1
            out['winner_disagree'] += va['winner'] != vb['winner']
    return out


def soak_doc(rows, scores, replayed_model, live_model_id, until) -> dict:
    """clause_replay.py --save-soak's format, for passthrough_gate --soak-rows:
    one data-critical list per CASCADE row, in row order."""
    kept = []
    for row in rows:
        endpoint, texts = cr.row_inputs(row)
        if endpoint == 'cascade':
            kept.append([scores[text_key(t)]['data-critical'] for t in texts])
    return {'replayed_model': replayed_model, 'live_model_id': live_model_id,
            'until': until, 'rows': kept}


def parse_startup(log_text: str) -> dict:
    """Device evidence from the daemon's `lfm2d: startup observability` line."""
    for raw in log_text.splitlines():
        line = ANSI.sub('', raw)
        if 'startup observability' not in line:
            continue
        fields = {k: v.strip('"') for k, v in FIELD.findall(line.split('startup observability', 1)[1])}
        try:
            return {'device_type': fields['device_type'], 'backend': fields['backend'],
                    'dtype': fields['dtype'], 'configured_threads': int(fields['configured_threads']),
                    'requested_device': fields['requested_device']}
        except KeyError as e:
            raise ValueError(f'startup observability line lacks {e}') from None
    raise ValueError('no `startup observability` line: cannot say which device scored this run')


def live_window(rows, until: float):
    """Live rows strictly before `until`. The advisory log grows while a
    collection runs, so the row set must be frozen at a moment every
    compared run had already read; rows after it have unscored clauses."""
    return [r for r in rows if r['ts'] < until]


def resolve_live_until(explicit, meta_a: dict, meta_b: dict) -> float:
    """The frozen live-row cutoff: --live-until if given, else the EARLIER
    of the two runs' recorded log reads. A flag later than either read
    would admit rows one run never saw, so it is refused, as is having
    neither a flag nor recorded reads."""
    reads = [m.get('log_read_at') for m in (meta_a, meta_b)]
    known = [r for r in reads if r is not None]
    if explicit is not None:
        if known and explicit > min(known):
            raise SystemExit(f'--live-until {explicit} is after a run read the log '
                             f'({min(known)}): rows past that read were never scored')
        return explicit
    if len(known) != 2:
        raise SystemExit('a run has no recorded log_read_at: pass --live-until, an epoch '
                         'before BOTH collections read the advisory log')
    return min(known)


def batch_identity(results: list) -> tuple:
    ids = {(r['model_id'], r['weight_hash']) for r in results}
    if len(ids) != 1:
        raise SystemExit(f'one batch answered by {len(ids)} heads {sorted(ids)!r} — refusing')
    return ids.pop()


def same_head(meta_a: dict, meta_b: dict):
    ka = (meta_a['model_id'], meta_a['weight_hash'])
    kb = (meta_b['model_id'], meta_b['weight_hash'])
    if ka != kb:
        raise SystemExit(f'comparing two different heads {ka!r} vs {kb!r} — a device '
                         'comparison must hold the weights fixed')


# -- IO ----------------------------------------------------------------------

def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def row_sets(log_path, live_until=None):
    """The two row sets, replayable text intact (clause_replay.load_rows' filter).
    live_until=None means "everything now in the log" — collect only."""
    gate = [r for r in cr.load_rows(log_path, GATE_LIVE_MODEL) if r['ts'] < GATE_UNTIL]
    live = cr.load_rows(log_path, LIVE_MODEL)
    if live_until is not None:
        live = live_window(live, live_until)
    return {'gate_window': gate, 'live_v10': live}


def corpus_texts(sets):
    texts = {}
    for rows in sets.values():
        for row in rows:
            for t in cr.row_inputs(row)[1]:
                texts[text_key(t)] = t
    for p in load_jsonl(SEVERITY_PROBES) + load_jsonl(BENIGN_PROBES):
        texts[text_key(p['cmd'])] = p['cmd']
    return texts


def classify(url, texts, timeout=900):
    body = json.dumps({'inputs': texts}).encode()
    req = urllib.request.Request(f'{url}/v1/classify', data=body,
                                 headers={'content-type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read())
    if len(out) != len(texts):
        raise SystemExit(f'{len(out)} results for {len(texts)} inputs — misaligned batch')
    return out


def write_private(path: Path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as f:
        json.dump(doc, f)
    tmp.replace(path)


def cmd_collect(args):
    device = parse_startup(Path(args.daemon_log).read_text())
    if args.expect_device and device['device_type'] != args.expect_device:
        raise SystemExit(f'daemon reports device_type={device["device_type"]!r}, '
                         f'expected {args.expect_device!r}')
    log_read_at = time.time()  # before the read: every row older than this is in the corpus
    sets = row_sets(args.log)
    texts = corpus_texts(sets)
    keys = sorted(texts)
    if args.sample:
        import random
        random.Random(args.seed).shuffle(keys)
        keys = sorted(keys[:args.sample])
    out_path = Path(args.out)
    done = {}
    meta = None
    if out_path.exists():  # resume: a two-hour CPU run must survive an interruption
        prev = json.loads(out_path.read_text())
        if prev['device'] != device:
            raise SystemExit(f'{out_path} was collected on {prev["device"]!r}, not {device!r}')
        done, meta = prev['scores'], prev['meta']
        log_read_at = prev.get('log_read_at', log_read_at)  # the corpus is the FIRST read
    todo = [k for k in keys if k not in done]
    print(f'{args.name}: {len(keys)} clauses, {len(done)} already scored, {len(todo)} to go '
          f'({device["device_type"]}/{device["backend"]} threads={device["configured_threads"]})',
          file=sys.stderr)
    t0 = time.monotonic()
    labels = None
    for i in range(0, len(todo), args.batch):
        chunk = todo[i:i + args.batch]
        res = classify(args.url, [texts[k] for k in chunk])
        mid, wh = batch_identity(res)
        if meta is None:
            meta = {'model_id': mid, 'weight_hash': wh}
        elif (meta['model_id'], meta['weight_hash']) != (mid, wh):
            raise SystemExit(f'head changed mid-run: {meta!r} -> {(mid, wh)!r}')
        for k, r in zip(chunk, res):
            done[k] = r['scores']
            labels = labels or list(r['scores'])
        n = i + len(chunk)
        if n % (args.batch * 20) == 0 or n == len(todo):
            write_private(out_path, {'name': args.name, 'device': device, 'meta': meta,
                                     'log_read_at': log_read_at, 'scores': done})
            rate = (time.monotonic() - t0) / n * 1000
            print(f'  {n}/{len(todo)}  {rate:.1f}ms/clause  eta {rate * (len(todo) - n) / 60000:.0f}min',
                  file=sys.stderr)
    write_private(out_path, {'name': args.name, 'device': device, 'meta': meta,
                             'log_read_at': log_read_at, 'scores': done})
    return 0


def probe_run(probes, scores, meta):
    return {'meta': dict(meta), 'results': {
        p['id']: {'top': top_label(scores[text_key(p['cmd'])], LABEL_ORDER),
                  'scores': scores[text_key(p['cmd'])]} for p in probes}}


def benign_run(probes, scores, meta):
    return {'model_id': meta['model_id'], 'weight_hash': meta['weight_hash'], 'results': {
        p['id']: {'cmd': p['cmd'], 'scores': scores[text_key(p['cmd'])],
                  'top': top_label(scores[text_key(p['cmd'])], LABEL_ORDER)} for p in probes}}


LABEL_ORDER = ['informative', 'situation-normal', 'data-critical']


def cmd_compare(args):
    a = json.loads(Path(args.a).read_text())
    b = json.loads(Path(args.b).read_text())
    same_head(a['meta'], b['meta'])
    shared = a['scores'].keys() & b['scores'].keys()
    sa = {k: a['scores'][k] for k in shared}
    sb = {k: b['scores'][k] for k in shared}
    summary = {
        'a': {'name': a['name'], 'device': a['device']},
        'b': {'name': b['name'], 'device': b['device']},
        'head': a['meta'],
        'clauses': compare_clauses(sa, sb, LABEL_ORDER),
        'rows': {},
    }
    if len(shared) < min(len(a['scores']), len(b['scores'])):
        summary['note'] = (f'compared the {len(shared)} clauses both runs scored '
                           f'(a={len(a["scores"])}, b={len(b["scores"])})')
    severe = cr.DEFAULT_SEVERE_ORDER
    if not args.clauses_only:
        live_until = resolve_live_until(args.live_until, a, b)
        sets = row_sets(args.log, live_until)
        summary['live_until'] = live_until
        for name, rows in sets.items():
            summary['rows'][name] = row_agreement(rows, sa, sb, LABEL_ORDER, severe)
        gate_dir = Path(args.gate_dir)
        gate_rows = sets['gate_window']
        sev, ben = load_jsonl(SEVERITY_PROBES), load_jsonl(BENIGN_PROBES)
        for side, run in (('a', a), ('b', b)):
            tag = run['name']
            write_private(gate_dir / f'probes_run_{tag}.json', probe_run(sev, run['scores'], run['meta']))
            write_private(gate_dir / f'benign_run_{tag}.json', benign_run(ben, run['scores'], run['meta']))
            write_private(gate_dir / f'soak_{tag}.json',
                          soak_doc(gate_rows, run['scores'], run['meta']['model_id'],
                                   GATE_LIVE_MODEL, GATE_UNTIL))
        summary['gate_files'] = str(gate_dir)
    text = json.dumps(summary, indent=1, sort_keys=True)
    print(text)
    if args.save:
        Path(args.save).write_text(text + '\n')
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    sub = ap.add_subparsers(dest='cmd', required=True)

    c = sub.add_parser('collect', help='score the corpus against one running daemon')
    c.add_argument('--name', required=True, help='run tag, e.g. cpu8 / rocm0 / cpu32')
    c.add_argument('--url', required=True)
    c.add_argument('--daemon-log', required=True,
                   help="that daemon's stderr; its startup line is the device evidence")
    c.add_argument('--expect-device', choices=['cpu', 'gpu'])
    c.add_argument('--log', type=Path, default=DEFAULT_LOG)
    c.add_argument('--out', required=True)
    c.add_argument('--batch', type=int, default=64)
    c.add_argument('--sample', type=int, help='score a seeded random subset (controls)')
    c.add_argument('--seed', type=int, default=20260912)

    m = sub.add_parser('compare', help='agreement between two collections of one head')
    m.add_argument('a')
    m.add_argument('b')
    m.add_argument('--log', type=Path, default=DEFAULT_LOG)
    m.add_argument('--gate-dir', default=str(DEFAULT_OUT / 'gate'))
    m.add_argument('--live-until', type=float,
                   help='freeze live rows before this epoch (default: the earlier recorded '
                        'log read of the two runs; required for runs that predate recording)')
    m.add_argument('--clauses-only', action='store_true',
                   help='skip rows and gate files (a sampled control covers no whole rows)')
    m.add_argument('--save', help='write the aggregate summary (no text) here')

    args = ap.parse_args(argv)
    return cmd_collect(args) if args.cmd == 'collect' else cmd_compare(args)


if __name__ == '__main__':
    sys.exit(main())
