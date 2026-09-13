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
           from a flag; --url must be a port that same log says it serves,
           and every checkpoint re-reads the log so a restart onto another
           device is refused. The label vocabulary comes from the daemon's
           GET /v1/models. Scores are keyed by sha256(text): no clause text
           is ever written (training corpora never live in the repo, and
           these runs are local, 0600).
  compare  two collections of the SAME head: clause top-label flips,
           the hook's row decisions (fired / cascade winner / top) with the
           winner calls' own headroom, and per-side gate inputs in the
           exact formats score_probes.py (--results) and passthrough_gate.py
           (--benign-run, --soak-rows) already read — so each device's gate
           verdict comes from the committed gate scorers, not from a
           re-implementation. Runs that scored different clause sets are
           refused unless --intersect, which records what was dropped.

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
from urllib.parse import urlsplit

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
TCP_LISTENER = re.compile(r'will serve on tcp\b.*?\baddr=(\S+)')
# Not this tool's vocabulary: the label passthrough_gate.py hard-codes on
# every leg (row_passes, split_run). Its soak files must be written in it.
SOAK_GATE_LABEL = 'data-critical'


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


def _below_counts():
    return {f'{th:g}': 0 for th in MARGIN_THRESHOLDS}


def _count_below(counts: dict, margin: float):
    for th in MARGIN_THRESHOLDS:
        if margin < th:
            counts[f'{th:g}'] += 1


def compare_clauses(a: dict, b: dict, labels: list) -> dict:
    """Clause-level agreement between two {text_key: {label: p}} runs."""
    if a.keys() != b.keys():
        raise ValueError(f'runs scored different clause sets: {len(a.keys() - b.keys())} '
                         f'only in a, {len(b.keys() - a.keys())} only in b')
    flips = Counter()
    max_abs = {l: 0.0 for l in labels}
    abs_diffs = {l: [] for l in labels}
    flip_margin_max = None
    # Headroom: a flip needs a clause whose top-two margin is smaller than
    # the device difference. Zero flips over clauses that all sit far from a
    # boundary would prove nothing, so report how close the closest calls are.
    min_margin = None
    margin_below = _below_counts()
    for k in a:
        sa, sb = a[k], b[k]
        margin_a = _margin(sa, labels)
        min_margin = margin_a if min_margin is None else min(min_margin, margin_a)
        _count_below(margin_below, margin_a)
        for l in labels:
            d = abs(sa[l] - sb[l])
            max_abs[l] = max(max_abs[l], d)
            abs_diffs[l].append(d)
        ta, tb = top_label(sa, labels), top_label(sb, labels)
        if ta != tb:
            flips[f'{ta}->{tb}'] += 1
            flip_margin_max = margin_a if flip_margin_max is None else max(flip_margin_max, margin_a)
    return {
        'clauses': len(a),
        'top_flips': sum(flips.values()),
        'flip_pairs': dict(flips),
        'max_abs_diff': max_abs,
        'abs_diff_p50': {l: _quantile(v, 0.5) for l, v in abs_diffs.items()} if a else None,
        'abs_diff_p99': {l: _quantile(v, 0.99) for l, v in abs_diffs.items()} if a else None,
        'flip_margin_max': flip_margin_max,
        'min_margin': min_margin,
        'margin_below': margin_below,
    }


def row_agreement(rows, a, b, labels, severe_order, fired) -> dict:
    """The decisions the hook makes, compared row by row. An unscored
    clause raises KeyError: a row silently skipped is a row not measured.

    The clause-margin bound covers top-label flips only. Which clause WINS a
    cascade turns on clause severities instead, so this also reports the
    closest winner call on side a (winner severity minus the best rival
    whose text differs — identical texts score identically and cannot swap)
    next to the largest clause severity change between the runs. A swap to a
    rival with the same side-a top label changes which clause is shown, so
    the closest call against a rival whose top differs is reported separately
    (top_changing_winner_*). That alone is not the whole top-change bound: a
    same-top rival whose OWN top flips can swap in and change the top, which
    compare_clauses' min_margin vs max_abs_diff bounds. Cite both."""
    weights = cr.severity_weights(labels, severe_order)
    out = {'rows': 0, 'fired_disagree': 0, 'top_disagree': 0,
           'cascade_rows': 0, 'winner_disagree': 0, 'max_severity_abs_diff': 0.0}
    for prefix in ('winner', 'top_changing_winner'):
        out.update({f'{prefix}_margin_rows': 0, f'min_{prefix}_margin': None,
                    f'{prefix}_margin_below': _below_counts()})
    for row in rows:
        endpoint, texts = cr.row_inputs(row)
        keys = [text_key(t) for t in texts]
        pa = [[a[k][l] for l in labels] for k in keys]
        pb = [[b[k][l] for l in labels] for k in keys]
        va = cr.aggregate_row(endpoint, pa, labels, weights, fired)
        vb = cr.aggregate_row(endpoint, pb, labels, weights, fired)
        out['rows'] += 1
        out['fired_disagree'] += va['fired'] != vb['fired']
        out['top_disagree'] += va['top'] != vb['top']
        if endpoint != 'cascade':
            continue
        out['cascade_rows'] += 1
        out['winner_disagree'] += va['winner'] != vb['winner']
        sev_a = [cr.clause_severity(p, labels, weights) for p in pa]
        sev_b = [cr.clause_severity(p, labels, weights) for p in pb]
        out['max_severity_abs_diff'] = max(out['max_severity_abs_diff'],
                                           *(abs(x - y) for x, y in zip(sev_a, sev_b)))
        w = va['winner']
        tops = [top_label(a[k], labels) for k in keys]
        rivals = [i for i, k in enumerate(keys) if k != keys[w]]
        if rivals:
            _record_margin(out, 'winner', sev_a[w] - max(sev_a[i] for i in rivals))
        changing = [i for i in rivals if tops[i] != tops[w]]
        if changing:
            _record_margin(out, 'top_changing_winner', sev_a[w] - max(sev_a[i] for i in changing))
    return out


def _record_margin(out: dict, prefix: str, margin: float):
    out[f'{prefix}_margin_rows'] += 1
    m = out[f'min_{prefix}_margin']
    out[f'min_{prefix}_margin'] = margin if m is None else min(m, margin)
    _count_below(out[f'{prefix}_margin_below'], margin)


def soak_label(fired) -> str:
    """passthrough_gate's soak file holds ONE score per clause: the fired label's.
    The gate reads SOAK_GATE_LABEL on every leg, so any other label is refused
    rather than written into a file the gate would misread."""
    # One exact-set check, not "one label" then "that label": a set holding
    # the gate label plus another must not slip through on iteration order.
    if set(fired) != {SOAK_GATE_LABEL}:
        raise SystemExit(f'passthrough_gate.py reads one {SOAK_GATE_LABEL!r} score per clause from '
                         f'a soak file; fired set {sorted(fired)!r} would be misread against its floor')
    return SOAK_GATE_LABEL


def soak_doc(rows, scores, label, replayed_model, live_model_id, until) -> dict:
    """clause_replay.py --save-soak's format, for passthrough_gate --soak-rows:
    one `label` score list per CASCADE row, in row order."""
    kept = []
    for row in rows:
        endpoint, texts = cr.row_inputs(row)
        if endpoint == 'cascade':
            kept.append([scores[text_key(t)][label] for t in texts])
    return {'replayed_model': replayed_model, 'live_model_id': live_model_id,
            'until': until, 'rows': kept}


def parse_startup(log_text: str) -> dict:
    """Device evidence from the daemon's `lfm2d: startup observability` line.
    A restart appends another line; if any two disagree the log cannot say
    which device scored a given batch, so that is refused."""
    found = []
    for raw in log_text.splitlines():
        line = ANSI.sub('', raw)
        if 'startup observability' not in line:
            continue
        fields = {k: v.strip('"') for k, v in FIELD.findall(line.split('startup observability', 1)[1])}
        try:
            found.append({'device_type': fields['device_type'], 'backend': fields['backend'],
                          'dtype': fields['dtype'], 'configured_threads': int(fields['configured_threads']),
                          'requested_device': fields['requested_device'],
                          'device_index': fields['device_index']})
        except KeyError as e:
            raise ValueError(f'startup observability line lacks {e}') from None
    if not found:
        raise ValueError('no `startup observability` line: cannot say which device scored this run')
    if any(d != found[0] for d in found[1:]):
        raise ValueError(f'daemon log holds {len(found)} startup lines that disagree '
                         f'({found[0]!r} vs {next(d for d in found if d != found[0])!r}): '
                         'restart the daemon with a fresh log')
    return found[0]


def check_url_matches_log(url: str, log_text: str):
    """--url must reach the daemon whose log is the device evidence. Ports
    are the checkable link: the log names each TCP listener it bound."""
    ports = set()
    for raw in log_text.splitlines():
        m = TCP_LISTENER.search(ANSI.sub('', raw))
        if m:
            ports.add(int(m.group(1).rsplit(':', 1)[1]))
    if not ports:
        raise SystemExit('daemon log shows no tcp listener: cannot tie --url to its device evidence')
    parts = urlsplit(url)
    port = parts.port or {'http': 80, 'https': 443}[parts.scheme]
    if port not in ports:
        raise SystemExit(f'--url port {port} is not a port this daemon log serves ({sorted(ports)}): '
                         'the scores would come from a different daemon than the device evidence')


def classifier_labels(models: list, model_id: str, weight_hash: str) -> list:
    """The head's label vocabulary, in checkpoint order, from GET /v1/models."""
    match = [m for m in models if m['id'] == model_id and m['weight_hash'] == weight_hash]
    if len(match) != 1:
        raise SystemExit(f'/v1/models lists {len(match)} models with id={model_id!r} and the '
                         'answering weight hash — cannot read its labels')
    if not match[0].get('labels'):
        raise SystemExit(f'/v1/models entry {model_id!r} carries no labels')
    return list(match[0]['labels'])


def check_score_labels(results: list, labels: list):
    for r in results:
        if set(r['scores']) != set(labels):
            raise SystemExit(f'scores keyed {sorted(r["scores"])!r}, but /v1/models says {labels!r}')


def resolve_labels(explicit, meta_a: dict, meta_b: dict) -> list:
    """The label vocabulary both runs were scored in: the recorded one, or
    --labels for runs that predate recording. Disagreement is refused."""
    known = [m['labels'] for m in (meta_a, meta_b) if m.get('labels') is not None]
    if len(known) == 2 and known[0] != known[1]:
        raise SystemExit(f'runs recorded different labels: {known[0]!r} vs {known[1]!r}')
    if explicit is not None:
        if any(k != list(explicit) for k in known):
            raise SystemExit(f'--labels {list(explicit)!r} contradicts the recorded {known[0]!r}')
        return list(explicit)
    if len(known) != 2:
        raise SystemExit('a run has no recorded labels: pass --labels in checkpoint order, '
                         "as that daemon's GET /v1/models lists them")
    return list(known[0])


def resume_state(prev: dict, device: dict, name: str):
    """(scores, meta, log_read_at) of an interrupted collection. The read
    time is the FIRST read's; a run that predates recording keeps None,
    because "now" would admit live rows that run never scored."""
    if prev['device'] != device:
        raise SystemExit(f'collection was made on {prev["device"]!r}, not {device!r}')
    if prev['name'] != name:
        raise SystemExit(f'collection is named {prev["name"]!r}; resuming it as {name!r} would relabel it')
    return prev['scores'], prev['meta'], prev.get('log_read_at')


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


def check_names(run_a: dict, run_b: dict):
    if run_a['name'] == run_b['name']:
        raise SystemExit(f'both runs are named {run_a["name"]!r}: their gate files would collide')


def shared_keys(a: dict, b: dict, intersect: bool):
    """(keys to compare, {'shared', 'only_a', 'only_b'}). Different key sets
    are refused unless --intersect: a truncated run must not quietly shrink
    the measured set, and a sampled control must say so."""
    ka, kb = set(a), set(b)
    counts = {'shared': len(ka & kb), 'only_a': len(ka - kb), 'only_b': len(kb - ka)}
    if ka != kb and not intersect:
        raise SystemExit(f'runs scored different clause sets {counts!r}: pass --intersect to '
                         'compare only the shared clauses')
    return ka & kb, counts


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


def get_json(url, timeout=60):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


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
    log_text = Path(args.daemon_log).read_text()
    device = parse_startup(log_text)
    check_url_matches_log(args.url, log_text)
    if args.expect_device and device['device_type'] != args.expect_device:
        raise SystemExit(f'daemon reports device_type={device["device_type"]!r}, '
                         f'expected {args.expect_device!r}')
    models = get_json(f'{args.url}/v1/models')
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
        done, meta, log_read_at = resume_state(json.loads(out_path.read_text()), device, args.name)
    todo = [k for k in keys if k not in done]
    print(f'{args.name}: {len(keys)} clauses, {len(done)} already scored, {len(todo)} to go '
          f'({device["device_type"]}/{device["backend"]} threads={device["configured_threads"]})',
          file=sys.stderr)

    def checkpoint():
        # a restart onto another device between checkpoints must not be
        # published under the first startup line's name
        if parse_startup(Path(args.daemon_log).read_text()) != device:
            raise SystemExit('daemon log changed device mid-run')
        write_private(out_path, {'name': args.name, 'device': device, 'meta': meta,
                                 'sample': {'n': args.sample, 'seed': args.seed} if args.sample else None,
                                 'log_read_at': log_read_at, 'scores': done})

    t0 = time.monotonic()
    for i in range(0, len(todo), args.batch):
        chunk = todo[i:i + args.batch]
        res = classify(args.url, [texts[k] for k in chunk])
        mid, wh = batch_identity(res)
        if meta is None:
            meta = {'model_id': mid, 'weight_hash': wh}
        elif (meta['model_id'], meta['weight_hash']) != (mid, wh):
            raise SystemExit(f'head changed mid-run: {meta!r} -> {(mid, wh)!r}')
        labels = classifier_labels(models, mid, wh)
        if meta.setdefault('labels', labels) != labels:
            raise SystemExit(f'labels changed mid-run: {meta["labels"]!r} -> {labels!r}')
        check_score_labels(res, labels)
        for k, r in zip(chunk, res):
            done[k] = r['scores']
        n = i + len(chunk)
        if n % (args.batch * 20) == 0 or n == len(todo):
            checkpoint()
            rate = (time.monotonic() - t0) / n * 1000
            print(f'  {n}/{len(todo)}  {rate:.1f}ms/clause  eta {rate * (len(todo) - n) / 60000:.0f}min',
                  file=sys.stderr)
    checkpoint()
    return 0


def probe_run(probes, scores, meta, labels):
    return {'meta': {'model_id': meta['model_id'], 'weight_hash': meta['weight_hash']}, 'results': {
        p['id']: {'top': top_label(scores[text_key(p['cmd'])], labels),
                  'scores': scores[text_key(p['cmd'])]} for p in probes}}


def benign_run(probes, scores, meta, labels):
    return {'model_id': meta['model_id'], 'weight_hash': meta['weight_hash'], 'results': {
        p['id']: {'cmd': p['cmd'], 'scores': scores[text_key(p['cmd'])],
                  'top': top_label(scores[text_key(p['cmd'])], labels)} for p in probes}}


def cmd_compare(args):
    a = json.loads(Path(args.a).read_text())
    b = json.loads(Path(args.b).read_text())
    check_names(a, b)
    same_head(a['meta'], b['meta'])
    labels = resolve_labels(args.labels, a['meta'], b['meta'])
    shared, overlap = shared_keys(a['scores'], b['scores'], args.intersect)
    sa = {k: a['scores'][k] for k in shared}
    sb = {k: b['scores'][k] for k in shared}
    summary = {
        'a': {'name': a['name'], 'device': a['device']},
        'b': {'name': b['name'], 'device': b['device']},
        'head': {'model_id': a['meta']['model_id'], 'weight_hash': a['meta']['weight_hash']},
        'labels': labels,
        'keys': overlap,
        'clauses': compare_clauses(sa, sb, labels),
        'rows': {},
    }
    if not args.clauses_only:
        if not args.severe_label or not args.fired_label:
            raise SystemExit('row verdicts need --severe-label (the deployed --cascade-severe-label '
                             "flags, least severe first) and --fired-label (the hook's fired set)")
        severe, fired = args.severe_label, frozenset(args.fired_label)
        summary['severe_order'], summary['fired_labels'] = severe, sorted(fired)
        live_until = resolve_live_until(args.live_until, a, b)
        sets = row_sets(args.log, live_until)
        summary['live_until'] = live_until
        for name, rows in sets.items():
            summary['rows'][name] = row_agreement(rows, sa, sb, labels, severe, fired)
        gate_dir = Path(args.gate_dir)
        gate_rows = sets['gate_window']
        label = soak_label(fired)
        sev, ben = load_jsonl(SEVERITY_PROBES), load_jsonl(BENIGN_PROBES)
        for run in (a, b):
            tag = run['name']
            write_private(gate_dir / f'probes_run_{tag}.json', probe_run(sev, run['scores'], run['meta'], labels))
            write_private(gate_dir / f'benign_run_{tag}.json', benign_run(ben, run['scores'], run['meta'], labels))
            write_private(gate_dir / f'soak_{tag}.json',
                          soak_doc(gate_rows, run['scores'], label, run['meta']['model_id'],
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
    c.add_argument('--url', required=True, help='that daemon; its port must appear in --daemon-log')
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
    m.add_argument('--labels', type=lambda s: s.split(','),
                   help='comma-separated, checkpoint order as GET /v1/models lists them; '
                        'only for runs that predate recorded labels')
    m.add_argument('--intersect', action='store_true',
                   help='compare only the clauses both runs scored (a sampled control, or a run '
                        'that read a longer log); the dropped counts are recorded')
    m.add_argument('--severe-label', action='append',
                   help='repeatable, least severe first: the deployed --cascade-severe-label flags')
    m.add_argument('--fired-label', action='append', help="repeatable: the hook's fired set")
    m.add_argument('--clauses-only', action='store_true',
                   help='skip rows and gate files (a sampled control covers no whole rows)')
    m.add_argument('--save', help='write the aggregate summary (no text) here')

    args = ap.parse_args(argv)
    return cmd_collect(args) if args.cmd == 'collect' else cmd_compare(args)


if __name__ == '__main__':
    sys.exit(main())
