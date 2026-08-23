#!/usr/bin/env python3
"""What v9_cal's soak firings are made of, and what would move them.

Written 2026-08-23 for the v10 plan. AGGREGATES ONLY -- the advisory log is
0600 real command text; this prints argv0/flag shapes and counts, never rows.

Three tables, all from the log's RECORDED per-clause probabilities (no model
needed; the pod's cascade is replayed exactly by `aggregate_row`'s rule):

  1. shapes    -- argv0 share of all clauses vs dc-argmax clauses vs firing
                  winners, plus the counterfactual where an upstream parser
                  removes interpreter-payload clauses (heredoc / `-c` / pipe
                  into interpreter) before the classifier sees them.
  2. tau       -- re-apply the prior correction at other exponents to BOTH the
                  live firings and the 68-probe gate. Needs a live probe run
                  (score_probes.py --save) taken from the SAME head as the log.
  3. coverage  -- how many distinct (argv0, flags, redirect) shapes cover the
                  clause population; sizes a labeled-by-shape sample.
  4. passthrough -- the bloom-filter reading (Amy, 2026-08-23): if the
                  classifier's job is to let the obviously-okay through and
                  hand everything else up a chain, what fraction of clauses /
                  rows pass at each dc floor, and which severe probes would
                  pass with them (the one failure a filter must not have).

    .venv-train/bin/python training/v10/soak_shapes.py --model-id kube_ordinal_v9_cal \
        --probes-run /path/to/probes_v9cal_live.json --base-tau 0.5
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'v9' / 'severity_probes'))
from shape_impact import HEREDOC_OPEN, DASH_C, PIPE_INTO  # noqa: E402

DEFAULT_LOG = Path('~/.cache/claude-hooks/lfm2d-advisory.jsonl').expanduser()
# Checkpoint label order is irrelevant here: scores are keyed by name.
LABELS = ['data-critical', 'informative', 'situation-normal']
SEVERE_ORDER = ['situation-normal', 'data-critical']   # deploy/k8s-zorak.yaml
WEIGHTS = {l: 0.0 for l in LABELS}
for rank, l in enumerate(SEVERE_ORDER, start=1):
    WEIGHTS[l] = float(rank)
SKIP_PREFIX = ('sudo', 'time', 'nohup', 'exec', 'command', 'builtin')
KEYWORDS = {'if', 'while', 'for', 'then', 'do', 'else', 'elif', 'fi', 'done',
            '{', '(', '!', '[', '[['}


def class_prior(train_jsonl):
    c = collections.Counter(json.loads(l)['label'] for l in open(train_jsonl) if l.strip())
    n = sum(c.values())
    missing = [l for l in LABELS if l not in c]
    if missing:
        raise SystemExit(f'{train_jsonl}: no rows for {missing}; refusing to invent a prior')
    return {l: c[l] / n for l in LABELS}


def shift(scores, prior, extra_tau):
    """p' ∝ p / prior^tau -- the same per-class logit constant calibrate_prior.py folds in."""
    q = {l: scores[l] / (prior[l] ** extra_tau) for l in LABELS}
    z = sum(q.values())
    return {l: v / z for l, v in q.items()}


def severity(scores):
    return sum(WEIGHTS[l] * scores[l] for l in LABELS)


def winner(scores_list):
    """Max severity, ties to the EARLIER clause (stable-descending, tests/cascade.rs)."""
    w = 0
    for i, s in enumerate(scores_list[1:], start=1):
        if severity(s) > severity(scores_list[w]):
            w = i
    return w


def top(scores):
    return max(LABELS, key=lambda l: scores[l])


def argv0(clause):
    toks = clause.strip().split()
    i = 0
    while i < len(toks) and (re.match(r'^[A-Za-z_]\w*=', toks[i]) or toks[i] in SKIP_PREFIX):
        i += 1
    if not toks:
        return '<empty>'
    if i >= len(toks):
        return 'assign'
    t = toks[i]
    if t in KEYWORDS:
        return 'kw:' + t
    a = t.rsplit('/', 1)[-1]
    if a == 'sed':
        return 'sed -i' if re.search(r'\s(-i\b|--in-place)', clause) else 'sed -n/other'
    if a == 'echo' and '===' in clause:
        return 'echo ==='
    return a


def shape(clause):
    """argv0 + subcommand (for the verbs that have one) + short flags + redirect kind."""
    t = clause.strip()
    toks = t.split()
    if not toks:
        return '<empty>'
    i = 0
    while i < len(toks) and (re.match(r'^[A-Za-z_]\w*=', toks[i]) or toks[i] in SKIP_PREFIX):
        i += 1
    if i >= len(toks):
        return 'assign'
    a = toks[i].rsplit('/', 1)[-1]
    flags = tuple(sorted({x for x in toks[i + 1:i + 6] if x.startswith('-') and len(x) < 6}))
    sub = ''
    if a in ('git', 'cargo', 'kubectl', 'gh', 'docker', 'systemctl', 'npm', 'pip', 'uv', 'kaish') \
            and i + 1 < len(toks) and not toks[i + 1].startswith('-'):
        sub = toks[i + 1]
    if '>>' in t:
        redir = '>>'
    elif re.search(r'(^|[^>2&])>(?!>)\s*\S', t):
        redir = '>'
    else:
        redir = ''
    return ' '.join(x for x in (a, sub, ' '.join(flags), redir) if x)


def is_payload(clause):
    return bool(HEREDOC_OPEN.search(clause) or DASH_C.search(clause) or PIPE_INTO.search(clause))


def load_rows(log, model_id):
    rows = []
    for line in open(log):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        lf = d.get('lfm2d') or {}
        if lf.get('ok') and lf.get('model_id') == model_id and lf.get('endpoint') == 'cascade':
            rows.append(d)
    if not rows:
        raise SystemExit(f'no cascade rows for model_id={model_id!r} in {log}')
    return rows


def table_shapes(rows):
    n = len(rows)
    all_c, dc_c, win_c, cf_win_c = (collections.Counter() for _ in range(4))
    clause_total = payload_clauses = fired = fired_payload_winner = fired_any_payload = 0
    cf_fired = cf_new = all_payload_rows = 0
    win_dc = []
    for d in rows:
        cl = d['lfm2d']['clauses']
        scores = [c['scores'] for c in cl]
        w = winner(scores)
        tops = [top(s) for s in scores]
        pay = [is_payload(c['clause']) for c in cl]
        payload_clauses += sum(pay)
        for c, t in zip(cl, tops):
            a = argv0(c['clause'])
            all_c[a] += 1
            clause_total += 1
            if t == 'data-critical':
                dc_c[a] += 1
        f = tops[w] == 'data-critical'
        if f:
            fired += 1
            win_c[argv0(cl[w]['clause'])] += 1
            win_dc.append(scores[w]['data-critical'])
            fired_payload_winner += pay[w]
            fired_any_payload += any(pay)
        keep = [i for i in range(len(cl)) if not pay[i]]
        if not keep:
            all_payload_rows += 1
            continue
        kw = keep[winner([scores[i] for i in keep])]
        if tops[kw] == 'data-critical':
            cf_fired += 1
            cf_win_c[argv0(cl[kw]['clause'])] += 1
            cf_new += not f
    print(f'cascade rows {n}, clauses {clause_total}')
    print(f'clause-level dc argmax {sum(dc_c.values())}/{clause_total} = {sum(dc_c.values())/clause_total:.1%}')
    print(f'row firings {fired}/{n} = {fired/n:.1%}')
    print(f'  winner is a payload clause: {fired_payload_winner} ({fired_payload_winner/fired:.1%}); '
          f'row has any payload clause: {fired_any_payload} ({fired_any_payload/fired:.1%})')
    print(f'payload clauses {payload_clauses}/{clause_total} = {payload_clauses/clause_total:.1%}; '
          f'rows that are ALL payload: {all_payload_rows}')
    cf_n = n - all_payload_rows
    print(f'COUNTERFACTUAL upstream removes payload clauses: firings {cf_fired}/{cf_n} = {cf_fired/cf_n:.1%} '
          f'(rows newly fired because a payload clause had masked a lower one: {cf_new})')
    wd = sorted(win_dc)
    q = lambda f: wd[min(len(wd) - 1, int(len(wd) * f))]
    print(f'winner dc prob p10/p50/p90 {q(.1):.3f}/{q(.5):.3f}/{q(.9):.3f}; '
          f'>=0.5: {sum(x >= .5 for x in wd)}, >=0.8: {sum(x >= .8 for x in wd)}')
    print()
    print(f'{"argv0":14s} {"clauses":>8s} {"share":>6s} {"dc-argmax":>10s} {"of own":>7s} {"winner":>7s} {"cf-win":>7s}')
    keys = sorted(set(a for a, _ in all_c.most_common(30)) | set(a for a, _ in win_c.most_common(20)),
                  key=lambda a: -all_c[a])
    for a in keys:
        print(f'{a:14s} {all_c[a]:8d} {all_c[a]/clause_total:6.1%} {dc_c[a]:10d} '
              f'{dc_c[a]/all_c[a]:7.1%} {win_c[a]:7d} {cf_win_c[a]:7d}')


def table_tau(rows, probes_run, base_tau, prior, taus):
    run = json.load(open(probes_run))
    model = run['meta'].get('model_id')
    live_model = rows[0]['lfm2d']['model_id']
    if model != live_model:
        raise SystemExit(f'probe run is from {model!r}, log rows are {live_model!r}: '
                         'a tau sweep across two heads describes neither')
    probes = {}
    for line in (HERE.parent / 'v9' / 'severity_probes' / 'probes.jsonl').read_text().splitlines():
        if line.strip():
            p = json.loads(line)
            probes[p['id']] = p
    res = run['results']
    severe_ids = [pid for pid, p in probes.items() if p['family'] != 'benign_control']
    benign_ids = [pid for pid, p in probes.items() if p['family'] == 'benign_control']
    cols = ['rm', 'sed -i', 'sed -n/other', 'echo ===', 'echo', 'python3', 'grep', 'awk', 'git']
    print(f'{"tau":>5} {"severe":>7} {"benign":>7} | {"fire%":>6} ' + ' '.join(f'{c:>12s}' for c in cols))
    base_hits = {pid for pid in severe_ids if top(res[pid]['scores']) == 'data-critical'}
    for tau in taus:
        extra = tau - base_tau
        sev = sum(top(shift(res[pid]['scores'], prior, extra)) == 'data-critical' for pid in severe_ids)
        ben = sum(top(shift(res[pid]['scores'], prior, extra)) == 'data-critical' for pid in benign_ids)
        fired = 0
        win_c = collections.Counter()
        for d in rows:
            cl = d['lfm2d']['clauses']
            ss = [shift(c['scores'], prior, extra) for c in cl]
            w = winner(ss)
            if top(ss[w]) == 'data-critical':
                fired += 1
                win_c[argv0(cl[w]['clause'])] += 1
        lost = [pid for pid in base_hits if top(shift(res[pid]['scores'], prior, extra)) != 'data-critical']
        print(f'{tau:5.2f} {sev:4d}/{len(severe_ids)} {ben:4d}/{len(benign_ids)}  | {fired/len(rows):6.1%} '
              + ' '.join(f'{win_c[c]:12d}' for c in cols)
              + (f'   lost vs base: {sorted(lost)}' if lost else ''))
    missed = [(pid, round(res[pid]['scores']['data-critical'], 3)) for pid in severe_ids if pid not in base_hits]
    print(f'severe probes already missed at base tau {base_tau}: {missed}')


def table_passthrough(rows, probes_run):
    run = json.load(open(probes_run))
    res = run['results']
    probes = {}
    for line in (HERE.parent / 'v9' / 'severity_probes' / 'probes.jsonl').read_text().splitlines():
        if line.strip():
            p = json.loads(line)
            probes[p['id']] = p
    sev = {pid: res[pid]['scores']['data-critical'] for pid, p in probes.items()
           if p['family'] != 'benign_control'}
    ben = sorted(round(res[pid]['scores']['data-critical'], 2) for pid, p in probes.items()
                 if p['family'] == 'benign_control')
    dcs = [c['scores']['data-critical'] for d in rows for c in d['lfm2d']['clauses']]
    print(f'clauses {len(dcs)}; a ROW passes only if ALL its clauses pass')
    print(f'{"pass if dc<":>12} {"clauses":>8} {"rows":>7}  severe probes that would PASS (= missed)')
    for t in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40):
        cp = sum(x < t for x in dcs) / len(dcs)
        rp = sum(all(c['scores']['data-critical'] < t for c in d['lfm2d']['clauses']) for d in rows) / len(rows)
        miss = sorted((v, pid) for pid, v in sev.items() if v < t)
        worst = ', '.join(f'{probes[pid]["cmd"][:30]}@{v:.2f}' for v, pid in miss[-4:])
        print(f'{t:12.2f} {cp:8.1%} {rp:7.1%}  {len(miss):2d}' + (f'  worst: {worst}' if miss else ''))
    print(f'benign controls dc: {ben}')
    print(f'severe probes dc, lowest 12: {sorted(round(v, 2) for v in sev.values())[:12]}')


def table_coverage(rows):
    all_c = collections.Counter()
    dc_c = collections.Counter()
    for d in rows:
        for c in d['lfm2d']['clauses']:
            s = shape(c['clause'])
            all_c[s] += 1
            if top(c['scores']) == 'data-critical':
                dc_c[s] += 1
    tot = sum(all_c.values())
    print(f'distinct shapes {len(all_c)} over {tot} clauses; singletons {sum(v == 1 for v in all_c.values())}')
    cum = 0
    for k, (_, n) in enumerate(all_c.most_common(), 1):
        cum += n
        if k in (10, 25, 50, 100, 200, 400, 800):
            print(f'  top {k:4d} shapes cover {cum/tot:.1%}')
    print('shapes by dc-argmax count:')
    for s, n in dc_c.most_common(25):
        print(f'  {n:4d}/{all_c[s]:5d} ({n/all_c[s]:4.0%})  {s}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', type=Path, default=DEFAULT_LOG)
    ap.add_argument('--model-id', required=True)
    ap.add_argument('--probes-run', type=Path, help='score_probes.py --save output from the same head')
    ap.add_argument('--base-tau', type=float, default=0.5, help='tau already folded into the head')
    ap.add_argument('--train', type=Path, default=HERE.parent / 'v9' / 'train.jsonl',
                    help='split the head was trained on (prior source, as calibrate_prior.py)')
    ap.add_argument('--taus', default='0.5,0.75,1.0,1.25,1.5')
    args = ap.parse_args()
    rows = load_rows(args.log, args.model_id)
    print(f'== shapes ({args.model_id})')
    table_shapes(rows)
    print()
    print('== coverage')
    table_coverage(rows)
    if args.probes_run:
        print()
        print(f'== tau sweep (base tau {args.base_tau}, prior from {args.train})')
        table_tau(rows, args.probes_run, args.base_tau, class_prior(args.train),
                  [float(t) for t in args.taus.split(',')])
        print()
        print('== passthrough (bloom-filter reading)')
        table_passthrough(rows, args.probes_run)


if __name__ == '__main__':
    main()
