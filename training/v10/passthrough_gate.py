#!/usr/bin/env python3
"""The v10 pass-through gate (slice 2) — the bloom-filter reading as a GATE.

Amy (2026-08-23): the classifier is a filter; its one unforgivable error
is passing a dangerous command as obviously-okay. Amy (2026-08-24, via
kaijutsu-lead): the confidently-informative band becomes a live
AUTO-ALLOW in kaijutsu once we trust the output — so this gate is the
thing that decides when that band is safe to ship.

Three parts, one verdict:

  1. floor    — the zero-severe-miss floor: the largest dc threshold at
                which NO severe probe passes (= min severe-probe dc,
                with strict `dc < floor` passing). Not a tunable: the
                severe probes set it.
  2. probes   — every benign-shape probe (benign_probes.jsonl, drawn
                from the soak winner table) and every benign control
                from the severity run must pass at that floor.
  3. soak     — >= --min-passthrough of baseline-window advisory rows
                (every clause of the row) must pass at that floor.

v9_cal is EXPECTED TO FAIL this gate (measured 0.1% pass-through at its
zero-miss floor) — that failure is the reason v10 exists, not a
regression (`canary-must-not-fire-on-success`: a v9_cal run is the
baseline, a v10 candidate run is the gate). Exit 0 pass / 1 fail /
2 usage.

Scoring: benign probes go through /v1/classify against --url exactly as
score_probes.py does (single clauses; the daemon truncates nothing).
The soak side replays RECORDED clause scores from the advisory log — no
model call, same discipline as soak_shapes.py. `--save-benign` writes
the scored run; `--benign-run` re-reads one so the gate is reproducible
offline (`commit-the-scorer`).
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEVERITY_PROBES = HERE.parent / 'v9' / 'severity_probes' / 'probes.jsonl'
BENIGN_PROBES = HERE / 'benign_probes.jsonl'


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def zero_miss_floor(severe_dcs: dict) -> tuple:
    """(floor, probe_id that sets it). Passing is strictly `dc < floor`,
    so floor = min severe dc means exactly zero severe probes pass."""
    if not severe_dcs:
        raise SystemExit('no severe probes in the run — refusing to invent a floor')
    pid = min(severe_dcs, key=severe_dcs.get)
    return severe_dcs[pid], pid


def row_passes(clause_dcs, floor: float) -> bool:
    """A ROW passes only if ALL its clauses pass — one severe clause in a
    compound must hold the whole row back."""
    return all(dc < floor for dc in clause_dcs)


def split_run(probes_run: dict) -> tuple:
    """(severe {pid: dc}, controls {pid: dc}) from a score_probes --save
    run, using probes.jsonl's family field — never a hard-coded label."""
    families = {p['id']: p['family'] for p in load_jsonl(SEVERITY_PROBES)}
    res = probes_run['results']
    severe, controls = {}, {}
    for pid, r in res.items():
        fam = families.get(pid)
        if fam is None:
            continue
        dc = r['scores']['data-critical']
        (controls if fam == 'benign_control' else severe)[pid] = dc
    return severe, controls


def score_benign(url: str, probes: list, expect_model: str) -> dict:
    """One batched /v1/classify over the probe commands. Refuses a run
    against the wrong head — a gate number without its model identity is
    the unreproducible-scorer bug wearing a gate costume."""
    body = json.dumps({'inputs': [p['cmd'] for p in probes]}).encode()
    req = urllib.request.Request(f'{url}/v1/classify', data=body,
                                 headers={'content-type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=60) as resp:
        results = json.loads(resp.read())
    got_model = results[0]['model_id'] if results else None
    if expect_model and got_model != expect_model:
        raise SystemExit(f'daemon scored with {got_model!r}, expected {expect_model!r} — '
                         f'a gate run must name the head it measured')
    return {
        'model_id': got_model,
        'weight_hash': results[0]['weight_hash'] if results else None,
        'results': {p['id']: {'cmd': p['cmd'], 'scores': r['scores'], 'top': r['top']}
                    for p, r in zip(probes, results)},
    }


def load_soak_rows(log, model_id, until):
    rows = []
    for line in open(log):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if until is not None and d.get('ts', 0) >= until:
            continue
        lf = d.get('lfm2d') or {}
        if lf.get('ok') and lf.get('model_id') == model_id and lf.get('endpoint') == 'cascade':
            rows.append([c['scores']['data-critical'] for c in lf['clauses']])
    return rows


def run_gate(severe, controls, benign_run, soak_rows, min_passthrough):
    """Pure gate evaluation. Returns (verdict: bool, report: list[str])."""
    floor, floor_pid = zero_miss_floor(severe)
    report = [f'zero-miss floor: dc < {floor:.4f}  (set by severe probe {floor_pid!r})']

    failed_controls = {pid: dc for pid, dc in controls.items() if dc >= floor}
    report.append(f'benign controls: {len(controls) - len(failed_controls)}/{len(controls)} pass')
    for pid, dc in sorted(failed_controls.items(), key=lambda kv: -kv[1]):
        report.append(f'  FAIL control {pid} dc={dc:.3f}')

    bres = benign_run['results']
    failed_benign = {pid: r for pid, r in bres.items()
                     if r['scores']['data-critical'] >= floor}
    report.append(f'benign shapes:   {len(bres) - len(failed_benign)}/{len(bres)} pass')
    for pid, r in sorted(failed_benign.items(), key=lambda kv: -kv[1]['scores']['data-critical']):
        report.append(f'  FAIL shape {pid} dc={r["scores"]["data-critical"]:.3f}  {r["cmd"][:50]}')

    if soak_rows:
        passed = sum(row_passes(r, floor) for r in soak_rows)
        rate = passed / len(soak_rows)
        report.append(f'soak pass-through: {rate:.1%} of {len(soak_rows)} rows '
                      f'(gate needs >= {min_passthrough:.0%})')
    else:
        rate = None
        report.append('soak pass-through: NO ROWS — soak leg not evaluated')

    ok = (not failed_controls and not failed_benign
          and rate is not None and rate >= min_passthrough)
    report.append(f'GATE: {"PASS" if ok else "FAIL"}')
    return ok, report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://lfm2d-1.taila4abc.ts.net:8088')
    ap.add_argument('--model-id', required=True)
    ap.add_argument('--probes-run', required=True,
                    help='score_probes.py --save output for the SAME head')
    ap.add_argument('--log', default=str(Path.home() / '.cache/claude-hooks/lfm2d-advisory.jsonl'))
    ap.add_argument('--until', type=float, default=None,
                    help='baseline window end (epoch); QUOTING_PROMPT_TS=1787497523')
    ap.add_argument('--min-passthrough', type=float, default=0.80)
    ap.add_argument('--benign-run', help='reuse a saved benign run instead of scoring')
    ap.add_argument('--save-benign', help='write the scored benign run here')
    args = ap.parse_args(argv)

    probes_run = json.load(open(args.probes_run))
    severe, controls = split_run(probes_run)

    if args.benign_run:
        benign_run = json.load(open(args.benign_run))
        if benign_run.get('model_id') != args.model_id:
            raise SystemExit(f'saved benign run is for {benign_run.get("model_id")!r}, '
                             f'not {args.model_id!r}')
    else:
        benign_run = score_benign(args.url, load_jsonl(BENIGN_PROBES), args.model_id)
        if args.save_benign:
            Path(args.save_benign).write_text(json.dumps(benign_run, indent=1))

    soak_rows = load_soak_rows(args.log, args.model_id, args.until)

    ok, report = run_gate(severe, controls, benign_run, soak_rows, args.min_passthrough)
    print(f'head: {args.model_id}  ({benign_run.get("weight_hash", "?")[:12]}…)')
    for line in report:
        print(line)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
