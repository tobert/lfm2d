#!/usr/bin/env python3
"""Which probes set the floor, and which are unstable across candidates.

kaijutsu-lead (2026-08-25): because the pass-through floor is min dc
over the data-critical-truth probes, every training run is coupled to
the auto-allow band -- coverage work aimed at npm moved
`find / -name -delete` 0.66 -> 0.40 and dragged the floor under two
benign shapes. The coupling is the safety property (a dc probe under
the floor IS a shape the band would auto-allow), so the floor stays
one number; what this report decouples is the DIAGNOSIS: for every
dc-truth probe, its dc across every saved candidate run, its spread,
and how often it set the floor. A probe with a wide spread is an
untaught form (floor-set-by-an-untaught-form-is-noise) and the next
slice should teach it BEFORE the next re-gate, not after.

    python3 training/v10/floor_report.py            # all probes_run_*.json in training/v10
"""
import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from passthrough_gate import split_run, zero_miss_floor  # noqa: E402


def collect(paths):
    """Pure. {probe: {run: dc}} for dc-truth probes, plus {run: floor probe}."""
    per_probe, floors = {}, {}
    for p in paths:
        run = Path(p).stem.replace('probes_run_', '')
        severe, _ = split_run(json.loads(Path(p).read_text()))
        if not severe:
            continue
        floor, pid = zero_miss_floor(severe)
        floors[run] = (pid, floor)
        for pid_, dc in severe.items():
            per_probe.setdefault(pid_, {})[run] = dc
    return per_probe, floors


def main(argv=None):
    paths = sorted(glob.glob(str(HERE / 'probes_run_*.json')))
    per_probe, floors = collect(paths)
    set_count = {}
    for pid, _ in floors.values():
        set_count[pid] = set_count.get(pid, 0) + 1
    rows = []
    for pid, runs in per_probe.items():
        vals = list(runs.values())
        rows.append((max(vals) - min(vals), pid, min(vals), max(vals), set_count.get(pid, 0), len(vals)))
    rows.sort(reverse=True)
    print(f'{len(paths)} runs; floor set by: ' + ', '.join(f'{k}x{v}' for k, v in sorted(set_count.items(), key=lambda kv: -kv[1])))
    print(f'{"probe":10} {"spread":>7} {"min":>6} {"max":>6}  set-floor  runs')
    for spread, pid, lo, hi, n_set, n in rows:
        print(f'{pid:10} {spread:7.3f} {lo:6.3f} {hi:6.3f}  {n_set:9d}  {n}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
