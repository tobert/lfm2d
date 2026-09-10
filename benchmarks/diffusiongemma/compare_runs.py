#!/usr/bin/env python3
"""Compare benchmark runs case by case, and say what a difference is NOT.

    python3 benchmarks/diffusiongemma/compare_runs.py \
        --run base-off=/tmp/dg-base-off --run base-on=/tmp/dg-base-on

Reads each run's `summary.json` plus its metadata row, prints per-case
medians side by side, and prints the provenance that decides whether a
comparison means anything at all.

WHY IT REPORTS PASSES NEXT TO TIME
----------------------------------
Response time here is dominated by pass count, and pass count is a
property of the ANSWER, not of the kernel or the cache: a run that
happened to converge in fewer passes is faster for reasons that have
nothing to do with what changed. The device RNG is seeded once at model
startup and warmups advance it, so requests are not paired random draws
across runs (the harness README says so explicitly). Any claim about a
cache or a kernel therefore has to survive the pass counts being
comparable -- so passes are printed beside every time, and a large pass
delta is flagged as "not attributable" rather than quietly reported as a
speedup.

The column that DOES isolate a prefix cache is `PREFILL s`, taken from
the engine's own `total_prompt_time_sec`. `first_content_s` cannot do it:
content arrives only when a whole canvas commits, so it comes out
identical to `response_s` on every row measured so far.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

# A pass-count difference this large or larger between two runs means the
# runs answered differently, so their times are not comparable as a
# speedup. Judgement call, stated rather than hidden.
PASS_DELTA_LIMIT = 0.15


def load(path):
    path = Path(path)
    summary = json.loads((path / 'summary.json').read_text())
    rows = [json.loads(line) for line in (path / 'results.jsonl').read_text().splitlines()]
    meta = rows[0]
    # PREFILL, measured by the engine rather than estimated: every request's
    # usage record carries total_prompt_time_sec and prompt_tokens. This is the
    # column a prefix cache is supposed to move, and the only one that
    # separates prompt cost from denoising cost -- first_content_s cannot,
    # because content arrives only when a whole canvas commits, which makes it
    # identical to response_s on every row.
    measured = [r for r in rows if r.get('type') == 'request' and r.get('phase') == 'measure']
    for r in measured:
        usage = r.get('usage') or {}
        case = summary['cases'].setdefault(r['case_id'], {})
        case.setdefault('_prompt_s', []).append(usage.get('total_prompt_time_sec'))
        case.setdefault('_prompt_tokens', []).append(usage.get('prompt_tokens'))
    for case in summary['cases'].values():
        for key in ('_prompt_s', '_prompt_tokens'):
            vals = [v for v in case.get(key, []) if isinstance(v, (int, float))]
            case[key[1:]] = statistics.median(vals) if vals else None
    return summary, meta


def median_of(case, field, key='p50'):
    value = case.get(field)
    if isinstance(value, dict):
        return value.get(key)
    return value


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--run', action='append', required=True, metavar='LABEL=DIR',
                    help='repeatable; the first is the baseline')
    args = ap.parse_args(argv)

    runs = []
    for spec in args.run:
        if '=' not in spec:
            raise SystemExit(f'--run wants LABEL=DIR, got {spec!r}')
        label, _, path = spec.partition('=')
        try:
            runs.append((label, *load(path)))
        except FileNotFoundError as e:
            print(f'  {label}: MISSING ({e.filename})')
    if not runs:
        raise SystemExit('no runs loaded')

    print('provenance -- a comparison across a difference here is not a speedup')
    for label, summary, meta in runs:
        print(f'  {label:10} isq={meta["isq"]} thinking={meta["thinking"]} '
              f'prefix_cache_n={meta.get("prefix_cache_n")} '
              f'cases_sha={meta["cases_sha256"][:12]} '
              f'binary_sha={meta["binary_sha256"][:12]} requests={summary["requests"]}')

    fields = [('response_s', 'response s'), ('prompt_s', 'PREFILL s (measured)'),
              ('prompt_tokens', 'prompt tokens'),
              ('passes', 'passes'), ('completion_tokens', 'out tokens')]
    for field, title in fields:
        print(f'\n{title} (per-case median)')
        header = '  ' + f'{"case":24}' + ''.join(f'{label:>16}' for label, _, _ in runs)
        print(header)
        ids = sorted({cid for _, s, _ in runs for cid in s['cases']})
        for cid in ids:
            row = f'  {cid:24}'
            for _, summary, _ in runs:
                case = summary['cases'].get(cid)
                value = median_of(case, field) if case else None
                row += f'{value:>16.3f}' if isinstance(value, (int, float)) else f'{"-":>16}'
            print(row)
        # Aggregate across cases: median of the per-case medians.
        row = f'  {"MEDIAN OF CASES":24}'
        for _, summary, _ in runs:
            vals = [median_of(c, field) for c in summary['cases'].values()]
            vals = [v for v in vals if isinstance(v, (int, float))]
            row += f'{statistics.median(vals):>16.3f}' if vals else f'{"-":>16}'
        print(row)

    print('\naccuracy (expected-field checks, from the harness own checker)')
    for label, summary, _ in runs:
        passed = total = 0
        for case in summary['cases'].values():
            checks = case.get('checks') or {}
            for status, count in checks.items():
                total += count
                if status == 'pass':
                    passed += count
        print(f'  {label:10} {passed}/{total} pass'
              + (f'   ({100 * passed / total:.0f}%)' if total else ''))

    print('\nprefill rate -- tokens/s, and what a 2,400-token preamble would cost')
    for label, summary, _ in runs:
        pairs = [(c.get('prompt_tokens'), c.get('prompt_s')) for c in summary['cases'].values()]
        pairs = [(n, s) for n, s in pairs if isinstance(n, (int, float))
                 and isinstance(s, (int, float)) and s > 0]
        if not pairs:
            print(f'  {label:10} -')
            continue
        rate = statistics.median(n / s for n, s in pairs)
        print(f'  {label:10} {rate:7.1f} tok/s   =>  2400 tokens would cost '
              f'{2400 / rate:5.2f} s of prefill per call')

    # The attribution guard. Compare every run to the baseline.
    base_label, base_summary, _ = runs[0]
    print(f'\nattribution vs {base_label}')
    for label, summary, _ in runs[1:]:
        deltas = []
        for cid, case in summary['cases'].items():
            base = base_summary['cases'].get(cid)
            if not base:
                continue
            bp, cp = median_of(base, 'passes'), median_of(case, 'passes')
            if isinstance(bp, (int, float)) and isinstance(cp, (int, float)) and bp:
                deltas.append(abs(cp - bp) / bp)
        worst = max(deltas) if deltas else 0.0
        rt = [median_of(c, 'response_s') for c in summary['cases'].values()]
        bt = [median_of(base_summary['cases'][cid], 'response_s')
              for cid in summary['cases'] if cid in base_summary['cases']]
        rt = [v for v in rt if isinstance(v, (int, float))]
        bt = [v for v in bt if isinstance(v, (int, float))]
        speed = (statistics.median(bt) / statistics.median(rt)) if rt and bt else float('nan')
        verdict = ('NOT ATTRIBUTABLE (pass counts differ by '
                   f'{100 * worst:.0f}% on some case)' if worst >= PASS_DELTA_LIMIT
                   else f'pass counts within {100 * worst:.0f}%')
        print(f'  {label:10} {speed:.3f}x on median response time -- {verdict}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
