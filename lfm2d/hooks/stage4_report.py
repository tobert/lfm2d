#!/usr/bin/env python3
"""What stage 4 would do to the live advisory log.

    python3 lfm2d/hooks/stage4_report.py            # the whole log
    python3 lfm2d/hooks/stage4_report.py --since 24h
    LFM2D_KAISH_BIN=/path/to/kaish-0.17.0 python3 ... --json a.json

Reads `~/.cache/claude-hooks/lfm2d-advisory.jsonl`, takes every row whose
classifier verdict was RAISED (data-critical winner by default), and asks
stage4.review whether the plan alone dismisses it. Prints the desk load
before and after, the reason histogram, and every survivor.

This measures a filter, not a threshold. The question it answers is the
one from ~/exomemory/lfm2d/adjudicator-cascade.md: how many statements a
day actually reach the adjudicator, and are they the ones a desk should
see.

WHERE THE FACTS COME FROM
-------------------------
`plan.commands[]` in the log carries the verb and redirects per clause
but NOT the args (they are already verbatim in the clause text), and
stage 4 needs args to read a git subcommand. So this tool RE-PLANS each
row's raw command with the installed kaish and aligns the result to the
scored clause by its rendered text.

That alignment is itself a measurement: kaish's canonical rendering is
the scored text, so a plan-path row that no longer aligns is a row whose
rendering CHANGED under the installed kaish. Point LFM2D_KAISH_BIN at two
builds and the drift count is the renderer diff.

A row scored on the clause_split FALLBACK is different: it never had plan
facts, so stage 4 having nothing to say is the design. Those are counted
as `fallback_path` and kept out of the drift signal -- but they still
reach the desk, because an unjudged row is not a dismissal. Nothing is
silently dropped: understating the desk load is the one error that
matters here.

LOCAL ONLY. Survivor clauses are real command text from real sessions.
"""
import argparse
import collections
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from kaish_plan import kaish_version, plan_clauses  # noqa: E402
from stage4 import review  # noqa: E402

DEFAULT_LOG = Path('~/.cache/claude-hooks/lfm2d-advisory.jsonl').expanduser()


def parse_since(s):
    """'24h' / '5d' / '90m' -> seconds. Loud on anything else."""
    if s is None:
        return None
    units = {'m': 60, 'h': 3600, 'd': 86400}
    if len(s) < 2 or s[-1] not in units:
        raise SystemExit(f'--since wants a number followed by m/h/d, got {s!r}')
    return float(s[:-1]) * units[s[-1]]


def raised_rows(log, top, since):
    """Rows whose classifier verdict was raised to `top`, newest-bounded
    by `since`. A row without a successful classify never reached stage 3
    and so has no verdict for stage 4 to review."""
    cutoff = (time.time() - since) if since else None
    rows = []
    for line in open(log):
        try:
            r = json.loads(line)
        except Exception:
            continue  # a torn last line while the hook is writing
        if cutoff and r.get('ts', 0) < cutoff:
            continue
        d = r.get('lfm2d') or {}
        if d.get('ok') and d.get('top') == top:
            rows.append(r)
    return rows


# Why a row cannot be judged. The distinction matters: a row that was
# already on the clause_split fallback when it was SCORED never had plan
# facts to begin with, so stage 4 having nothing to say about it is the
# design, not a regression. A plan-path row that no longer plans or no
# longer renders the same IS a regression -- kaish's canonical rendering
# is the scored text. Collapsing these two into one "unreadable" count
# buries the second in the first (there are ~29 of the first and 0 of the
# second on the live window), which is how a canary gets muted.
EXPECTED_UNJUDGED = {'fallback_path'}


def winner_facts(row):
    """The planned facts for the clause the classifier ranked highest, or
    (None, why). Re-plans the raw command and aligns by rendered text."""
    d = row['lfm2d']
    want = d.get('winner_clause')
    if not want:
        return None, 'no_winner_clause'
    fallback = d.get('split_path') != 'kaish_plan'
    res = plan_clauses(row.get('command') or '')
    if not res.get('ok'):
        # A fallback row failing to plan is the same parser saying the same
        # thing it said at score time, not drift.
        return None, ('fallback_path' if fallback else f'replan_failed:{res["error"]}')
    for clause in res['clauses']:
        if clause['text'] == want:
            return clause, None
    return None, ('fallback_path' if fallback else 'unaligned')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--log', type=Path, default=DEFAULT_LOG)
    ap.add_argument('--top', default='data-critical',
                    help='the raised verdict stage 4 is asked to review')
    ap.add_argument('--since', help='window, e.g. 24h / 5d (default: whole log)')
    ap.add_argument('--json', type=Path, help='write the aggregates here too')
    args = ap.parse_args(argv)

    rows = raised_rows(args.log, args.top, parse_since(args.since))
    if not rows:
        print(f'no {args.top} rows in {args.log}'
              + (f' within {args.since}' if args.since else ''))
        return 0

    span = (min(r['ts'] for r in rows), max(r['ts'] for r in rows))
    days = max((span[1] - span[0]) / 86400, 1 / 24)

    dismissed, survived, unreadable, drifted = [], [], [], []
    reasons = collections.Counter()
    for row in rows:
        facts, why = winner_facts(row)
        if facts is None:
            unreadable.append((row, why))
            if why not in EXPECTED_UNJUDGED:
                drifted.append((row, why))
            reasons[why] += 1
            continue
        verdict = review(facts, guard_decision=(row.get('regex') or {}).get('decision'))
        reasons[verdict.reason] += 1
        (dismissed if verdict.dismissible else survived).append((row, verdict))

    n = len(rows)
    # An unjudged row is NOT a dismissal: it reaches the desk. Counting it
    # anywhere else would understate the load.
    desk = len(survived) + len(unreadable)
    judged = len(dismissed) + len(survived)
    print(f'{"=" * 68}\nstage 4 over {n} {args.top} rows  '
          f'({days:.1f} days, {kaish_version()})\n{"=" * 68}')
    print(f'  before stage 4: {n:4d} to the desk  ({n / days:5.1f}/day)')
    print(f'  no plan facts:  {len(unreadable):4d}'
          f'  (scored on the clause_split fallback; nothing to judge)')
    print(f'  judged:         {judged:4d}')
    print(f'    dismissed:    {len(dismissed):4d}'
          f'  ({100 * len(dismissed) / judged:.0f}% of judged)' if judged else '')
    print(f'    survived:     {len(survived):4d}')
    print(f'  after stage 4:  {desk:4d} to the desk  ({desk / days:5.1f}/day)')
    if drifted:
        print(f'\n  !! {len(drifted)} plan-path row(s) no longer plan or no longer '
              f'render the same under {kaish_version()}.\n'
              f'     kaish\'s rendering IS the scored text: this is drift, not noise.')
    else:
        print(f'\n  plan-path rows all re-plan and align under {kaish_version()}: '
              f'no renderer drift.')

    print('\nreasons')
    for reason, count in reasons.most_common():
        print(f'  {count:4d}  {reason}')

    if drifted:
        print('\ndrifted -- plan-path rows the installed kaish reads differently')
        for row, why in drifted[:20]:
            print(f'  {why:24} {row["lfm2d"].get("winner_clause")!r}')

    print(f'\nsurvivors -- what the adjudicator would see  (LOCAL ONLY, real text)')
    for row, verdict in survived:
        dc = row['lfm2d']['scores'].get('data-critical', 0)
        print(f'  {dc:.3f}  {verdict.reason:34} {row["lfm2d"]["winner_clause"]!r}')

    print(f'\ndismissed -- the classifier raised these and the plan clears them')
    for row, verdict in dismissed:
        dc = row['lfm2d']['scores'].get('data-critical', 0)
        print(f'  {dc:.3f}  {verdict.reason:34} {row["lfm2d"]["winner_clause"]!r}')

    if args.json:
        args.json.write_text(json.dumps({
            'kaish_version': kaish_version(),
            'log': str(args.log), 'top': args.top, 'since': args.since,
            'rows': n, 'days': round(days, 3),
            'dismissed': len(dismissed), 'survived': len(survived),
            'judged': judged, 'unreadable': len(unreadable),
            'drifted': len(drifted), 'desk': desk,
            'per_day_before': round(n / days, 2),
            'per_day_after': round(desk / days, 2),
            'reasons': dict(reasons),
        }, indent=1) + '\n')
        print(f'\nwrote {args.json}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
