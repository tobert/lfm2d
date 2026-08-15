# The delta-margin gate — why, and what it found

Amy, 2026-08-15, on the pass-2 saturation finding (baseline probes drifting
toward 0.99+ alongside the sysadmin probes compared against them, making
several gate PASSes numerically fragile rather than robust): **"probably
score deltas esp if we can expose something based on it."**

## Is anything already exposed?

Yes, and no wire change was needed. `/v1/classify` (`ClassifyResult.scores`,
`lfm2d/src/types.rs:147`) and `/v1/cascade` (`CascadeClause.severity_scores`
and `CascadeWinner.severity_scores`, `lfm2d/src/types.rs:199,219`) both
already carry the full per-label probability map for every clause, not just
the winner's aggregate `severity_score`. A client can already compute a
margin/delta between any two scored statements from the existing contract —
verified by reading `engine_real.rs:433-442` and `types.rs`, not assumed.
**No gap found; no wire/API proposal filed.** (Contrast with the still-open
`severity_score` → `expected_rank` question, which IS a wire contract change
and has correctly sat AWAITING AMY rather than being taken unilaterally —
this is not that kind of change.)

## `score_probes.py --delta-margin <n>`

New flag, default `0.0` (preserves the original pass/fail behavior
byte-for-byte — old saved runs score identically unless the flag is set).
With a positive value, a constraint must win by at least that margin to
count as a gate PASS; a constraint that holds the ordering but by less is
reported `FRAGILE` and counts as a gate failure in this mode. Margin is
normalized so it always reads "how robustly did the ordering hold" regardless
of the constraint's `>`/`>=`/`<` operator.

## What it found, reproducibly

```
python3 training/v9/severity_probes/score_probes.py --results <file> --delta-margin 0.05
```

| checkpoint | classic pass/fail | margin ≥ 0.05 |
|---|---|---|
| v8 | 8/23 | **5/23** |
| v9 pass 1 (`811b51d`) | 15/23 | **8/23** |
| v9 pass 2 (this session) | 17/23 | **8/23** |

**Pass 1 → pass 2 shows zero robust improvement under the delta metric**,
despite the classic gate showing +2 (15→17). Both land at 8/23 with a
margin ≥ 0.05. Nine of pass 2's seventeen nominal passes are FRAGILE —
margins as thin as +0.000 to +0.009 (`root_delete_over_source_file`,
`home_delete_over_source_file`, `etc_delete_over_source_file`,
`shadow_over_readme`, `shadow_over_source_file`, `ssh_key_over_source_file`,
`md_suffix_does_not_neuter_shadow`, `key_material_over_config`,
`no_preserve_root_not_lower`, `truncate_forms_agree`) — several are
literally three decimal places from a coin flip.

**Recommendation for pass 3 and beyond: report BOTH metrics, but treat the
delta-margin number as the honest one for "is this checkpoint better."**
The classic pass/fail count is what the ordinal-collapse mechanics this
project has already been bitten by once (see
`ordinal-collapsed-to-a-set-loses-order`) predict it would be: a metric that
looks like it's moving while the underlying separation isn't.
