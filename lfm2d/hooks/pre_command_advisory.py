#!/usr/bin/env python3
"""PreToolUse Bash hook — the regex guard, plus an ADVISORY lfm2d second opinion.

WHAT THIS DOES AND DOES NOT DO
------------------------------
The regex logic below decides EVERY outcome, byte-for-byte identically to
the baseline guard this was cloned from (test_parity.py gates 33 cases).
lfm2d's verdict is recorded beside the decision and never enforced. The
product of this phase is the log: where the model and the regex disagree,
and which of them is right when they do.

LFM2D_HOOK_MODE picks `advisory` (the default), `off` (no call at all), or
`enforce`, which is deliberately unimplemented — it prints a refusal and
falls back to advisory. Wiring an untested enforcement path and leaving it
reachable is how a "temporary" mode ships.

WHAT GETS SCORED
----------------
Plan-first. `kaish_plan.py` runs `kaish --plan-file -` and renders each
simple command in the resulting plan as one clause: canonically quoted
argv, redirect operators and targets explicit, heredoc bodies stripped out
of argv and tagged by kind, pipelines split into their members. That is
the unit the deployed checkpoint is trained on.

`clause_split.py` is the recorded FALLBACK for input kaish rejects. It
cuts on top-level `&&`, `||`, `;`, `&` and newlines, and never inside
quotes, substitutions or heredoc bodies — cutting there would fabricate a
bare severe command out of data. It also does not split pipelines, which
the plan path does, so the two paths produce different clause populations;
`split_path` on every scored row is what keeps them separable.

Neither path can be reached by exception. A raise anywhere in extraction
would crash the hook before emit() and silently disable the regex guard,
so every call site catches and records the failure by name instead.

Routing is by clause count: one clause goes to /v1/classify; two through
CASCADE_MAX_CLAUSES go to /v1/cascade, which ranks clauses WITHIN the
statement instead of diluting the severe one across a long command; past
that the row falls back to one batched /v1/classify, which keeps
per-clause truth in the log without inventing a winner client-side. Plan
clauses are deduped first — identical text scores identically, so
duplicates are pure daemon work — then bounded by PLAN_MAX_CLAUSES, and
both reductions are recorded on the row.

WHAT GETS LOGGED
----------------
One JSONL row per Bash call: the command, the regex verdict, the lfm2d
verdict, and a precomputed `disagree` bucket so analysis is a grep rather
than a join. The buckets are agree_flag, agree_clear, regex_only,
lfm2d_only, no_verdict, circuit_open and vocab_mismatch. `regex_only` is
the false-positive pile; `lfm2d_only` is the recall pile.

The label vocabulary is checked per call, never assumed. SEVERE_LABELS is
what the buckets key on, and when the deployed checkpoint speaks none of
them the row is bucketed `vocab_mismatch` rather than counted as "the
model saw nothing" — the vocabulary belongs to the checkpoint and has
changed wholesale before. No extra round trip is needed to notice: every
/v1/classify response already carries the full label set in `scores`.

KNOWN GAPS
----------
- The renderer is not pinned. LFM2D_KAISH_BIN defaults to whatever `kaish`
  is on PATH, so a kaish upgrade changes how clauses render mid-stream.
  That has happened, and it changed the argv rendering of a whole flag
  family. A floor measured across that boundary is not comparable with one
  measured after it; pin the binary or re-baseline deliberately.
- No OpenTelemetry. The daemon is fully instrumented; this hook writes only
  the local JSONL, so the disagreement data is readable only by opening a
  file on the machine that produced it.
- Command text leaves the machine and is written to disk — to the daemon
  over whatever LFM2D_URL points at, and to a 0600 local log. Commands can
  contain secrets. Weigh that before installing this everywhere.
- The circuit breaker leaves holes. Skipped calls are logged as
  `circuit_open` rows rather than omitted, so a quiet stretch is
  distinguishable from a stretch where nothing was asked — but they are
  still gaps when the log is mined as training signal.
- The classifier is wrong in both directions on real shell text, and the
  miss families are stable enough to name: bare build / test / run-script
  forms and plain `git push` scoring as severe; shell-structure fragments
  the fallback path hands it (`run() {`, `set -euo pipefail`); ordinary
  English in an `echo` argument it has no anchor for; and verbs belonging
  to programs the training corpus has never seen. Treat a firing as a
  ranking signal, not as ground truth.
- Data position is out of scope for this head by ruling: a severe command
  quoted inside a benign carrier's argument reads as severe. Secret
  detection belongs to the token-classification head and the routing
  suite. The practical consequence to work around is that the REGEX guard
  blocks on prose — write such a payload to a file and pass it by path
  rather than reformulating the command.
- The BLOCKED (git hygiene) rules are POLICY, not severity. Requiring
  explicit paths instead of a stage-everything flag is a convention, and a
  severity vocabulary cannot express it. The classifier is a candidate
  replacement for the rm/find SOFT_BLOCKED and WARNED half only; do not
  expect it to own the git half.
- Fail-open on any lfm2d error is correct here and invisible by design,
  because the regex was always going to decide. In an enforce mode the
  same behavior would be a silent downgrade, which is the house rule
  against silent fallbacks — hence `advisory_error` is always logged, and
  an enforce mode must decide this explicitly.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

# The splitter lives beside this file; the hook is invoked by absolute path
# from settings.json, so make the import location explicit rather than
# trusting whatever sys.path the caller left us.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from clause_split import split_clauses
    _SPLIT_IMPORT_ERROR = None
except Exception as _e:  # the splitter must never be able to break the guard
    _SPLIT_IMPORT_ERROR = f'{type(_e).__name__}: {_e}'

    def split_clauses(cmd: str) -> list:
        stripped = cmd.strip()
        return [stripped] if stripped else []

# Plan-first clause extraction (v10 slice 1, 2026-08-24): `kaish --plan`
# renders each simple command canonically — heredoc bodies stripped,
# quoting normalized, redirect targets explicit, pipelines split into
# members (deliberate reversal of clause_split's no-pipeline rule; see
# kaish_plan.py's docstring). The regex splitter becomes the RECORDED
# fallback for bash kaish rejects, so the advisory log carries a live
# fallback rate (`split_path` on every scored row). Decisions are still
# the regex's alone; this changes what gets SCORED, not what gets decided.
try:
    from kaish_plan import plan_clauses
    _PLAN_IMPORT_ERROR = None
except Exception as _e:  # the plan path must never break the guard either
    _PLAN_IMPORT_ERROR = f'{type(_e).__name__}: {_e}'

    def plan_clauses(cmd: str) -> dict:
        return {'ok': False, 'error': 'plan_import_error', 'detail': _PLAN_IMPORT_ERROR}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# lfm2d advisory configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Loopback by default: lfm2d is a sidecar, and the only endpoint a hook may
# assume without being told is the machine it runs on. The default used to
# be one specific host on one tailnet, which on any OTHER machine either
# sent every command off-box silently (advisory fails open, the breaker
# hides an unreachable default) or timed out on every call. A remote
# daemon is that machine's settings — `LFM2D_URL=... python3 <hook>` in
# the hook's command string (install.sh writes it that way) — not the
# code's. Tested in test_hook_config.py.
LFM2D_URL = os.environ.get('LFM2D_URL', 'http://127.0.0.1:8088')
# Timeout scales with input length, because inference latency does.
#
# The first version was a flat 400 ms, picked from a 75 ms measurement of a
# 29-character command. Real traffic killed it: 4 of the first 14 live rows
# timed out, and every one was a long command. Re-measured against the pod,
# the curve is linear and steep — ~65 ms fixed + ~0.37 ms per character:
#
#     46 chars   85 ms      1840 chars   604 ms
#    184 chars  113 ms      3680 chars  1293 ms
#    920 chars  318 ms      5520 chars  2112 ms
#
# A flat budget therefore doesn't just lose calls, it loses them
# SELECTIVELY: the dropped ones are the long commands — heredocs, quoted
# payloads, piped scripts — which are exactly where data-position content
# lives and exactly the phenomenon this advisory phase exists to measure.
# A sample biased against its own subject is worse than a smaller one.
#
# So: base + per-char, with ~3x margin over measured, capped. Short commands
# still fail fast (the common case, and what makes an outage detectable),
# long ones get the time they actually need. Affordable now only because the
# circuit breaker below bounds what a dead service can cost overall.
# Setting LFM2D_HOOK_TIMEOUT overrides the whole formula with a flat value.
TIMEOUT_BASE_S = float(os.environ.get('LFM2D_TIMEOUT_BASE', '0.35'))
TIMEOUT_PER_CHAR_S = float(os.environ.get('LFM2D_TIMEOUT_PER_CHAR', '0.0011'))
# Cap raised 5.0 -> 8.0 with clause splitting (2026-08-12): a 60-clause
# batch classify measured >5 s on the live pod under load, and paying the
# full cap to then LOSE the row is the worst of both worlds — the lost
# rows would be exactly the most-diluted monster commands this instrument
# exists to measure. Only the >CASCADE_MAX_CLAUSES batch path ever
# approaches the cap; a median 3-clause command budgets ~1.9 s.
TIMEOUT_CAP_S = float(os.environ.get('LFM2D_TIMEOUT_CAP', '8.0'))


LFM2D_TIMEOUT_S = float(os.environ.get('LFM2D_HOOK_TIMEOUT', '0')) or None

# /v1/cascade runs one classifier + one router forward PER CLAUSE (measured
# live 2026-08-12: ~145 ms/clause warm, ~290 cold, vs ~65 ms/clause for
# batch /v1/classify — the daemon does not batch cascade forwards yet), so
# a cascade call's budget needs a per-clause term the way classify's needs
# a per-char one. ~3x margin over warm. The shared cap still applies: a
# 64-clause monster measured 12.4 s live, and no Bash call should stall
# that long for an advisory opinion — past the cap it times out and the
# log says so (endpoint + error visible), which beats both stalling and
# silent truncation.
TIMEOUT_PER_CLAUSE_S = float(os.environ.get('LFM2D_TIMEOUT_PER_CLAUSE', '0.45'))
# Batch /v1/classify amortizes to ~65 ms/clause measured; 3x margin.
TIMEOUT_PER_BATCH_ITEM_S = float(os.environ.get('LFM2D_TIMEOUT_PER_BATCH_ITEM', '0.2'))

# Past this many clauses, cascade at deployed per-clause cost cannot finish
# under the cap (32 clauses measured 8.4 s), and losing those rows to
# timeouts would be SELECTIVE loss of the most-diluted commands — the very
# population this instrument exists to measure. So monsters fall back to
# one batched /v1/classify over the clauses (~65 ms/clause), keeping
# per-clause truth in the log without the daemon-owned ranking. Raise this
# once the daemon batches cascade forwards.
CASCADE_MAX_CLAUSES = int(os.environ.get('LFM2D_CASCADE_MAX_CLAUSES', '20'))

# Upper bound on distinct plan clauses sent to the daemon per row — the
# plan path's analogue of clause_split's MAX_CLAUSES=64, applied AFTER
# dedupe. 64 distinct clauses take the classify_batch path (~65 ms/item
# amortized) and fit the 8 s cap with margin.
PLAN_MAX_CLAUSES = int(os.environ.get('LFM2D_PLAN_MAX_CLAUSES', '64'))


def timeout_for(cmd: str) -> float:
    if LFM2D_TIMEOUT_S:
        return LFM2D_TIMEOUT_S
    return min(TIMEOUT_CAP_S, TIMEOUT_BASE_S + len(cmd) * TIMEOUT_PER_CHAR_S)


def cascade_timeout_for(clauses: list) -> float:
    if LFM2D_TIMEOUT_S:
        return LFM2D_TIMEOUT_S
    chars = sum(len(c) for c in clauses)
    return min(
        TIMEOUT_CAP_S,
        TIMEOUT_BASE_S + TIMEOUT_PER_CLAUSE_S * len(clauses) + chars * TIMEOUT_PER_CHAR_S,
    )


def batch_timeout_for(clauses: list) -> float:
    if LFM2D_TIMEOUT_S:
        return LFM2D_TIMEOUT_S
    chars = sum(len(c) for c in clauses)
    return min(
        TIMEOUT_CAP_S,
        TIMEOUT_BASE_S + TIMEOUT_PER_BATCH_ITEM_S * len(clauses) + chars * TIMEOUT_PER_CHAR_S,
    )


# 'advisory' (default): lfm2d is recorded, regex decides. 'off': no call at
# all. 'enforce': NOT READY — read the shortcomings above.
LFM2D_MODE = os.environ.get('LFM2D_HOOK_MODE', 'advisory')
# Which of the checkpoint's labels count as "the model flagged this", for the
# disagreement buckets only — this decides nothing, it labels rows.
#
# Configurable because the vocabulary is a property of the DEPLOYED
# checkpoint, not of this file: v8 speaks data-critical, v6 spoke
# destructive. The default matches what is deployed today; when it stops
# matching, `_disagreement` reports `vocab_mismatch` instead of quietly
# treating every verdict as unflagged.
SEVERE_LABELS = [
    l.strip() for l in os.environ.get('LFM2D_SEVERE_LABELS', 'data-critical').split(',') if l.strip()
]

CACHE_DIR = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'claude-hooks'
SOFT_BLOCK_CACHE = CACHE_DIR / 'soft-blocked.jsonl'
SOFT_BLOCK_TTL = 300  # 5 minutes
ADVISORY_LOG = CACHE_DIR / 'lfm2d-advisory.jsonl'
BREAKER_STATE = CACHE_DIR / 'lfm2d-breaker.json'

# Circuit breaker. Without one, an lfm2d outage taxes EVERY Bash call in
# every session the full LFM2D_HOOK_TIMEOUT — measured 447 ms against a
# black-holed host — for as long as the outage lasts. The verdict is
# advisory, so paying half a second per command to re-learn "still down" is
# pure loss.
#
# After BREAKER_THRESHOLD consecutive failures, stop calling for
# BREAKER_COOLDOWN_S, then let exactly one call through to test the water.
# Deliberately NOT silent: skipped calls are logged as `circuit_open` rows,
# so a quiet stretch in the advisory log is distinguishable from a stretch
# where we simply stopped asking.
BREAKER_THRESHOLD = int(os.environ.get('LFM2D_BREAKER_THRESHOLD', '3'))
BREAKER_COOLDOWN_S = float(os.environ.get('LFM2D_BREAKER_COOLDOWN', '60'))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BLOCKED / SOFT_BLOCKED / WARNED — VERBATIM from the current hook.
# Do not "improve" these here. This file's whole claim is that the regex
# half behaves identically, so any disagreement in the log is the model's,
# not a rules edit. Port changes from the dotfiles hook, don't fork them.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BLOCKED = {
    'git': [
        (r'\bgit\s+add\s+(-A|--all)\b', 'git add -A/--all', 'Use explicit paths or `git add -p`'),
        (r'\bgit\s+add\s+\.\s*($|[;&|])', 'git add .', 'Use explicit paths or `git add -p`'),
        (r'\bgit\s+commit\s+-[a-zA-Z]*a', 'git commit -a', 'Stage files explicitly first'),
        # git stash rule removed 2026-08-24, ported from the dotfiles
        # baseline (its working copy dropped the a3b6e05 stash rule; the
        # parity gate caught the drift). Port changes, don't fork.
    ],
}

SOFT_BLOCKED = {
    'git': [
        (r'\bgit\s+commit\s+.*--amend\b', 'git commit --amend', 'rewrites history'),
        (r'\bgit\s+push\s+.*--force\b', 'git push --force', 'rewrites remote history'),
        (r'\bgit\s+reset\s+--hard\b', 'git reset --hard', 'discards uncommitted changes'),
    ],
    'rm': [
        (r'\brm\s+-[a-zA-Z]*r[a-zA-Z]*f', 'rm -rf', 'recursive force delete'),
        (r'\brm\s+-[a-zA-Z]*f[a-zA-Z]*r', 'rm -fr', 'recursive force delete'),
    ],
    'find': [
        (r'\bfind\b.*-exec\b', 'find -exec', 'executes commands on matches'),
        (r'\bfind\b.*-execdir\b', 'find -execdir', 'executes commands on matches'),
        (r'\bfind\b.*-ok\b', 'find -ok', 'executes commands on matches'),
    ],
}

WARNED = {
    'rm': [
        (r'\brm\s+.*\.local/share/', 'rm in ~/.local/share', 'contains app data'),
        (r'\brm\s+.*\.config/', 'rm in ~/.config', 'contains config files'),
        (r'\brm\s+.*\.db\b', 'rm *.db files', 'databases are hard to recover'),
        (r'\brm\s+-[a-zA-Z]*r', 'rm -r', 'recursive delete'),
        (r'\brm\s+-[a-zA-Z]*f', 'rm -f', 'force delete'),
    ],
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# lfm2d advisory call
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def lfm2d_classify(cmd: str) -> dict:
    """One /v1/classify call. Returns a dict that ALWAYS has 'ok'.

    Never raises: an advisory second opinion must not be able to break the
    guard it is advising. Every failure path records why, so a silent run
    of empty verdicts is distinguishable from a run of successful ones —
    the log is the whole product of this phase and a blank field in it
    would be worse than useless.
    """
    started = time.monotonic()
    body = json.dumps({'inputs': cmd}).encode()
    req = urllib.request.Request(
        f'{LFM2D_URL}/v1/classify',
        data=body,
        headers={'content-type': 'application/json'},
        method='POST',
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_for(cmd)) as resp:
            results = json.loads(resp.read())
        r = results[0]
        return {
            'ok': True,
            'top': r['top'],
            'scores': r['scores'],
            'model_id': r['model_id'],
            # The audit pair. Label vocabularies change between checkpoints,
            # so a verdict without the hash it came from cannot be compared
            # across a redeploy — which is exactly what this log is for.
            'weight_hash': r['weight_hash'],
            'latency_ms': round((time.monotonic() - started) * 1000, 1),
        }
    except urllib.error.HTTPError as e:
        return {'ok': False, 'error': f'http {e.code}', 'detail': e.read()[:200].decode('utf-8', 'replace')}
    except Exception as e:  # timeout, DNS, connection refused, malformed body
        return {'ok': False, 'error': type(e).__name__, 'detail': str(e)[:200]}


def lfm2d_cascade(clauses: list) -> dict:
    """One /v1/cascade call over pre-split clauses. Same never-raises
    contract as `lfm2d_classify`.

    The returned dict carries `top` and `scores` at the SAME keys classify
    rows use — `top` is the winning clause's argmax label, `scores` its
    per-label probabilities — so `_disagreement` reads both row kinds
    without caring which endpoint produced them. The vocabulary check rides
    on `scores` exactly as before. Everything cascade-specific (winner,
    per-clause breakdown, lane) is additional fields, `endpoint` names
    which contract the row speaks.
    """
    started = time.monotonic()
    body = json.dumps({'clauses': clauses}).encode()
    req = urllib.request.Request(
        f'{LFM2D_URL}/v1/cascade',
        data=body,
        headers={'content-type': 'application/json'},
        method='POST',
    )
    try:
        with urllib.request.urlopen(req, timeout=cascade_timeout_for(clauses)) as resp:
            r = json.loads(resp.read())
        winner = r['winner']
        rows = r['clauses']
        models = r.get('models') or []
        # models[0] is the classifier by the server's contract
        # (engine_real.rs assembles [classifier, router]); keep the full
        # array too so the row never loses the router's identity.
        clf = models[0] if models else {}
        return {
            'ok': True,
            'endpoint': 'cascade',
            'top': rows[winner['index']]['top_severity'],
            'scores': winner['severity_scores'],
            'winner_index': winner['index'],
            'winner_clause': winner['clause'],
            'clause_count': len(clauses),
            'clauses': [
                {'clause': c['clause'], 'top': c['top_severity'], 'scores': c['severity_scores']}
                for c in rows
            ],
            'lane': r.get('lane'),
            'model_id': clf.get('model_id'),
            'weight_hash': clf.get('weight_hash'),
            'models': models,
            'latency_ms': round((time.monotonic() - started) * 1000, 1),
        }
    except urllib.error.HTTPError as e:
        return {'ok': False, 'endpoint': 'cascade', 'error': f'http {e.code}',
                'detail': e.read()[:200].decode('utf-8', 'replace')}
    except Exception as e:  # timeout, DNS, connection refused, malformed body
        return {'ok': False, 'endpoint': 'cascade', 'error': type(e).__name__, 'detail': str(e)[:200]}


def lfm2d_classify_batch(clauses: list) -> dict:
    """One batched /v1/classify over pre-split clauses — the fallback for
    commands with more clauses than cascade can serve under the cap.

    Deliberately carries NO row-level `top`/`scores` and NO winner: picking
    one clause to represent the command IS the severity aggregation, and
    that belongs to the daemon (`ordinal-collapsed-to-a-set` is the memory
    this respects). `_disagreement` handles these rows explicitly with a
    set-membership check — did ANY clause's argmax land in the severe set —
    which is a recall question, not a ranking."""
    started = time.monotonic()
    body = json.dumps({'inputs': clauses}).encode()
    req = urllib.request.Request(
        f'{LFM2D_URL}/v1/classify',
        data=body,
        headers={'content-type': 'application/json'},
        method='POST',
    )
    try:
        with urllib.request.urlopen(req, timeout=batch_timeout_for(clauses)) as resp:
            results = json.loads(resp.read())
        first = results[0] if results else {}
        return {
            'ok': True,
            'endpoint': 'classify_batch',
            'clause_count': len(clauses),
            'clauses': [
                {'clause': cl, 'top': r['top'], 'scores': r['scores']}
                for cl, r in zip(clauses, results)
            ],
            'model_id': first.get('model_id'),
            'weight_hash': first.get('weight_hash'),
            'latency_ms': round((time.monotonic() - started) * 1000, 1),
        }
    except urllib.error.HTTPError as e:
        return {'ok': False, 'endpoint': 'classify_batch', 'error': f'http {e.code}',
                'detail': e.read()[:200].decode('utf-8', 'replace')}
    except Exception as e:  # timeout, DNS, connection refused, malformed body
        return {'ok': False, 'endpoint': 'classify_batch', 'error': type(e).__name__, 'detail': str(e)[:200]}


def breaker_read() -> dict:
    try:
        return json.loads(BREAKER_STATE.read_text())
    except Exception:
        return {'consecutive_failures': 0, 'open_until': 0.0}


def breaker_write(state: dict):
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        BREAKER_STATE.write_text(json.dumps(state))
    except Exception:
        pass


def breaker_should_skip() -> bool:
    return time.time() < breaker_read().get('open_until', 0.0)


def breaker_record(ok: bool):
    """One success closes the circuit outright.

    Not a decaying counter or a rolling window: this guards a local network
    call with no cost to being wrong in the optimistic direction, and a
    single good response is sufficient evidence the service came back.
    """
    state = breaker_read()
    if ok:
        breaker_write({'consecutive_failures': 0, 'open_until': 0.0})
        return
    fails = state.get('consecutive_failures', 0) + 1
    open_until = time.time() + BREAKER_COOLDOWN_S if fails >= BREAKER_THRESHOLD else 0.0
    breaker_write({'consecutive_failures': fails, 'open_until': open_until})


def log_advisory(cmd: str, regex_verdict: dict, lfm2d_verdict: dict):
    """Append one comparison row. Best-effort: logging must never block a command."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        row = {
            'ts': time.time(),
            'cwd': os.getcwd(),
            'command': cmd,
            'regex': regex_verdict,
            'lfm2d': lfm2d_verdict,
            # Precomputed so analysis is a grep, not a join. 'regex_only' is
            # the false-positive candidate pile (the 6:1 ratio we are trying
            # to fix); 'lfm2d_only' is the recall pile (what the regex misses
            # and we would gain).
            'disagree': _disagreement(regex_verdict, lfm2d_verdict),
        }
        with open(ADVISORY_LOG, 'a') as f:
            f.write(json.dumps(row) + '\n')
        ADVISORY_LOG.chmod(0o600)  # commands can contain secrets
    except Exception:
        pass


def _disagreement(regex_verdict: dict, lfm2d_verdict: dict) -> str:  # noqa: C901
    """Bucket one comparison. Returns 'vocab_mismatch' rather than guessing
    when the checkpoint doesn't speak the labels we're keying on.

    fleet.md: "Label vocabularies change between checkpoints — consumers MUST
    read labels at runtime; hard-coded label strings are a live breakage."
    Comparing `top` against a fixed string is exactly that. Roll the pod back
    to v6 (informative/mutating/destructive) and every `destructive` verdict
    silently buckets as "the model did not flag it" — an entire deployment of
    confidently wrong rows, with nothing in the log to say so.

    No extra round trip is needed to avoid it: every /v1/classify response
    carries the checkpoint's full label set in `scores`, so the vocabulary is
    already in hand on every call.
    """
    if not lfm2d_verdict.get('ok'):
        return 'circuit_open' if lfm2d_verdict.get('error') == 'circuit_open' else 'no_verdict'

    if lfm2d_verdict.get('endpoint') == 'classify_batch':
        # No winner exists on these rows by design (see lfm2d_classify_batch)
        # — flagging is set-membership over per-clause argmaxes, not ranking.
        clause_rows = lfm2d_verdict.get('clauses') or []
        known = [l for l in SEVERE_LABELS
                 if any(l in (c.get('scores') or {}) for c in clause_rows)]
        if not known:
            return 'vocab_mismatch'
        lfm2d_flagged = any(c.get('top') in known for c in clause_rows)
    else:
        scores = lfm2d_verdict.get('scores') or {}
        known = [l for l in SEVERE_LABELS if l in scores]
        if not known:
            return 'vocab_mismatch'
        lfm2d_flagged = lfm2d_verdict['top'] in known

    regex_flagged = regex_verdict['decision'] in ('deny', 'soft_deny', 'warn')
    if regex_flagged and lfm2d_flagged:
        return 'agree_flag'
    if not regex_flagged and not lfm2d_flagged:
        return 'agree_clear'
    return 'regex_only' if regex_flagged else 'lfm2d_only'


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Soft-block cache — VERBATIM from the current hook
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def cmd_hash(cmd: str) -> str:
    return hashlib.sha256(cmd.encode()).hexdigest()[:16]


def cache_soft_block(cmd: str, name: str):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    entry = {'hash': cmd_hash(cmd), 'name': name, 'ts': time.time()}
    with open(SOFT_BLOCK_CACHE, 'a') as f:
        f.write(json.dumps(entry) + '\n')


def check_and_consume_approval(cmd: str) -> Optional[str]:
    if not SOFT_BLOCK_CACHE.exists():
        return None

    h = cmd_hash(cmd)
    now = time.time()
    kept, matched_name = [], None

    with open(SOFT_BLOCK_CACHE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if now - entry.get('ts', 0) > SOFT_BLOCK_TTL:
                continue
            if entry.get('hash') == h and matched_name is None:
                matched_name = entry.get('name', 'command')
                continue
            kept.append(entry)

    if matched_name is not None or len(kept) == 0:
        if kept:
            with open(SOFT_BLOCK_CACHE, 'w') as f:
                for entry in kept:
                    f.write(json.dumps(entry) + '\n')
        else:
            SOFT_BLOCK_CACHE.unlink(missing_ok=True)

    return matched_name


def get_git_status_summary() -> Optional[dict]:
    try:
        r = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True, timeout=5)
        if r.returncode != 0:
            return None
        lines = [l for l in r.stdout.strip().split('\n') if l]
        return {
            'untracked': sum(1 for l in lines if l.startswith('??')),
            'modified': sum(1 for l in lines if len(l) > 1 and l[1] in 'MD'),
            'total': len(lines),
        }
    except Exception:
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Hook responses
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def emit(decision: str, reason: Optional[str] = None):
    out = {'hookSpecificOutput': {'hookEventName': 'PreToolUse', 'permissionDecision': decision}}
    if reason is not None:
        out['hookSpecificOutput']['permissionDecisionReason'] = reason
    print(json.dumps(out))


def check_patterns(cmd: str, patterns: dict) -> Optional[tuple]:
    for category, rules in patterns.items():
        for pattern, name, hint in rules:
            if re.search(pattern, cmd):
                return (category, name, hint)
    return None


def regex_verdict_for(cmd: str) -> dict:
    """The CURRENT hook's decision, as data. Pure — no I/O, no side effects.

    Split out from the acting code so the advisory log records exactly what
    the regex half decided, and so it can be tested without a subprocess.
    """
    match = check_patterns(cmd, BLOCKED)
    if match:
        category, name, hint = match
        return {'decision': 'deny', 'rule': name, 'hint': hint, 'category': category}

    match = check_patterns(cmd, SOFT_BLOCKED)
    if match:
        category, name, reason = match
        return {'decision': 'soft_deny', 'rule': name, 'hint': reason, 'category': category}

    match = check_patterns(cmd, WARNED)
    if match:
        category, name, reason = match
        return {'decision': 'warn', 'rule': name, 'hint': reason, 'category': category}

    return {'decision': 'allow', 'rule': None, 'hint': None, 'category': None}


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        emit('allow')
        return

    if data.get('tool_name') != 'Bash':
        emit('allow')
        return

    cmd = data.get('tool_input', {}).get('command', '')

    # The approved-retry path short-circuits before any advisory call: the
    # user has already ruled on this exact command, and a model opinion on a
    # settled question is noise.
    approved = check_and_consume_approval(cmd)
    if approved:
        emit('allow', f"✅ `{approved}` approved by user, allowing retry.")
        return

    verdict = regex_verdict_for(cmd)

    lfm2d = {'ok': False, 'error': 'disabled'}
    if LFM2D_MODE != 'off':
        if breaker_should_skip():
            # Still logged. A skipped call and a healthy quiet period must not
            # look the same in the advisory log, or the record silently
            # overstates how much of the session lfm2d actually saw.
            lfm2d = {'ok': False, 'error': 'circuit_open',
                     'detail': f'{BREAKER_THRESHOLD}+ consecutive failures; not retrying yet'}
        else:
            # Plan-first: `kaish --plan` renders the simple commands; the
            # regex splitter is the recorded fallback for bash kaish
            # rejects (~13% at the 08-23 floor baseline). Compound
            # commands go to /v1/cascade so the severe clause is ranked
            # instead of diluted; a single clause goes to /v1/classify.
            # Belt to kaish_plan's own suspenders: a raise ANYWHERE in the
            # plan path would crash the hook before emit(), silently
            # disabling the regex guard. The fallback must be unreachable
            # by exception (kaibo review 2026-08-24).
            try:
                plan = plan_clauses(cmd)
            except Exception as _pe:
                plan = {'ok': False, 'error': f'plan_raise:{type(_pe).__name__}',
                        'detail': str(_pe)[:200]}
            if plan.get('ok'):
                split_path = 'kaish_plan'
                # Dedupe for the wire: an identical clause text scores
                # identically, so duplicates are pure daemon work — a
                # degenerate `a && b && a && b …` monster planned 177
                # clauses (2 distinct) and timed out a batch call at the
                # cap, backlogging the single worker for the calls behind
                # it. Then bound like the splitter (MAX_CLAUSES=64) so a
                # monster with many DISTINCT clauses stays under budget
                # too; both reductions are recorded on the row.
                planned = plan['clauses']
                seen, clause_rows = set(), []
                for c in planned:
                    if c['text'] not in seen:
                        seen.add(c['text'])
                        clause_rows.append(c)
                truncated = max(0, len(clause_rows) - PLAN_MAX_CLAUSES)
                clause_rows = clause_rows[:PLAN_MAX_CLAUSES]
                clauses = [c['text'] for c in clause_rows]
                plan_meta = {
                    'kaish_version': plan.get('kaish_version'),
                    'statement_count': plan.get('statement_count'),
                    # Counted over the PRE-dedupe plan so the fallback rate
                    # can't be hidden by a duplicate being deduped away;
                    # the heredocs list below indexes the SENT clauses so
                    # rows correlate with what was scored.
                    'stmt_fallbacks': sum(1 for c in planned if c.get('stmt_fallback')),
                    'clauses_planned': len(planned),
                    'clauses_deduped': len(planned) - (len(clause_rows) + truncated),
                    'clauses_truncated': truncated,
                    'heredocs': [
                        {'clause_index': i, **c['heredoc']}
                        for i, c in enumerate(clause_rows) if c.get('heredoc')
                    ],
                    # Verb and redirect shape per SENT clause, indexed the
                    # same way heredocs are. The clause text carries these
                    # too, but only as text: recovering "does this redirect,
                    # and where" from the rendered string means re-parsing
                    # it, which is the prose-reading mistake kaish_plan
                    # exists to avoid. Recorded so an offline read can ask
                    # "which verbs won a cascade, and did any of them write"
                    # without a second parser -- that question had to be
                    # answered by splitting text on whitespace as recently
                    # as today. `args` is deliberately NOT logged: it is
                    # already in `text` verbatim and would be most of the
                    # added bytes. Rows with nothing to say (a pure
                    # assignment, a statement-level fallback) are omitted,
                    # exactly as heredocs are.
                    'commands': [
                        {'clause_index': i, 'name': c['name'], 'redirects': c['redirects']}
                        for i, c in enumerate(clause_rows)
                        if c.get('name') or c.get('redirects')
                    ],
                }
            else:
                split_path = 'clause_split'
                clauses = split_clauses(cmd)
                plan_meta = None
            if len(clauses) > CASCADE_MAX_CLAUSES:
                lfm2d = lfm2d_classify_batch(clauses)
            elif len(clauses) >= 2:
                lfm2d = lfm2d_cascade(clauses)
            else:
                sent = clauses[0] if clauses else cmd
                lfm2d = lfm2d_classify(sent)
                lfm2d['endpoint'] = 'classify'
                if sent != cmd:
                    # The splitter stripped comments/keywords; record what
                    # was actually scored so the row can't mislead analysis.
                    lfm2d['sent'] = sent
            # Which extraction produced the scored clauses, on EVERY row —
            # the live fallback rate is a slice-1 deliverable, and a row
            # that doesn't say its path can't be windowed by it later.
            lfm2d['split_path'] = split_path
            if plan_meta is not None:
                lfm2d['plan'] = plan_meta
            else:
                lfm2d['plan_error'] = plan.get('error')
                lfm2d['plan_error_detail'] = plan.get('detail')
            if _SPLIT_IMPORT_ERROR:
                lfm2d['split_import_error'] = _SPLIT_IMPORT_ERROR
            breaker_record(lfm2d.get('ok', False))
        log_advisory(cmd, verdict, lfm2d)

    if LFM2D_MODE == 'enforce':
        # Deliberately unimplemented. Wiring an untested enforcement path
        # and leaving it reachable is how a "temporary" mode ships. Read the
        # measured shortcomings at the top of this file; enforcement needs a
        # v9 slice covering data-position negatives first.
        print(
            'lfm2d hook: LFM2D_HOOK_MODE=enforce is not implemented — '
            'refusing to guess at a decision. Falling back to advisory.',
            file=sys.stderr,
        )

    # --- advisory: the regex decides, exactly as it does today ---
    d = verdict['decision']
    if d == 'deny':
        if verdict['category'] == 'git':
            status = get_git_status_summary()
            count = status['total'] if status else '?'
            emit('deny', f"⛔ Blocked: `{verdict['rule']}` would stage {count} files. {verdict['hint']}")
        else:
            emit('deny', f"⛔ Blocked: `{verdict['rule']}`. {verdict['hint']}")
    elif d == 'soft_deny':
        cache_soft_block(cmd, verdict['rule'])
        emit(
            'deny',
            f"🛑 `{verdict['rule']}` blocked ({verdict['hint']}). "
            f"If intentional, ask the user to confirm before retrying.",
        )
    elif d == 'warn':
        emit('allow', f"⚠️ `{verdict['rule']}` ({verdict['hint']})")
    else:
        emit('allow')


if __name__ == '__main__':
    main()
