#!/usr/bin/env python3
"""Integration tests for the advisory hook. REQUIRES a reachable lfm2d.

Run:  python3 lfm2d/hooks/test_advisory_live.py
      LFM2D_URL=http://host:8088 python3 lfm2d/hooks/test_advisory_live.py

Fails loudly rather than skipping when lfm2d is unreachable. A guard-rail
suite that quietly turns into a no-op is worse than no suite: it reports
green while testing nothing, which is the failure mode this project keeps
writing memories about.

Two kinds of test here, and the split is deliberate:

  LIVE    — against the real daemon. These are the ones that can catch
            "the contract changed under us."
  STUB    — against a local HTTP stub we control. These cover the paths a
            healthy daemon cannot produce on demand: a different label
            vocabulary, a malformed body, a hang. You cannot test a
            checkpoint rollback by asking production to roll back.

`test_parity.py` covers the regex half offline. This file covers what
happens once lfm2d is in the loop.
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
HOOK = HERE / 'pre_command_advisory.py'
LFM2D_URL = os.environ.get('LFM2D_URL', 'http://lfm2d-1.taila4abc.ts.net:8088')

PASS, FAIL = [], []


def check(name: str, ok: bool, detail: str = ''):
    (PASS if ok else FAIL).append(name)
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}")
    if detail and not ok:
        print(f"        {detail}")


# ─────────────────────────────────────────────────────────────── harness ────
def run_hook(command: str, cache: Path, env_extra: dict = None, tool: str = 'Bash') -> tuple:
    """Run the hook once. Returns (decision-dict, log-rows-written, wall_seconds)."""
    payload = (
        {'tool_name': tool, 'tool_input': {'command': command}}
        if tool == 'Bash'
        else {'tool_name': tool, 'tool_input': {'file_path': command}}
    )
    env = dict(os.environ)
    env['XDG_CACHE_HOME'] = str(cache)
    env['LFM2D_URL'] = LFM2D_URL
    env.pop('LFM2D_HOOK_MODE', None)
    env.pop('LFM2D_HOOK_TIMEOUT', None)
    env.update(env_extra or {})

    log = cache / 'claude-hooks' / 'lfm2d-advisory.jsonl'
    before = log.read_text().count('\n') if log.exists() else 0

    started = time.monotonic()
    p = subprocess.run(
        [sys.executable, str(HOOK)], input=json.dumps(payload),
        capture_output=True, text=True, timeout=60, env=env,
    )
    wall = time.monotonic() - started

    rows = []
    if log.exists():
        rows = [json.loads(l) for l in log.read_text().splitlines() if l.strip()][before:]
    try:
        decision = json.loads(p.stdout)
    except json.JSONDecodeError:
        decision = {'__unparseable__': p.stdout, '__stderr__': p.stderr}
    return decision, rows, wall


class StubHandler(BaseHTTPRequestHandler):
    """Serves whatever `ThreadingHTTPServer.scripted` says. Silent."""

    def do_POST(self):
        body, status = self.server.scripted
        if body is None:  # simulate a hang
            time.sleep(5)
            return
        raw = json.dumps(body).encode() if not isinstance(body, bytes) else body
        self.send_response(status)
        self.send_header('content-type', 'application/json')
        self.send_header('content-length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a):
        pass


class stub_lfm2d:
    """Context manager: a local stand-in for lfm2d returning a scripted body."""

    def __init__(self, body, status=200):
        self.body, self.status = body, status

    def __enter__(self):
        self.srv = ThreadingHTTPServer(('127.0.0.1', 0), StubHandler)
        self.srv.scripted = (self.body, self.status)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        return f'http://127.0.0.1:{self.srv.server_address[1]}'

    def __exit__(self, *a):
        self.srv.shutdown()


def v8_response(top='data-critical'):
    return [{
        'scores': {'informative': 0.01, 'situation-normal': 0.02, 'data-critical': 0.97},
        'top': top,
        'model_id': 'kube_ordinal_v8',
        'weight_hash': 'deadbeef' * 8,
    }]


def v6_response():
    """A DIFFERENT checkpoint's vocabulary — v6 spoke informative/mutating/
    destructive. This is the rollback case fleet.md warns about."""
    return [{
        'scores': {'informative': 0.01, 'mutating': 0.02, 'destructive': 0.97},
        'top': 'destructive',
        'model_id': 'kube_ordinal_v6',
        'weight_hash': 'cafebabe' * 8,
    }]


def v6_cascade_response():
    """The same rollback case, but cascade-shaped — compound commands take
    the /v1/cascade path now, so the vocabulary guard must hold there too."""
    scores = {'informative': 0.01, 'mutating': 0.02, 'destructive': 0.97}
    return {
        'winner': {'index': 1, 'clause': 'rm -rf ./x', 'severity_scores': scores},
        'lane': {'route': 'destructive-lane', 'cosine': 0.5},
        'clauses': [
            {'index': 0, 'clause': 'cd /tmp', 'severity_scores': scores, 'top_severity': 'informative'},
            {'index': 1, 'clause': 'rm -rf ./x', 'severity_scores': scores, 'top_severity': 'destructive'},
        ],
        'models': [
            {'model_id': 'kube_ordinal_v6', 'weight_hash': 'cafebabe' * 8},
            {'model_id': 'router', 'weight_hash': 'beefcafe' * 8},
        ],
    }


# ────────────────────────────────────────────────────────────── preflight ────
def preflight() -> dict:
    try:
        with urllib.request.urlopen(f'{LFM2D_URL}/v1/models', timeout=10) as r:
            models = json.loads(r.read())
    except Exception as e:
        print(f"\nCANNOT RUN: lfm2d unreachable at {LFM2D_URL} ({type(e).__name__}: {e})")
        print("These tests require a running lfm2d. Set LFM2D_URL, or start the daemon.")
        print("Refusing to skip — a guard-rail suite that silently no-ops reports")
        print("green while testing nothing.")
        sys.exit(2)

    classifiers = [m for m in models if m['kind'] == 'classifier']
    if not classifiers:
        print(f"\nCANNOT RUN: {LFM2D_URL} has no classifier head loaded.")
        print(f"Loaded: {[(m['id'], m['kind']) for m in models]}")
        sys.exit(2)
    c = classifiers[0]
    print(f"lfm2d:      {LFM2D_URL}")
    print(f"classifier: {c['id']} {c['weight_hash'][:12]}… labels={c.get('labels')}\n")
    return c


# ────────────────────────────────────────────────────────────────── tests ────
def test_advisory_never_changes_the_decision(_c):
    """THE invariant. Advisory means advisory: for every command, the decision
    with lfm2d in the loop must equal the decision with it switched off."""
    cases = [
        'ls -la /home', 'npm install', 'cd db/migrations',
        'git add -A', 'git stash push -m x', 'git commit -am wip',
        'git push --force origin main', 'git reset --hard origin/main',
        'find . -name "*.tmp" -exec echo {} ;',
        'kubectl delete namespace prod', 'rm -r ./build', 'rm -f ./file',
        'curl -s -X POST http://h/v1/classify -d \'{"inputs":"danger"}\'',
        '', 'set -e\necho done',
    ]
    bad = []
    for cmd in cases:
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            off, _, _ = run_hook(cmd, Path(a), {'LFM2D_HOOK_MODE': 'off'})
            adv, _, _ = run_hook(cmd, Path(b), {'LFM2D_HOOK_MODE': 'advisory'})
        if off != adv:
            bad.append((cmd, off, adv))
    check('advisory never changes the decision', not bad,
          f"{len(bad)} differ, first: {bad[0] if bad else ''}")


def test_every_bash_call_logs_one_row(_c):
    with tempfile.TemporaryDirectory() as d:
        _, rows, _ = run_hook('ls -la /home', Path(d))
        check('one log row per Bash call', len(rows) == 1, f'got {len(rows)}')


def test_log_row_carries_the_audit_pair(c):
    """A verdict without the weight hash it came from cannot be compared
    across a redeploy — and comparing across redeploys is what this log is
    for. Must match the deployed classifier, not merely be present."""
    with tempfile.TemporaryDirectory() as d:
        _, rows, _ = run_hook('kubectl delete namespace prod', Path(d))
    if not rows:
        return check('log row carries the audit pair', False, 'no row written')
    v = rows[0]['lfm2d']
    ok = v.get('ok') and v.get('model_id') == c['id'] and v.get('weight_hash') == c['weight_hash']
    check('log row carries the audit pair', ok,
          f"got model_id={v.get('model_id')} hash={str(v.get('weight_hash'))[:12]}… "
          f"want {c['id']} {c['weight_hash'][:12]}…")


def test_log_row_is_complete(_c):
    with tempfile.TemporaryDirectory() as d:
        _, rows, _ = run_hook('rm -rf ./build', Path(d))
    if not rows:
        return check('log row is complete', False, 'no row written')
    r = rows[0]
    missing = [k for k in ('ts', 'cwd', 'command', 'regex', 'lfm2d', 'disagree') if k not in r]
    ok = not missing and r['command'] == 'rm -rf ./build' and 'latency_ms' in r['lfm2d']
    check('log row is complete', ok, f'missing={missing} row={r}')


def test_disagreement_buckets_are_right(_c):
    """Characterization test against the DEPLOYED checkpoint. If this fails
    after a model change, that is the signal working — read the new numbers
    before editing the expectations.

    2026-08-24, read against kube_ordinal_v9_cal (deployed 08-16): the
    tau=0.5 prior calibration moved the argmax down across the board, and
    `kubectl delete namespace prod` now argmaxes situation-normal (dc
    0.250). Probed for a replacement lfm2d_only command and found NONE
    that flags reliably (`git clean -fdx` 0.271, `rm /etc/hosts` 0.411,
    `sed -i …/sshd_config` 0.353, `rm src/main.rs` 0.226) — under the
    calibrated head no single short command the regex allows argmaxes
    data-critical. That bucket's LOGIC is now pinned by the stubbed test
    below instead of by a live command's mood."""
    expected = {
        'ls -la /home': 'agree_clear',
        'rm -rf /var/lib/data': 'agree_flag',
        'kubectl delete namespace prod': 'agree_clear',
    }
    bad = []
    for cmd, want in expected.items():
        with tempfile.TemporaryDirectory() as d:
            _, rows, _ = run_hook(cmd, Path(d))
        got = rows[0]['disagree'] if rows else 'NO ROW'
        if got != want:
            bad.append(f'{cmd!r}: want {want}, got {got}')
    check('disagreement buckets match the deployed model', not bad, '; '.join(bad))


def test_lfm2d_only_bucket_is_reachable(_c):
    """The lfm2d_only bucket (model flags, regex allows — the recall pile)
    pinned deterministically with a stubbed daemon, since the deployed
    v9_cal checkpoint no longer produces it on any known stable command
    (see the characterization test above)."""
    with stub_lfm2d(v8_response()) as url:
        with tempfile.TemporaryDirectory() as d:
            _, rows, _ = run_hook('ls -la /home', Path(d), {'LFM2D_URL': url})
    got = rows[0]['disagree'] if rows else 'NO ROW'
    check('lfm2d_only bucket is reachable', got == 'lfm2d_only',
          f'want lfm2d_only, got {got}')


def test_split_path_is_recorded_and_pipelines_cascade(_c):
    """The live fallback rate is a slice-1 deliverable, so every row must
    say which extraction produced its clauses — and the deliberate
    pipeline-split reversal must be visible at the endpoint: a two-member
    pipeline now cascades (2 clauses) where clause_split sent one. The
    fallback row's path must say clause_split (kaish rejects the unquoted
    `===` adjacency)."""
    cases = [
        ('cat README.md | grep -n pattern', 'kaish_plan', 'cascade'),
        ('echo === step ===', 'clause_split', 'classify'),
    ]
    bad = []
    for cmd, want_path, want_ep in cases:
        with tempfile.TemporaryDirectory() as d:
            _, rows, _ = run_hook(cmd, Path(d))
        v = rows[0]['lfm2d'] if rows else {}
        if v.get('split_path') != want_path or v.get('endpoint') != want_ep:
            bad.append(f'{cmd!r}: path={v.get("split_path")} ep={v.get("endpoint")}, '
                       f'want {want_path}/{want_ep}')
    check('split_path recorded, pipeline takes cascade', not bad, '; '.join(bad))


def test_non_bash_does_not_reach_lfm2d(_c):
    """Read/Write/Edit inputs are file paths and file contents. Sending them
    to a network service — and writing them to the advisory log — would widen
    this hook's data exposure far past the Bash commands it exists to judge."""
    bad = []
    for tool in ('Read', 'Write', 'Edit'):
        with tempfile.TemporaryDirectory() as d:
            _, rows, _ = run_hook('/etc/shadow', Path(d), tool=tool)
        if rows:
            bad.append(f'{tool} wrote {len(rows)} row(s)')
    check('non-Bash tools never reach lfm2d', not bad, '; '.join(bad))


def test_approved_retry_does_not_reach_lfm2d(_c):
    """The user has already ruled on this exact command. A model opinion on a
    settled question is noise, and it costs a round trip on the retry path."""
    with tempfile.TemporaryDirectory() as d:
        cache = Path(d)
        cmd = 'rm -rf ./build'
        run_hook(cmd, cache)              # first call soft-denies and caches
        _, rows, _ = run_hook(cmd, cache)  # retry consumes the approval
    check('approved retry does not reach lfm2d', len(rows) == 0, f'{len(rows)} row(s)')


def test_log_is_not_world_readable(_c):
    with tempfile.TemporaryDirectory() as d:
        run_hook('echo hi', Path(d))
        log = Path(d) / 'claude-hooks' / 'lfm2d-advisory.jsonl'
        mode = log.stat().st_mode & 0o777
    check('advisory log is 0600', mode == 0o600, f'mode={oct(mode)}')


def test_live_latency_is_bounded(_c):
    """Every Bash call in every session pays this. Generous bound — this is a
    smoke alarm for 'the endpoint got slow', not a benchmark."""
    walls = []
    for _ in range(5):
        with tempfile.TemporaryDirectory() as d:
            _, _, w = run_hook('kubectl delete namespace prod', Path(d))
            walls.append(w)
    worst = max(walls)
    check('live hook latency under 2s', worst < 2.0,
          f'worst {worst*1000:.0f}ms of {[f"{w*1000:.0f}" for w in walls]}')


def test_compound_command_takes_the_cascade_path(c):
    """Compound commands must reach /v1/cascade — whole-command classify is
    the distorted instrument (11x dilution, measured). The row must carry
    the winner, the per-clause breakdown, and the CLASSIFIER's audit pair
    (models[0] by the server's contract), or the log can't be analyzed
    across a redeploy."""
    with tempfile.TemporaryDirectory() as d:
        _, rows, _ = run_hook('cd /tmp && rm -rf ./build', Path(d))
    if not rows:
        return check('compound command takes the cascade path', False, 'no row written')
    v = rows[0]['lfm2d']
    ok = (v.get('ok') and v.get('endpoint') == 'cascade'
          and v.get('clause_count') == 2 and len(v.get('clauses', [])) == 2
          and 'winner_clause' in v and 'winner_index' in v
          and v.get('model_id') == c['id'] and v.get('weight_hash') == c['weight_hash'])
    check('compound command takes the cascade path', bool(ok), f'row={v}')


def test_cascade_winner_is_the_severe_clause(_c):
    """The dilution fixture, live, in miniature: a destructive clause inside
    a loop must WIN the in-statement ranking instead of being averaged away.
    Characterization against the deployed checkpoint — measured 2026-08-12,
    the rm clause scored 0.540 alone vs 0.047 diluted into its script."""
    cmd = 'for d in worktree-a worktree-b; do\n  echo "cleaning $d"\n  rm -rf -- "$d.venv"\ndone'
    with tempfile.TemporaryDirectory() as d:
        _, rows, _ = run_hook(cmd, Path(d))
    if not rows or not rows[0]['lfm2d'].get('ok'):
        return check('cascade winner is the severe clause', False,
                      f'row={rows[0]["lfm2d"] if rows else None}')
    v = rows[0]['lfm2d']
    ok = v['endpoint'] == 'cascade' and v['winner_clause'].startswith('rm -rf')
    check('cascade winner is the severe clause', ok,
          f"winner={v.get('winner_clause')!r} of {v.get('clause_count')} clauses")


def test_quoted_payload_is_not_split(_c):
    """The named hazard: a curl whose JSON body contains `rm -rf && …` must
    reach the classifier as ONE clause. Cutting inside the payload would
    fabricate a bare `rm -rf` out of data — the exact discrimination this
    product exists to make."""
    cmd = 'curl -s -X POST http://h/v1/classify -d \'{"inputs":"rm -rf /var/lib/data && echo done"}\''
    with tempfile.TemporaryDirectory() as d:
        _, rows, _ = run_hook(cmd, Path(d))
    if not rows:
        return check('quoted payload is not split', False, 'no row written')
    v = rows[0]['lfm2d']
    ok = v.get('ok') and v.get('endpoint') == 'classify'
    check('quoted payload is not split', bool(ok),
          f"endpoint={v.get('endpoint')} (want classify: 1 clause)")


# ── stub-backed: paths a healthy daemon cannot produce on demand ──────────
def test_unreachable_lfm2d_fails_open_and_says_so(_c):
    with tempfile.TemporaryDirectory() as d:
        dec, rows, _ = run_hook('ls -la /home', Path(d), {'LFM2D_URL': 'http://127.0.0.1:9'})
    allowed = dec.get('hookSpecificOutput', {}).get('permissionDecision') == 'allow'
    logged = rows and rows[0]['disagree'] == 'no_verdict' and rows[0]['lfm2d'].get('error')
    check('unreachable lfm2d fails open, with the reason recorded', allowed and logged,
          f'decision={dec} row={rows[0] if rows else None}')


def test_timeout_is_honored(_c):
    """A wedged endpoint must cost the session a bounded wait, not a hang."""
    with stub_lfm2d(None) as url:  # handler sleeps 5s
        with tempfile.TemporaryDirectory() as d:
            dec, rows, wall = run_hook(
                'ls -la /home', Path(d), {'LFM2D_URL': url, 'LFM2D_HOOK_TIMEOUT': '0.2'})
    allowed = dec.get('hookSpecificOutput', {}).get('permissionDecision') == 'allow'
    timed_out = rows and rows[0]['lfm2d'].get('error') in ('TimeoutError', 'URLError', 'timeout')
    check('timeout is honored and fails open', allowed and timed_out and wall < 3.0,
          f'wall={wall:.2f}s decision={dec} row={rows[0]["lfm2d"] if rows else None}')


def test_http_error_fails_open(_c):
    with stub_lfm2d({'error': {'message': 'boom', 'type': 'internal'}}, status=500) as url:
        with tempfile.TemporaryDirectory() as d:
            dec, rows, _ = run_hook('ls -la /home', Path(d), {'LFM2D_URL': url})
    allowed = dec.get('hookSpecificOutput', {}).get('permissionDecision') == 'allow'
    logged = rows and rows[0]['lfm2d'].get('error', '').startswith('http 500')
    check('http 500 fails open, with the reason recorded', allowed and logged,
          f'row={rows[0]["lfm2d"] if rows else None}')


def test_malformed_body_fails_open(_c):
    with stub_lfm2d(b'not json at all') as url:
        with tempfile.TemporaryDirectory() as d:
            dec, rows, _ = run_hook('ls -la /home', Path(d), {'LFM2D_URL': url})
    allowed = dec.get('hookSpecificOutput', {}).get('permissionDecision') == 'allow'
    logged = rows and not rows[0]['lfm2d'].get('ok')
    check('malformed response fails open', allowed and logged, f'rows={rows}')


def test_unknown_label_vocabulary_is_loud(_c):
    """fleet.md: "Label vocabularies change between checkpoints — consumers
    MUST read labels at runtime; hard-coded label strings are a live
    breakage."

    A rollback to v6 (informative/mutating/destructive) must NOT silently
    bucket every command as agree_clear just because the string
    'data-critical' stops appearing. That would report "the model sees
    nothing severe" for an entire deployment — a silent wrong answer, which
    is the one thing this project refuses.

    Every /v1/classify response carries the full label set in `scores`, so
    this is checkable per call with no extra round trip.
    """
    with stub_lfm2d(v6_response()) as url:
        with tempfile.TemporaryDirectory() as d:
            dec, rows, _ = run_hook('rm -rf /var/lib/data', Path(d), {'LFM2D_URL': url})
    allowed = 'permissionDecision' in dec.get('hookSpecificOutput', {})
    bucket = rows[0]['disagree'] if rows else 'NO ROW'
    check('unknown label vocabulary is reported, not silently cleared',
          allowed and bucket == 'vocab_mismatch',
          f"bucket={bucket!r} (want 'vocab_mismatch'); row={rows[0]['lfm2d'] if rows else None}")


def test_known_vocabulary_still_classifies(_c):
    """The guard above must not fire on the deployed vocabulary."""
    with stub_lfm2d(v8_response()) as url:
        with tempfile.TemporaryDirectory() as d:
            _, rows, _ = run_hook('ls -la /home', Path(d), {'LFM2D_URL': url})
    bucket = rows[0]['disagree'] if rows else 'NO ROW'
    check('known vocabulary classifies normally', bucket == 'lfm2d_only', f'bucket={bucket}')


def test_unknown_vocabulary_is_loud_on_the_cascade_path(_c):
    """Same rollback guard, cascade shape: a compound command's winner
    speaking v6's labels must bucket vocab_mismatch, not silently clear."""
    with stub_lfm2d(v6_cascade_response()) as url:
        with tempfile.TemporaryDirectory() as d:
            dec, rows, _ = run_hook('cd /tmp && echo hi', Path(d), {'LFM2D_URL': url})
    allowed = 'permissionDecision' in dec.get('hookSpecificOutput', {})
    bucket = rows[0]['disagree'] if rows else 'NO ROW'
    check('unknown vocabulary is loud on the cascade path',
          allowed and bucket == 'vocab_mismatch',
          f"bucket={bucket!r}; row={rows[0]['lfm2d'] if rows else None}")


def test_cascade_http_error_fails_open(_c):
    """The cascade path's failure modes must degrade exactly like classify's:
    fail open, reason recorded, endpoint named."""
    with stub_lfm2d({'error': {'message': 'boom', 'type': 'internal'}}, status=500) as url:
        with tempfile.TemporaryDirectory() as d:
            dec, rows, _ = run_hook('cd /tmp && echo hi', Path(d), {'LFM2D_URL': url})
    allowed = dec.get('hookSpecificOutput', {}).get('permissionDecision') == 'allow'
    v = rows[0]['lfm2d'] if rows else {}
    ok = allowed and v.get('error', '').startswith('http 500') and v.get('endpoint') == 'cascade'
    check('cascade http 500 fails open, endpoint named', ok, f'row={v}')


def test_long_commands_still_get_a_verdict(_c):
    """Regression: the first live run lost 4 of 14 rows to timeouts, and every
    loss was a long command. That is not merely lossy, it is BIASED — the
    dropped rows are the heredocs and quoted payloads where data-position
    content lives, i.e. exactly what the advisory phase exists to measure.

    Lengths chosen from the real failures: 1119, 1990 and 2840 characters."""
    bad = []
    for n in (500, 1200, 2000, 3000):
        cmd = ('git commit -m "wip" && echo ok && ' * 200)[:n]
        with tempfile.TemporaryDirectory() as d:
            _, rows, wall = run_hook(cmd, Path(d))
        v = rows[0]['lfm2d'] if rows else {}
        if not v.get('ok'):
            bad.append(f'len {n}: {v.get("error")} after {wall:.2f}s')
    check('long commands still get a verdict', not bad, '; '.join(bad))


def test_timeout_scales_with_command_length(_c):
    """A flat budget is what caused the bias. Short commands must still fail
    fast — that is what makes an outage detectable quickly."""
    short, long_ = timeout_probe(50), timeout_probe(3000)
    check('timeout scales with input length', short < 0.6 and long_ > 3.0,
          f'50 chars -> {short:.2f}s, 3000 chars -> {long_:.2f}s')


def timeout_probe(n: int) -> float:
    """Ask the hook module itself what timeout it would use, so this tests the
    real formula rather than a copy of it."""
    out = subprocess.run(
        [sys.executable, '-c',
         f'import importlib.util,sys;'
         f'spec=importlib.util.spec_from_file_location("h", "{HOOK}");'
         f'm=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);'
         f'print(m.timeout_for("x"*{n}))'],
        capture_output=True, text=True, timeout=30,
    )
    return float(out.stdout.strip())


def test_breaker_opens_after_repeated_failures(_c):
    """An lfm2d outage must stop taxing every Bash call. Without the breaker,
    a black-holed host costs the full timeout forever (measured 447ms)."""
    down = {'LFM2D_URL': 'http://192.0.2.1:8088', 'LFM2D_HOOK_TIMEOUT': '0.3',
            'LFM2D_BREAKER_THRESHOLD': '3', 'LFM2D_BREAKER_COOLDOWN': '60'}
    with tempfile.TemporaryDirectory() as d:
        cache = Path(d)
        walls, buckets = [], []
        for _ in range(5):
            _, rows, w = run_hook('ls -la /home', cache, down)
            walls.append(w)
            buckets.append(rows[0]['disagree'] if rows else 'NO ROW')

    opened = buckets[3:] == ['circuit_open', 'circuit_open']
    fast = max(walls[3:]) < min(walls[:3])
    check('breaker opens after repeated failures', opened and fast,
          f"buckets={buckets} walls={[f'{w*1000:.0f}ms' for w in walls]}")


def test_breaker_still_logs_while_open(_c):
    """A skipped call and a healthy quiet period must not look alike in the
    log, or the record overstates how much lfm2d actually saw."""
    down = {'LFM2D_URL': 'http://192.0.2.1:8088', 'LFM2D_HOOK_TIMEOUT': '0.3',
            'LFM2D_BREAKER_THRESHOLD': '2', 'LFM2D_BREAKER_COOLDOWN': '60'}
    with tempfile.TemporaryDirectory() as d:
        cache = Path(d)
        for _ in range(4):
            _, rows, _ = run_hook('ls -la /home', cache, down)
        last = rows[0] if rows else None
    ok = last and last['disagree'] == 'circuit_open' and last['lfm2d'].get('detail')
    check('breaker still logs a row while open', bool(ok), f'last={last}')


def test_breaker_never_changes_the_decision(_c):
    """The breaker is an optimization, not a policy. Even fully open, the
    regex verdict must be what it always was."""
    down = {'LFM2D_URL': 'http://192.0.2.1:8088', 'LFM2D_HOOK_TIMEOUT': '0.3',
            'LFM2D_BREAKER_THRESHOLD': '1', 'LFM2D_BREAKER_COOLDOWN': '60'}
    bad = []
    for cmd in ('ls -la /home', 'rm -rf ./build', 'git add -A'):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            off, _, _ = run_hook(cmd, Path(a), {'LFM2D_HOOK_MODE': 'off'})
            # Trip the breaker with a DIFFERENT, non-soft-blocked command.
            # Using `cmd` itself here would leave an approval in the
            # soft-block cache, and the second call would take the
            # approved-retry path instead of the one under test — which is
            # exactly how this test failed the first time it ran.
            run_hook('echo tripping-the-breaker', Path(b), down)
            open_dec, _, _ = run_hook(cmd, Path(b), down)  # breaker now open
        if off != open_dec:
            bad.append((cmd, off, open_dec))
    check('breaker never changes the decision', not bad, str(bad))


def test_breaker_closes_on_a_good_response(_c):
    """One success is sufficient evidence the service came back."""
    down = {'LFM2D_URL': 'http://192.0.2.1:8088', 'LFM2D_HOOK_TIMEOUT': '0.3',
            'LFM2D_BREAKER_THRESHOLD': '1', 'LFM2D_BREAKER_COOLDOWN': '0'}
    with tempfile.TemporaryDirectory() as d:
        cache = Path(d)
        run_hook('ls -la /home', cache, down)  # fail -> open (cooldown 0)
        # cooldown 0 means the next call is allowed through; point it at the
        # real daemon and the circuit must close.
        _, rows, _ = run_hook('ls -la /home', cache,
                              {'LFM2D_BREAKER_THRESHOLD': '1', 'LFM2D_BREAKER_COOLDOWN': '0'})
        state = json.loads((cache / 'claude-hooks' / 'lfm2d-breaker.json').read_text())
    ok = rows and rows[0]['lfm2d'].get('ok') and state['consecutive_failures'] == 0
    check('breaker closes on a good response', bool(ok), f'state={state} row={rows[0] if rows else None}')


TESTS = [
    test_advisory_never_changes_the_decision,
    test_every_bash_call_logs_one_row,
    test_log_row_carries_the_audit_pair,
    test_log_row_is_complete,
    test_disagreement_buckets_are_right,
    test_lfm2d_only_bucket_is_reachable,
    test_split_path_is_recorded_and_pipelines_cascade,
    test_non_bash_does_not_reach_lfm2d,
    test_approved_retry_does_not_reach_lfm2d,
    test_log_is_not_world_readable,
    test_live_latency_is_bounded,
    test_compound_command_takes_the_cascade_path,
    test_cascade_winner_is_the_severe_clause,
    test_quoted_payload_is_not_split,
    test_unreachable_lfm2d_fails_open_and_says_so,
    test_timeout_is_honored,
    test_http_error_fails_open,
    test_malformed_body_fails_open,
    test_unknown_label_vocabulary_is_loud,
    test_known_vocabulary_still_classifies,
    test_unknown_vocabulary_is_loud_on_the_cascade_path,
    test_cascade_http_error_fails_open,
    test_long_commands_still_get_a_verdict,
    test_timeout_scales_with_command_length,
    test_breaker_opens_after_repeated_failures,
    test_breaker_still_logs_while_open,
    test_breaker_never_changes_the_decision,
    test_breaker_closes_on_a_good_response,
]


def main() -> int:
    classifier = preflight()
    for t in TESTS:
        try:
            t(classifier)
        except Exception as e:
            check(t.__name__, False, f'{type(e).__name__}: {e}')
    print()
    if FAIL:
        print(f"FAILED: {len(FAIL)} of {len(PASS) + len(FAIL)}")
        for n in FAIL:
            print(f"  - {n}")
        return 1
    print(f"OK: {len(PASS)} tests passed")
    return 0


if __name__ == '__main__':
    sys.exit(main())
