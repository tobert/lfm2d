#!/usr/bin/env python3
"""Plan-first clause extraction for the advisory hook (v10 slice 1).

Runs `kaish --plan-file -` over the raw Bash command and renders each
simple command in the plan as one clause: `name` + canonically-quoted
args + redirect operators/targets, heredoc bodies stripped (they live in
the plan's `heredocs[]`, never in argv) and tagged by kind. This is the
unit v10 trains on (training/v10/PLAN.md, "the unit is the simple
command as `kaish --plan` renders it"); the hook falls back to
`clause_split.py` on any failure here and RECORDS which path it took, so
the live fallback rate is measurable from the advisory log.

Behaviour change from the clause_split path, on purpose: pipelines ARE
split into their member commands. The old no-pipeline-split rule guarded
a v8-era misreading (`grep 'git push --force'` alone scored 0.892 dc);
v10's direction is the opposite — the simple command is the unit, the
noisy middle is training data, and the advisory log needs rows in
exactly that shape. The fallback path keeps the old behaviour.

Latency: ~12 ms p50 per invocation measured 2026-08-24 (kaish 0.16.0
b27ea4dd, 20 runs), against ~65-145 ms/clause scoring cost downstream.

Never raises: like the splitter, a plan failure must not be able to
break the guard. Every failure names itself so `plan_error` rows in the
log are countable by cause.
"""
import json
import os
import subprocess

KAISH_BIN = os.environ.get('LFM2D_KAISH_BIN', 'kaish')
PLAN_TIMEOUT_S = float(os.environ.get('LFM2D_KAISH_PLAN_TIMEOUT', '2.0'))

_KAISH_VERSION = None


def kaish_version() -> str:
    """`kaish --version` output, memoized per process (one hook invocation
    is one process, so this costs one ~6 ms subprocess per Bash call).

    On every plan-path row because kaish's canonical rendering IS the
    scored text, and it changes across kaish releases (0.17 will re-render
    `-0`-family argv words, per kaish-lead 2026-08-24) — a row that
    doesn't name the renderer can't be windowed against the floor
    baseline (kaish 0.16.0 b27ea4dd)."""
    global _KAISH_VERSION
    if _KAISH_VERSION is None:
        try:
            proc = subprocess.run([KAISH_BIN, '--version'], capture_output=True,
                                  text=True, timeout=PLAN_TIMEOUT_S)
            _KAISH_VERSION = proc.stdout.strip() or f'exit {proc.returncode}'
        except Exception as e:
            _KAISH_VERSION = f'unknown ({type(e).__name__})'
    return _KAISH_VERSION

# For heredoc `kind` tagging only — routing taxonomy from
# training/v10/PLAN.md section 6 (68% of real heredocs feed an
# interpreter). This tags log rows; it never decides anything.
INTERPRETERS = {
    'python', 'python2', 'python3', 'ruby', 'perl', 'node', 'nodejs',
    'php', 'lua', 'Rscript', 'irb', 'deno', 'bun',
}
MESSAGE_COMMANDS = {'git', 'gh'}
PAGER_COMMANDS = {'cat', 'tee'}


def _heredoc_kind(name: str, redirect_kinds: list) -> str:
    writes_file = any(k in ('>', '>>') for k in redirect_kinds)
    if name in INTERPRETERS or (name.endswith('/python3') or name.endswith('/python')):
        return 'interpreter'
    if name in MESSAGE_COMMANDS:
        return 'message'
    if writes_file:
        return 'file'
    if name in PAGER_COMMANDS:
        return 'stdout'
    return 'other'


def _render_command(c: dict):
    """One simple command as clause text, or None when the plan's args
    carry a shape we don't know how to render faithfully (no `plain`
    value) — the caller then falls back to the statement's own rendering
    rather than inventing text the parser didn't produce."""
    parts = [c.get('name') or '']
    if not parts[0]:
        return None
    for a in c.get('args') or []:
        if not isinstance(a, dict) or 'plain' not in a:
            return None
        parts.append(a['plain'])
    for r in c.get('redirects') or []:
        kind = r.get('kind') or ''
        target = (r.get('target') or {}).get('plain')
        if kind.startswith('<<'):
            # The heredoc operator + delimiter stay (they are shape); the
            # body never appears — it isn't in argv to begin with.
            parts.append(f'{kind}{target or ""}')
        elif target in (None, '', 'null'):
            parts.append(kind)
        else:
            parts.append(f'{kind} {target}')
    return ' '.join(parts)


def plan_clauses(cmd: str) -> dict:
    """Plan `cmd` and return clause rows. ALWAYS has 'ok'.

    ok=True  -> 'clauses': non-empty list of {'text', 'stmt_index',
                'heredoc': {'delimiter','literal','kind'} | None,
                'stmt_fallback': bool}, 'statement_count': int.
    ok=False -> 'error' in {'empty', 'kaish_missing', 'timeout',
                'parse', 'bad_json', 'no_clauses', <exception name>},
                'detail': str.  Caller falls back to clause_split.
    """
    if not cmd.strip():
        return {'ok': False, 'error': 'empty', 'detail': 'blank command'}
    try:
        proc = subprocess.run(
            [KAISH_BIN, '--plan-file', '-'],
            input=cmd,
            capture_output=True,
            text=True,
            timeout=PLAN_TIMEOUT_S,
        )
    except FileNotFoundError:
        return {'ok': False, 'error': 'kaish_missing', 'detail': f'{KAISH_BIN} not on PATH'}
    except subprocess.TimeoutExpired:
        return {'ok': False, 'error': 'timeout', 'detail': f'plan exceeded {PLAN_TIMEOUT_S}s'}
    except Exception as e:  # the plan path must never break the guard
        return {'ok': False, 'error': type(e).__name__, 'detail': str(e)[:200]}

    try:
        doc = json.loads(proc.stdout)
    except Exception:
        return {'ok': False, 'error': 'bad_json',
                'detail': f'exit {proc.returncode}: {proc.stdout[:120]!r} {proc.stderr[:80]!r}'}

    if proc.returncode != 0 or 'errors' in doc:
        msgs = '; '.join(e.get('message', '?') for e in doc.get('errors') or [])
        return {'ok': False, 'error': 'parse', 'detail': msgs[:300] or f'exit {proc.returncode}'}

    clauses = []
    statements = doc.get('statements') or []
    for stmt in statements:
        plan = stmt.get('plan') or {}
        commands = plan.get('commands') or []
        if not commands:
            # A pure assignment / no-command statement still gets scored
            # as the parser rendered it — parity with the old path, and
            # `X=$(cmd)` is not this case (its commands[] is populated).
            rendered = plan.get('rendered')
            if rendered:
                clauses.append({'text': rendered, 'stmt_index': stmt.get('index'),
                                'heredoc': None, 'stmt_fallback': False})
            continue
        rendered_cmds = [_render_command(c) for c in commands]
        if any(r is None for r in rendered_cmds):
            # An arg shape we can't render faithfully: score the whole
            # statement as kaish rendered it rather than a lossy guess.
            clauses.append({'text': plan.get('rendered') or '', 'stmt_index': stmt.get('index'),
                            'heredoc': None, 'stmt_fallback': True})
            continue
        for c, text in zip(commands, rendered_cmds):
            heredocs = c.get('heredocs') or []
            tag = None
            if heredocs:
                h = heredocs[0]
                tag = {
                    'delimiter': h.get('delimiter'),
                    'literal': bool(h.get('literal')),
                    'kind': _heredoc_kind(c.get('name') or '',
                                          [r.get('kind') for r in c.get('redirects') or []]),
                }
            clauses.append({'text': text, 'stmt_index': stmt.get('index'),
                            'heredoc': tag, 'stmt_fallback': False})

    clauses = [c for c in clauses if c['text'].strip()]
    if not clauses:
        return {'ok': False, 'error': 'no_clauses', 'detail': f'{len(statements)} statements, none rendered'}
    return {'ok': True, 'clauses': clauses, 'statement_count': len(statements),
            'kaish_version': kaish_version()}
