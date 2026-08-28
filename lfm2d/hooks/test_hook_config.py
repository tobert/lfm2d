#!/usr/bin/env python3
"""Tests for the hook's deploy surface: the LFM2D_URL default and
install.sh on a machine that has never had a hook.

Run:  python3 lfm2d/hooks/test_hook_config.py

Plain script, no pytest (none on this machine). Exits non-zero on failure.

WHY THESE CASES
---------------
Both came out of planning a second, independent install (2026-08-28):

- The URL default was a specific host on one tailnet. A copy of the hook
  on any other machine either reached that host (commands leaving the
  machine, silently — advisory mode fails open and the breaker hides an
  unreachable default) or timed out on every call. The only safe default
  for a sidecar is loopback; any other endpoint is that machine's
  settings, not the code's.
- install.sh required the dotfiles baseline hook to exist for EVERY
  subcommand (even `status`) and refused to create a PreToolUse entry
  when settings had none. Both were right for the machine it was written
  on — the swap was the point — and wrong for a fresh one. `bootstrap`
  is the fresh-machine path; `install` keeps its parity gate.

These run install.sh with HOME pointed at a temp dir, so the real
~/.claude/settings.json is never touched.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
HOOK = HERE / 'pre_command_advisory.py'
LIVE = HERE / 'test_advisory_live.py'
INSTALL = HERE / 'install.sh'

FAILS = []


def check(name, cond, detail=''):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail and not cond else ''))
    if not cond:
        FAILS.append(name)


def module_default_url(path: Path) -> str:
    """Import the module in a subprocess with LFM2D_URL unset and print
    what it resolved. A subprocess because the live test module and the
    hook both read the env at import time."""
    env = {k: v for k, v in os.environ.items() if k != 'LFM2D_URL'}
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('m', {str(path)!r})\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "print(m.LFM2D_URL)\n"
    )
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env, timeout=20)
    if r.returncode != 0:
        return f'<import failed: {r.stderr.strip()[-200:]}>'
    return r.stdout.strip()


def run_install(home: Path, *args, env_extra=None):
    env = dict(os.environ)
    env['HOME'] = str(home)
    env.pop('XDG_CACHE_HOME', None)
    env.pop('LFM2D_URL', None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(['bash', str(INSTALL), *args], capture_output=True, text=True, env=env, timeout=30)


def advisory_entries(settings: dict):
    out = []
    for group in settings.get('hooks', {}).get('PreToolUse', []):
        for h in group.get('hooks', []):
            if 'pre_command_advisory.py' in h.get('command', ''):
                out.append((group, h))
    return out


# ---------------------------------------------------------------- URL default

def test_hook_default_url_is_loopback():
    url = module_default_url(HOOK)
    check('hook LFM2D_URL default is loopback', url.startswith('http://127.0.0.1:'), url)


def test_live_suite_default_url_is_loopback():
    url = module_default_url(LIVE)
    check('test_advisory_live LFM2D_URL default is loopback', url.startswith('http://127.0.0.1:'), url)


def test_env_override_still_wins():
    env = dict(os.environ, LFM2D_URL='http://example.invalid:1')
    code = (
        "import importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('m', {str(HOOK)!r})\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "print(m.LFM2D_URL)\n"
    )
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env, timeout=20)
    check('LFM2D_URL env overrides the default', r.stdout.strip() == 'http://example.invalid:1', r.stdout + r.stderr)


# ---------------------------------------------------------- install.sh fresh

def test_status_works_without_dotfiles_or_settings():
    with tempfile.TemporaryDirectory() as d:
        r = run_install(Path(d), 'status')
        check('status exits 0 on a bare HOME', r.returncode == 0, r.stderr.strip()[-200:])
        check('status reports no hook', 'none' in r.stdout.lower(), r.stdout)


def test_bootstrap_creates_settings_from_nothing():
    with tempfile.TemporaryDirectory() as d:
        home = Path(d)
        r = run_install(home, 'bootstrap')
        check('bootstrap exits 0 with no settings.json', r.returncode == 0, r.stderr.strip()[-300:])
        s_path = home / '.claude' / 'settings.json'
        check('bootstrap wrote settings.json', s_path.exists())
        if not s_path.exists():
            return
        s = json.loads(s_path.read_text())
        entries = advisory_entries(s)
        check('exactly one advisory entry', len(entries) == 1, str(entries))
        if entries:
            group, h = entries[0]
            check('entry matches Bash', group.get('matcher') == 'Bash', str(group))
            check('entry has a timeout', isinstance(h.get('timeout'), int), str(h))
            check('entry is type command', h.get('type') == 'command', str(h))
            check('command carries loopback LFM2D_URL', 'LFM2D_URL=http://127.0.0.1:8088' in h['command'], h['command'])
            check('command uses python3 and an absolute hook path',
                  h['command'].split('LFM2D_URL=http://127.0.0.1:8088 ', 1)[-1].startswith(f'python3 {HOOK}'),
                  h['command'])
        # status sees what bootstrap wrote
        r2 = run_install(home, 'status')
        check('status after bootstrap reports ADVISORY', 'ADVISORY' in r2.stdout, r2.stdout)


def test_bootstrap_preserves_existing_settings():
    with tempfile.TemporaryDirectory() as d:
        home = Path(d)
        (home / '.claude').mkdir()
        existing = {
            'permissions': {'defaultMode': 'auto'},
            'model': 'fable',
            'hooks': {
                'PreToolUse': [
                    {'matcher': 'Edit', 'hooks': [{'type': 'command', 'command': 'other-tool hook'}]}
                ],
                'SessionStart': [
                    {'hooks': [{'type': 'command', 'command': 'other-tool hook claude'}]}
                ],
            },
        }
        (home / '.claude' / 'settings.json').write_text(json.dumps(existing, indent=2) + '\n')
        r = run_install(home, 'bootstrap')
        check('bootstrap exits 0 with unrelated hooks present', r.returncode == 0, r.stderr.strip()[-300:])
        s = json.loads((home / '.claude' / 'settings.json').read_text())
        check('permissions preserved', s.get('permissions') == existing['permissions'])
        check('model preserved', s.get('model') == 'fable')
        check('other PreToolUse group preserved', existing['hooks']['PreToolUse'][0] in s['hooks']['PreToolUse'])
        check('SessionStart preserved', s['hooks'].get('SessionStart') == existing['hooks']['SessionStart'])
        check('one advisory entry added', len(advisory_entries(s)) == 1)
        backups = list((home / '.claude').glob('settings.json.bak-*'))
        check('backup taken before editing an existing file', len(backups) == 1, str(backups))


def test_bootstrap_refuses_when_hook_already_wired():
    with tempfile.TemporaryDirectory() as d:
        home = Path(d)
        r1 = run_install(home, 'bootstrap')
        s_path = home / '.claude' / 'settings.json'
        before = s_path.read_text()
        r2 = run_install(home, 'bootstrap')
        check('second bootstrap is refused (non-zero)', r2.returncode != 0, r2.stdout + r2.stderr)
        check('second bootstrap leaves settings untouched', s_path.read_text() == before)
        # a baseline dotfiles-style entry also counts as "already wired": that
        # machine wants `install` (parity-gated swap), not bootstrap.
        (home / '.claude').mkdir(exist_ok=True)
        s_path.write_text(json.dumps({'hooks': {'PreToolUse': [{'matcher': 'Bash', 'hooks': [
            {'type': 'command', 'command': 'python ~/.claude/hooks/pre-command.py'}]}]}}) + '\n')
        r3 = run_install(home, 'bootstrap')
        check('bootstrap refuses when a baseline hook exists', r3.returncode != 0, r3.stdout + r3.stderr)
        check('refusal names install as the path', 'install' in (r3.stdout + r3.stderr))


def test_bootstrap_takes_url_from_env():
    with tempfile.TemporaryDirectory() as d:
        home = Path(d)
        r = run_install(home, 'bootstrap', env_extra={'LFM2D_URL': 'http://sidecar.example:8088'})
        check('bootstrap with LFM2D_URL exits 0', r.returncode == 0, r.stderr.strip()[-300:])
        s = json.loads((home / '.claude' / 'settings.json').read_text())
        entries = advisory_entries(s)
        check('command carries the env URL', bool(entries) and 'LFM2D_URL=http://sidecar.example:8088 ' in entries[0][1]['command'],
              entries[0][1]['command'] if entries else 'no entry')


def test_bootstrap_rejects_invalid_json():
    with tempfile.TemporaryDirectory() as d:
        home = Path(d)
        (home / '.claude').mkdir()
        (home / '.claude' / 'settings.json').write_text('{ not json')
        r = run_install(home, 'bootstrap')
        check('bootstrap refuses invalid settings.json', r.returncode != 0)
        check('invalid file left as-is', (home / '.claude' / 'settings.json').read_text() == '{ not json')


def test_install_still_requires_baseline():
    """The parity-gated swap must not silently become bootstrap."""
    with tempfile.TemporaryDirectory() as d:
        home = Path(d)
        r = run_install(home, 'install')
        check('install without dotfiles baseline is refused', r.returncode != 0, r.stdout + r.stderr)
        check('install refusal names the missing baseline', 'dotfiles hook missing' in (r.stdout + r.stderr))


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for t in tests:
        print(t.__name__)
        try:
            t()
        except Exception as e:  # a raise is a failure with a name, not a crash
            check(t.__name__, False, f'{type(e).__name__}: {e}')
    print()
    if FAILS:
        print(f'FAILED: {len(FAILS)}')
        for f in FAILS:
            print(f'  - {f}')
        sys.exit(1)
    print('all passed')
