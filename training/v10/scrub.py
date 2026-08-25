#!/usr/bin/env python3
"""Structural identifier scrub for real clauses (ruling (b), 2026-08-24).

Operates on the parser's argv — kaish's plan hands us `name`, each arg as
one `plain` word (quoting intact) and typed redirect targets — so paths,
hosts and tokens are replaced as WORDS, not guessed by regex over a line.
The rendering is the serving renderer's (`kaish_plan._render_command`),
so a scrubbed clause is spelled exactly the way the hook would score it.

What changes, and what deliberately does not:
  - `/home/<user>`, `/Users/<user>`            -> `~`
  - `/tmp/claude-<n>/<slug>/<uuid>/scratchpad` -> `/tmp/scratch`
  - a path component not on the GENERIC list -> `d<N>` (dir) / `f<N>.<ext>`
    (file), numbered by first appearance within the clause; extensions,
    globs and well-known file names survive because they are the form.
  - `/dev/*`, `/etc/*`, `/proc/*`, `/sys/*`, `/var/log/*`, `/usr/*`,
    `/boot/*`, `/bin/*` stay whole: those are the paths severity rides on.
  - URL hosts not on KNOWN_HOSTS -> `host.example`; `*.ts.net`/`.local`/
    `.lan` hostnames likewise; emails -> `user@example.com`.
  - credential-looking tokens -> `<redacted>` (the rubric's own spelling).
  - the VALUE of a message flag (`-m`, `--message`, `--title`, `--body`)
    -> a placeholder in the same quote style; the flag is the form.
  - `$VAR`, `${VAR}`, `$(...)`, flags, numbers, hashes, search patterns,
    the command name's basename: untouched.

Pure functions; tests in test_scrub.py.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / 'lfm2d' / 'hooks'))
from kaish_plan import _render_command  # noqa: E402  (the serving renderer)

GENERIC_DIRS = {
    '.', '..', '~', 'src', 'crates', 'tests', 'test', 'docs', 'doc', 'target', 'debug', 'release',
    'bin', 'lib', 'libs', 'include', 'etc', 'var', 'log', 'logs', 'dev', 'proc', 'sys', 'usr',
    'opt', 'tmp', 'home', 'mnt', 'run', 'srv', 'root', 'cache', '.cache', '.config', '.local',
    'share', 'state', 'node_modules', '.git', 'hooks', 'scripts', 'examples', 'benches',
    'assets', 'static', 'templates', 'config', 'configs', 'build', 'dist', 'out', 'pkg',
    'vendor', 'content', 'en', 'migrations', 'api', 'cli', 'kernel', 'server', 'client', 'app',
    'core', 'util', 'utils', 'common', 'models', 'model', 'data', 'training', 'deploy', 'k8s',
    'fixtures', 'snapshots', 'wt', 'main', 'origin', 'HEAD', 'refs', 'heads', 'tags', 'remotes',
    'scratch', 'scratchpad', 'memory', 'issues', 'daily', 'notes', 'tools', 'internal', 'public',
    'private', 'proto', 'schema', 'sql', 'db', 'web', 'ui', 'views', 'view', 'components',
    'services', 'service', 'handlers', 'routes', 'controllers', 'plugins', 'plugin', 'ext',
    'extension', 'extensions', 'rc', 'create', 'skills', '.claude', 'projects', 'agents',
    'v1', 'v2', 'v3', 'health', 'metrics', 'classify', 'cascade', 'route', 'embed', 'raw', 'blob',
}
GENERIC_FILES = {
    'Cargo.toml', 'Cargo.lock', 'package.json', 'package-lock.json', 'pnpm-lock.yaml', 'yarn.lock',
    'README.md', 'README', 'CHANGELOG.md', 'CHANGELOG', 'LICENSE', 'LICENSE.md', 'Makefile',
    'Dockerfile', 'docker-compose.yml', '.gitignore', '.gitattributes', '.env', 'lib.rs', 'main.rs',
    'mod.rs', 'build.rs', 'error.rs', 'errors.rs', 'types.rs', 'config.rs', 'cli.rs', 'kernel.rs',
    'parser.rs', 'lexer.rs', 'tests.rs', '__init__.py', 'setup.py', 'pyproject.toml',
    'requirements.txt', 'go.mod', 'go.sum', 'main.go', 'index.js', 'index.ts', 'main.py', 'app.py',
    'CLAUDE.md', 'AGENTS.md', 'MEMORY.md', 'signoff.md', 'PLAN.md', 'issues.md', 'null', 'stdin',
    'stdout', 'stderr', 'config.json', 'config.toml', 'config.yaml', 'settings.json',
}
KEEP_WHOLE_PREFIXES = ('/dev/', '/etc/', '/proc/', '/sys/', '/var/log/', '/usr/', '/boot/', '/bin/',
                       '/sbin/', '/lib/', '/lib64/', '/var/lib/', '/var/run/', '/run/', '/opt/', '/snap/')
KNOWN_HOSTS = {
    'localhost', '127.0.0.1', '0.0.0.0', '::1', 'github.com', 'api.github.com',
    'raw.githubusercontent.com', 'huggingface.co', 'hf.co', 'crates.io', 'index.crates.io',
    'static.crates.io', 'docs.rs', 'pypi.org', 'files.pythonhosted.org', 'npmjs.com',
    'registry.npmjs.org', 'example.com', 'host.example', 'kubernetes.default.svc',
}
MESSAGE_FLAGS = {'-m', '--message', '--title', '--body', '-t', '-b', '--description', '--notes'}

HOME_RE = re.compile(r'^(/home|/Users)/[^/]+')
SCRATCH_RE = re.compile(r'^/tmp/claude-\d+/[^/]+/[^/]+/scratchpad')
UUID_RE = re.compile(r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$')
EMAIL_RE = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
URL_RE = re.compile(r'(?P<scheme>[a-z][a-z0-9+.-]*://)(?P<host>[^/\s:\'"]+)(?P<port>:\d+)?(?P<path>/[^\s\'"]*)?')
PRIVATE_HOST_RE = re.compile(r'\b[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.(ts\.net|local|lan|internal|home\.arpa)\b')
SECRET_RE = re.compile(
    r'\b(?:sk-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]{20,}|gho_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}'
    r'|hf_[A-Za-z0-9]{20,}|xox[abp]-[A-Za-z0-9-]{20,}|AKIA[A-Z0-9]{16}|eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,}'
    r'|[A-Za-z0-9+]{40,}={0,2})\b')  # no '/' in the generic arm: a long path is not a token
DOLLAR_RE = re.compile(r'\$\{[^}]*\}|\$\([^)]*\)|\$[A-Za-z_][A-Za-z0-9_]*|\$[0-9@#?*!-]')
# Amy's handles: a bare username is an identifier wherever it appears
# (a grep pattern, a slug inside a program). Public handles, but the
# scrubbed set should not be greppable for them.
USERNAME_RE = re.compile(r'\b(?:atobey|tobert)\b', re.I)
# Inside program/pattern text (a sed expression, a `sh -c` body) only
# these regex-level scrubs apply — never component numbering.
HOME_ANYWHERE_RE = re.compile(r'(?:/home|/Users)/[^/\s\'"]+')
SCRATCH_ANYWHERE_RE = re.compile(r'/tmp/claude-\d+/[^/\s]+/[^/\s]+/scratchpad')
SED_PROGRAM_RE = re.compile(r'^[sy][^A-Za-z0-9\s]|^\d*(,\d*)?[dpi]$|^/[^/]*/[dpis]')
GLOB_CHARS = set('*?[')


class Numbering:
    """Per-clause placeholder numbering: the same original maps to the
    same placeholder within one clause, so `cp a.rs b.rs` keeps two
    distinct files and `diff x x.bak` keeps one stem."""

    def __init__(self):
        self.dirs, self.files = {}, {}

    def dir(self, comp):
        if comp not in self.dirs:
            self.dirs[comp] = f'd{len(self.dirs) + 1}'
        return self.dirs[comp]

    def file(self, stem):
        if stem not in self.files:
            self.files[stem] = f'f{len(self.files) + 1}'
        return self.files[stem]


def _split_quotes(word):
    """('quote', inner, 'quote') if the word is wholly quoted, else ('', word, '')."""
    if len(word) >= 2 and word[0] == word[-1] and word[0] in ('"', "'"):
        return word[0], word[1:-1], word[-1]
    return '', word, ''


def _scrub_component(comp, is_last, num):
    if not comp or comp in GENERIC_DIRS or comp in GENERIC_FILES or DOLLAR_RE.search(comp):
        return comp
    if UUID_RE.match(comp) or re.fullmatch(r'[0-9a-f]{12,}', comp):
        return 'id'
    if is_last:
        if any(ch in GLOB_CHARS for ch in comp):
            return comp  # `*.rs`, `*.log`: the glob is the form
        stem, dot, ext = comp.rpartition('.')
        if dot and stem and not stem.startswith('.'):
            return f'{num.file(stem)}.{ext}'
        return num.file(comp)
    if any(ch in GLOB_CHARS for ch in comp):
        return comp
    return num.dir(comp)


def scrub_path(path, num):
    """Scrub one path-like string. Severity-carrying system prefixes stay
    whole; home and scratchpad prefixes collapse; other components are
    numbered unless generic."""
    if path.startswith(KEEP_WHOLE_PREFIXES) or path == '/dev/null':
        return path
    path = HOME_RE.sub('~', path)
    path = SCRATCH_RE.sub('/tmp/scratch', path)
    if path.startswith(KEEP_WHOLE_PREFIXES):
        return path
    trailing = path.endswith('/')
    comps = path.split('/')
    out = []
    for i, comp in enumerate(comps):
        is_last = i == len(comps) - 1 and not trailing
        out.append(_scrub_component(comp, is_last, num))
    return '/'.join(out)


def _scrub_url(m, num):
    host = m.group('host')
    if host not in KNOWN_HOSTS and not re.fullmatch(r'\d+\.\d+\.\d+\.\d+', host):
        host = 'host.example'
    path = m.group('path') or ''
    if path and host == 'host.example':
        path = scrub_path(path, num)
    return f"{m.group('scheme')}{host}{m.group('port') or ''}{path}"


def _looks_like_path(s):
    return '/' in s or s.startswith('~') or s.startswith('./') or s.startswith('../')


def _is_program_text(inner, quoted):
    """A quoted word that is a program or pattern, not a path: whitespace,
    a newline, a backslash, a `;`, or sed's own syntax."""
    return quoted and (bool(re.search(r'[\s\\;]', inner)) or bool(SED_PROGRAM_RE.match(inner)))


def scrub_text(inner, num, quoted=False):
    """Scrub identifiers inside one word's text (quotes already stripped).
    Order matters: secrets and emails first (they may sit in URLs), then
    URLs, hostnames, usernames; then either the word-as-path (component
    numbering) or, for program text, regex-level home/scratchpad only."""
    if DOLLAR_RE.fullmatch(inner):
        return inner
    inner = SECRET_RE.sub('<redacted>', inner)
    inner = EMAIL_RE.sub('user@example.com', inner)
    inner = URL_RE.sub(lambda m: _scrub_url(m, num), inner)
    inner = PRIVATE_HOST_RE.sub('host.example', inner)
    inner = USERNAME_RE.sub('user', inner)
    if _is_program_text(inner, quoted):
        inner = HOME_ANYWHERE_RE.sub('~', inner)
        return SCRATCH_ANYWHERE_RE.sub('/tmp/scratch', inner)
    if '://' not in inner and _looks_like_path(inner) and not inner.startswith('-'):
        # a bare path word; `key=value` and `--opt=value` scrub the value
        return scrub_path(inner, num)
    return inner


def scrub_word(word, num, after_message_flag=False):
    """Scrub one argv word. A flag's `=value` is scrubbed as a value; a
    message flag's value becomes a placeholder in its own quote style."""
    q, inner, qe = _split_quotes(word)
    if after_message_flag:
        return f'{q or ""}msg{qe or ""}' if q else 'msg'
    if inner.startswith('-') and '=' in inner:
        k, _, v = inner.partition('=')
        vq, vinner, vqe = _split_quotes(v)
        if k in MESSAGE_FLAGS:
            return f'{q}{k}={vq}msg{vqe}{qe}'
        return f'{q}{k}={vq}{scrub_text(vinner, num, quoted=bool(q or vq))}{vqe}{qe}'
    if inner.startswith('-') and not _looks_like_path(inner):
        return word  # a flag; `-o-`, `-F-`, `-1`, `--all`
    if not inner.startswith('-') and '=' in inner and not _looks_like_path(inner.partition('=')[0]) \
            and not _is_program_text(inner, bool(q)):
        k, _, v = inner.partition('=')  # dd-style `of=/dev/sda`, `key=value`
        vq, vinner, vqe = _split_quotes(v)
        return f'{q}{k}={vq}{scrub_text(vinner, num, quoted=bool(q or vq))}{vqe}{qe}'
    return f'{q}{scrub_text(inner, num, quoted=bool(q))}{qe}'


def scrub_command(c):
    """Scrub one plan command dict (name/args/redirects/heredocs) and
    render it with the serving renderer. Returns None when the renderer
    would (an arg shape without a `plain`)."""
    num = Numbering()
    name = c.get('name') or ''
    if _looks_like_path(name):
        d, _, base = name.rpartition('/')
        name = f'{scrub_path(d, num)}/{base}' if d else name
    args = []
    prev_flag = None
    for a in c.get('args') or []:
        if not isinstance(a, dict) or not isinstance(a.get('plain'), str):
            return None
        w = a['plain']
        args.append({'plain': scrub_word(w, num, after_message_flag=prev_flag in MESSAGE_FLAGS)})
        prev_flag = w if w.startswith('-') else None
    redirects = []
    for r in c.get('redirects') or []:
        target = (r.get('target') or {}).get('plain')
        nr = dict(r)
        if isinstance(target, str) and not (r.get('kind') or '').startswith('<<') and '&' not in (r.get('kind') or ''):
            nr['target'] = {'plain': scrub_word(target, num)}
        redirects.append(nr)
    return _render_command({**c, 'name': name, 'args': args, 'redirects': redirects})
