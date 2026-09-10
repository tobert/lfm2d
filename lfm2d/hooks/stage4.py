#!/usr/bin/env python3
"""Stage 4 of the escalation cascade: static checks on classifier data.

    kaish plan -> static checks -> classifier -> STATIC CHECKS ON
    CLASSIFIER DATA -> adjudicator

Given the winning clause's structured plan facts, decide whether a raised
classifier verdict is dismissible from the plan ALONE. `echo restored`
scoring data-critical is dismissible because the verb is `echo` and
nothing redirects; `echo restored > /etc/shadow` is not, and the two
differ by a redirect and by nothing else in the rendered text. Reading
the text instead is the prose-reading failure `_command_facts` exists to
prevent (kaish_plan.py:91).

THE RATCHET
-----------
Stages 1-4 may only RAISE concern. Stage 5 (the adjudicator) is the
single release valve, and it is the only stage that can approve.

Stage 4 lowering a *classifier verdict* is not a violation of that: it
corrects a known-bad signal (the severity axis is corpus familiarity, and
`echo restored` at 0.605 data-critical is the axis misfiring), it does
not override a control. What must NEVER be lowered is a DENIAL from stage
2. That is why `guard_decision` is a required keyword argument rather
than something a caller may forget: a deny short-circuits to
not-dismissible, and any decision this module does not recognise is
treated as a deny, not an allow.

This keeps docs/integration.md invariant 5 intact -- an lfm2d score may
raise a prompt, never lower one or auto-decide. Nothing here approves
anything; dismissing a classifier verdict returns the statement to
whatever the earlier stages already said about it.

WHY AN ALLOWLIST IS SOUND HERE
------------------------------
A verb allowlist over TEXT is fragile -- `rm -rf` blocked and `rm -r`
sailing through is a recorded live failure. It is sound at stage 4 only
because stage 1 hands us structure: verb, argv words and redirects as the
parser resolved them, not a regex's reading of a rendered string. Every
match below is an exact comparison against one argv word.

Everything fails closed. An unlisted verb, an unreadable subcommand, a
redirect kind we have never seen, a verb spelled as a path: all survive
to the desk with a reason that NAMES what stopped them, so the report can
rank what is worth a ruling. A silent dismissal is the one outcome this
stage must not have.

POLICY, NOT MODEL
-----------------
The tables below are policy. Where they should ultimately live -- with
the agent, in kaijutsu, or as a shared table -- is Amy's open question
(~/exomemory/lfm2d/adjudicator-cascade.md). They live here for now
because the hook is the consumer and this file is diffable. Adding a verb
is a ruling, and test_stage4.py pins every entry.
"""
from collections import namedtuple

Review = namedtuple('Review', ('dismissible', 'reason'))

# A rule over the argv words that FOLLOW a verb (or its subcommand).
#
#   forbid       argv words that defeat the dismissal outright. Matched
#                exactly, and against the head of a `--flag=value` word.
#   allowed      when non-empty the rule is a STRICT allowlist: every
#                remaining word must be an allowed flag, or the value of
#                a value-taking allowed flag. Use this where an ordinary
#                OPERAND is a mutation -- `git branch foo` creates a
#                branch, so listing flags alone cannot carry the verb.
#   value_flags  allowed flags that consume the next word.
Rule = namedtuple('Rule', ('forbid', 'allowed', 'value_flags'))


def rule(forbid=(), allowed=(), value_flags=()):
    return Rule(frozenset(forbid), frozenset(allowed), frozenset(value_flags))


# No argv form of these touches the filesystem or another process, so an
# unredirected call is a read whatever its arguments say. `sort` and
# `find` are here with the flags that make them write: their ordinary
# form is a pure read and both appear in live traffic.
#
# Deliberately ABSENT, and each for a reason worth keeping:
#   sed, awk, perl, python, ruby, node   -- edit in place / write files
#   tee, dd, truncate, split, tar, gzip  -- write by construction
#   sudo, doas, env, xargs, nohup,
#   timeout, watch, command, ssh         -- named by the wrapper, and the
#                                           wrapped verb is invisible here
#   curl, wget                           -- -o/-O write
#   rm, cp, mv, ln, mkdir, touch,
#   chmod, chown, kill, pkill            -- the desk should see these
READ_ONLY_VERBS = {
    # shell builtins with no filesystem effect
    'echo': rule(),
    'printf': rule(),
    'pwd': rule(),
    'cd': rule(),
    'set': rule(),
    'shift': rule(),
    'true': rule(),
    'false': rule(),
    ':': rule(),
    'export': rule(),
    'unset': rule(),
    'jobs': rule(),
    'disown': rule(),
    'wait': rule(),
    'type': rule(),
    'sleep': rule(),
    # readers and filters
    'cat': rule(),
    'head': rule(),
    'tail': rule(),
    'wc': rule(),
    'nl': rule(),
    'rev': rule(),
    'fold': rule(),
    'column': rule(),
    'cut': rule(),
    'tr': rule(),
    'uniq': rule(),
    'grep': rule(),
    'egrep': rule(),
    'fgrep': rule(),
    'rg': rule(),
    'jq': rule(),
    'diff': rule(),
    'cmp': rule(),
    'base64': rule(),
    'od': rule(),
    'xxd': rule(),
    'strings': rule(),
    'md5sum': rule(),
    'sha256sum': rule(),
    'seq': rule(),
    # `sort -o FILE` and `--output=FILE` write; nothing else does
    'sort': rule(forbid=('-o', '--output')),
    # find's actions write, delete and execute; its predicates only read
    'find': rule(forbid=('-delete', '-exec', '-execdir', '-ok', '-okdir',
                         '-fprint', '-fprint0', '-fprintf', '-fls')),
    # inspection
    'ls': rule(),
    'stat': rule(),
    'file': rule(),
    'du': rule(),
    'df': rule(),
    'tree': rule(),
    'basename': rule(),
    'dirname': rule(),
    'realpath': rule(),
    'readlink': rule(),
    'which': rule(),
    'whereis': rule(),
    'printenv': rule(),
    'date': rule(),
    'uname': rule(),
    'hostname': rule(),
    'whoami': rule(),
    'id': rule(),
    'uptime': rule(),
    'ps': rule(),
    'pgrep': rule(),
    'free': rule(),
    'tty': rule(),
    'locale': rule(),
}

# Verbs whose subcommand decides whether they read. Keys are the leading
# non-flag argv words; the longest match wins, so a two-word path such as
# `rustup target list` can be listed without listing `rustup target`.
#
# git is here because it is ten of the twenty-five real data-critical
# winners in the live window, and because the severity head's git mass
# sits on history rewriting -- exactly the subcommands NOT listed.
SUBCOMMAND_VERBS = {
    'git': {
        ('status',): rule(),
        ('log',): rule(),
        ('show',): rule(),
        ('diff',): rule(),
        ('blame',): rule(),
        ('shortlog',): rule(),
        ('rev-parse',): rule(),
        ('rev-list',): rule(),
        ('describe',): rule(),
        ('ls-files',): rule(),
        ('ls-tree',): rule(),
        ('ls-remote',): rule(),
        ('cat-file',): rule(),
        ('merge-base',): rule(),
        ('name-rev',): rule(),
        ('diff-tree',): rule(),
        ('count-objects',): rule(),
        ('whatchanged',): rule(),
        ('version',): rule(),
        ('grep',): rule(),
        ('worktree', 'list'): rule(),
        ('stash', 'list'): rule(),
        ('stash', 'show'): rule(),
        ('remote', 'get-url'): rule(),
        ('remote', '-v'): rule(),
        ('config', '--get'): rule(),
        ('config', '--list'): rule(),
        # A bare operand CREATES a branch, so this one is a strict
        # allowlist: `-a`, `-r`, `--merged main` pass; `-D name` is
        # forbidden and `newbranch` is not an allowed word at all.
        ('branch',): rule(
            forbid=('-d', '-D', '--delete', '-m', '-M', '--move',
                    '-c', '-C', '--copy', '-f', '--force', '-u',
                    '--set-upstream-to', '--unset-upstream',
                    '--edit-description', '-t', '--track', '--no-track'),
            allowed=('-a', '--all', '-r', '--remotes', '-v', '-vv',
                     '--verbose', '-q', '--quiet', '--list',
                     '--show-current', '--merged', '--no-merged',
                     '--contains', '--no-contains', '--points-at',
                     '--sort', '--format', '--color', '--no-color',
                     '-i', '--ignore-case'),
            value_flags=('--merged', '--no-merged', '--contains',
                         '--no-contains', '--points-at', '--sort',
                         '--format', '--color')),
    },
    'rustup': {
        ('show',): rule(),
        ('which',): rule(),
        ('target', 'list'): rule(),
        ('toolchain', 'list'): rule(),
        ('component', 'list'): rule(),
    },
    'cargo': {
        ('tree',): rule(),
        ('metadata',): rule(),
    },
}

# Flags that come BEFORE a subcommand and consume the next word. Without
# these `git -C d1 status` reads `d1` as the subcommand.
GLOBAL_VALUE_FLAGS = {
    'git': frozenset({'-C', '-c', '--git-dir', '--work-tree',
                      '--namespace', '--exec-path', '--config-env'}),
    'rustup': frozenset({'--toolchain'}),
    'cargo': frozenset({'--manifest-path', '--color'}),
}

# The stage-2 guard's decision vocabulary, counted over the live advisory
# log: allow 52508, soft_deny 194, warn 160, deny 119
# (pre_command_advisory.py:654-666 emits exactly these four).
#
# A DENIAL is a control, and stage 4 does not touch controls -- only the
# adjudicator can release one. `soft_deny` is a denial with an escape
# hatch, so it is a control too: reading the vocabulary as just
# allow/deny would have let stage 4 lower 194 real soft denials.
#
# A `warn` is NOT a control. The statement proceeds carrying its warning,
# so the classifier verdict on a warn row is dismissible -- and the warn
# is untouched by that, because stage 4 reviews the classifier's verdict
# and nothing else.
GUARD_DENIALS = frozenset({'deny', 'soft_deny'})
GUARD_ELIGIBLE = frozenset({'allow', 'warn'})


def _writes_filesystem(redirect):
    """True when this redirect can put bytes on the filesystem.

    The discriminator is the KIND, never the target's spelling: an fd-dup
    (`2>&1`) carries its destination inside the kind and kaish emits a
    placeholder target for it, so a target-based check reads it as
    "writes a file named null" (kaish_plan.py nulls that target for the
    same reason). `/dev/null` is deliberately NOT special-cased -- a
    target's spelling is exactly what stage 4 must not reason about.
    """
    kind = redirect.get('kind') or ''
    if not kind:
        # A redirect we cannot name is a redirect we cannot clear.
        return True
    if '&' in kind:
        return False  # fd duplication: no filesystem target
    return '>' in kind


def _flag_head(word):
    """`--output=x` -> `--output`; anything else unchanged."""
    return word.split('=', 1)[0] if word.startswith('--') and '=' in word else word


def _check_args(args, r, label):
    """Apply `r` to the argv words `args`. Returns a reason on refusal,
    None when the words are clear."""
    for word in args:
        if _flag_head(word) in r.forbid:
            return f'forbidden_flag:{label} {_flag_head(word)}'
    if not r.allowed:
        return None
    # Strict allowlist: every word must be an allowed flag or the value of
    # a value-taking allowed flag.
    expect_value = False
    for word in args:
        if expect_value:
            expect_value = False
            continue
        head = _flag_head(word)
        if head not in r.allowed:
            if word.startswith('-'):
                return f'flag_not_allowed:{label} {head}'
            return f'operand_not_allowed:{label}'
        if head in r.value_flags and '=' not in word:
            expect_value = True
    return None


def _subcommand(verb, args, table):
    """Split `args` into (matched subcommand path, remaining words), or
    (None, reason) when the subcommand cannot be located or is not listed.

    Fails closed on an unknown leading flag: if we cannot say which word
    is the subcommand, we do not guess which verb we are looking at.
    """
    globals_ = GLOBAL_VALUE_FLAGS.get(verb, frozenset())
    i = 0
    while i < len(args):
        word = args[i]
        if word in globals_:
            i += 2  # flag plus its value
            continue
        if word.startswith('-'):
            if _flag_head(word) in globals_:
                i += 1  # --git-dir=x form, value attached
                continue
            return None, f'subcommand_unreadable:{verb}'
        break
    words = []
    for word in args[i:]:
        if word.startswith('-'):
            break
        words.append(word)
    if not words:
        return None, f'no_subcommand:{verb}'
    depth = max(len(k) for k in table)
    for n in range(min(len(words), depth), 0, -1):
        path = tuple(words[:n])
        if path in table:
            return path, args[i + n:]
    # Name only as many words as could possibly have matched a key. Beyond
    # that the words are OPERANDS -- a branch name, a path -- and this slug
    # is counted and logged: an operand in it fragments the histogram and
    # puts real names in a report.
    return None, f'subcommand_not_listed:{verb} {" ".join(words[:depth])}'


def review(facts, *, guard_decision):
    """Can a raised classifier verdict on this clause be dismissed from
    the plan alone?

    `facts` is one clause's `{name, args, redirects}` as
    kaish_plan._command_facts produced it. `guard_decision` is what stage
    2 said ('allow' / 'deny' / ...) and is required: stage 4 may dismiss a
    classifier verdict, never a stage-2 denial.

    Returns Review(dismissible: bool, reason: str). The reason is a stable
    slug naming what decided, dismissal or not, so a report can count and
    rank them. Never raises: a shape we cannot read is not dismissible.
    """
    # The ratchet, checked before anything else.
    if guard_decision in GUARD_DENIALS:
        return Review(False, f'guard_denied:{guard_decision}')
    if guard_decision not in GUARD_ELIGIBLE:
        # A guard verdict this module has never heard of fails closed and
        # shows up in the report as a countable reason. A new decision
        # name must not become a silent dismissal.
        return Review(False, f'guard_unknown:{guard_decision}')

    name = (facts or {}).get('name')
    if not name:
        return Review(False, 'no_verb')
    if '/' in name:
        # `/bin/cat` may well be cat. Stage 4 cannot know that, and
        # `./contrib/install-codex-app-server-systemd.sh` is why it matters.
        return Review(False, f'verb_is_a_path:{name}')

    for redirect in (facts.get('redirects') or []):
        if _writes_filesystem(redirect):
            return Review(False, f'write_redirect:{redirect.get("kind") or "?"}')

    args = list(facts.get('args') or [])

    if name in SUBCOMMAND_VERBS:
        table = SUBCOMMAND_VERBS[name]
        path, rest = _subcommand(name, args, table)
        if path is None:
            return Review(False, rest)  # rest carries the reason
        label = f'{name} {" ".join(path)}'
        refusal = _check_args(rest, table[path], label)
        if refusal:
            return Review(False, refusal)
        return Review(True, f'read_subcommand:{label}')

    if name in READ_ONLY_VERBS:
        refusal = _check_args(args, READ_ONLY_VERBS[name], name)
        if refusal:
            return Review(False, refusal)
        return Review(True, f'read_only_verb:{name}')

    return Review(False, f'verb_not_listed:{name}')
