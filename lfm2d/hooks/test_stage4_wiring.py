#!/usr/bin/env python3
"""Tests for the hook's stage-4 block — pre_command_advisory.stage4_block.

Run:  python3 lfm2d/hooks/test_stage4_wiring.py

Plain script, no pytest (none on this machine). Exits non-zero on failure.

test_stage4.py pins the POLICY (what is dismissible). This file pins the
WIRING: which rows stage 4 is asked about, what it records when it cannot
answer, and that it is off unless the flag says otherwise.

WHY THESE CASES
---------------
The wiring's failure modes are all silent ones, so every case here is a
row that must carry a REASON rather than nothing:

- a fallback-path row has no plan facts, and stage 4 saying nothing about
  it must not look like a dismissal (case: fallback_path);
- a classify_batch row has no winner by design, and inventing one
  client-side is the thing lfm2d_classify_batch refuses to do (no_winner);
- a checkpoint speaking none of SEVERE_LABELS must bucket as
  vocab_mismatch, never as "stage 4 cleared it" -- the vocabulary has
  changed wholesale between checkpoints and will again;
- a winner_index that does not index the sent clauses is a real bug in
  the alignment, and must be loud rather than clamped (winner_unindexed);
- an unrecognised LFM2D_STAGE4 value must not quietly mean `record`.

And the load-bearing one: stage 4 NEVER changes what the hook emits. The
regex decides every outcome (test_parity.py gates that byte-for-byte);
stage 4 only writes a field on the log row.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pre_command_advisory as hook  # noqa: E402

FAILURES = []


def check(label, got, expected):
    ok = got == expected
    print(f'{"ok " if ok else "FAIL"}  {label}')
    if not ok:
        print(f'      got:      {got!r}')
        print(f'      expected: {expected!r}')
        FAILURES.append(label)


def rows(*facts):
    """Sent-clause rows in the shape the plan path builds them."""
    return list(facts)


def clause(name, args=(), redirects=(), text=None):
    return {'name': name, 'args': list(args), 'redirects': list(redirects),
            'text': text or name}


def cascade(top='data-critical', winner_index=0, scores=None):
    return {'ok': True, 'endpoint': 'cascade', 'top': top,
            'winner_index': winner_index,
            'scores': scores or {'data-critical': 0.9, 'informative': 0.05,
                                 'situation-normal': 0.05}}


def block(lfm2d, clause_rows, decision='allow', mode='record'):
    old = hook.STAGE4_MODE
    hook.STAGE4_MODE = mode
    try:
        return hook.stage4_block(lfm2d, {'decision': decision}, clause_rows)
    finally:
        hook.STAGE4_MODE = old


def main():
    # -- the flag. Off is the default, and off means the row carries no
    # stage4 key at all rather than an empty one.
    check('default mode is off', os.environ.get('LFM2D_STAGE4', 'off'), 'off')
    check('off records nothing',
          block(cascade(), rows(clause('echo', ['restored'])), mode='off'), None)
    # An unrecognised value is not `record`. It names itself on the row.
    b = block(cascade(), rows(clause('echo', ['restored'])), mode='enforce')
    check('unknown mode refuses and names itself',
          (b.get('error'), 'dismissible' in b), ('unknown_mode:enforce', False))

    # -- the happy path: a raised cascade verdict whose winning clause the
    # plan clears. The facts come from the SENT clause rows, which carry
    # args in-process -- no re-parse of the rendered text.
    b = block(cascade(winner_index=1),
              rows(clause('rm', ['-rf', '/tmp/x']), clause('echo', ['restored'])))
    check('dismisses the winning clause, not the first one',
          (b['dismissible'], b['reason'], b['winner_index'], b['verb']),
          (True, 'read_only_verb:echo', 1, 'echo'))

    # A raised verdict the plan does NOT clear survives, with the verb named.
    b = block(cascade(winner_index=0), rows(clause('rm', ['-f', 'tests/sz.rs'])))
    check('rm survives to the desk',
          (b['dismissible'], b['reason']), (False, 'verb_not_listed:rm'))

    # -- a single-clause classify row has no winner_index: the winner is the
    # only clause. Requiring winner_index here would skip every /v1/classify
    # row, which is most of the traffic.
    b = block({'ok': True, 'endpoint': 'classify', 'top': 'data-critical',
               'scores': {'data-critical': 0.9}},
              rows(clause('echo', ['GREEN'])))
    check('classify row uses its only clause',
          (b['dismissible'], b['reason'], b['winner_index']),
          (True, 'read_only_verb:echo', 0))

    # -- rows stage 4 must decline, each with a reason
    check('no classifier verdict',
          block({'ok': False, 'error': 'TimeoutError'},
                rows(clause('echo'))).get('skipped'), 'no_verdict')
    check('fallback path has no plan facts',
          block(cascade(), None).get('skipped'), 'fallback_path')
    check('classify_batch has no winner by design',
          block({'ok': True, 'endpoint': 'classify_batch',
                 'clauses': [{'top': 'data-critical', 'scores': {'data-critical': 1.0}}]},
                rows(clause('echo'))).get('skipped'), 'no_winner')
    # A verdict that was never raised is not stage 4's business: stage 4
    # reviews a RAISED verdict, and reviewing a clear one would be inventing
    # work (and, if it ever dismissed, a lowering with nothing to lower).
    check('an unraised verdict is not reviewed',
          block(cascade(top='situation-normal'),
                rows(clause('echo'))).get('skipped'), 'not_raised')
    # The vocabulary is read from the response, never assumed. A checkpoint
    # speaking none of SEVERE_LABELS buckets as a mismatch.
    check('vocabulary mismatch is named, not treated as clear',
          block({'ok': True, 'endpoint': 'cascade', 'top': 'destructive',
                 'winner_index': 0, 'scores': {'mutating': 0.1, 'destructive': 0.9}},
                rows(clause('echo'))).get('skipped'), 'vocab_mismatch')
    # An index that does not address the sent clauses is a real alignment
    # bug. Loud, not clamped to 0 -- clamping would review the WRONG clause
    # and record a confident answer about it.
    check('winner_index past the sent clauses is loud',
          block(cascade(winner_index=7), rows(clause('echo'))).get('skipped'),
          'winner_unindexed')
    check('winner_index missing on a cascade row is loud',
          block({'ok': True, 'endpoint': 'cascade', 'top': 'data-critical',
                 'scores': {'data-critical': 0.9}},
                rows(clause('echo'), clause('rm'))).get('skipped'),
          'winner_unindexed')

    # -- the ratchet reaches the hook: a guard denial is never lowered here.
    b = block(cascade(), rows(clause('echo', ['restored'])), decision='deny')
    check('guard deny is not dismissed at the hook',
          (b['dismissible'], b['reason']), (False, 'guard_denied:deny'))
    b = block(cascade(), rows(clause('echo', ['restored'])), decision='soft_deny')
    check('guard soft_deny is not dismissed at the hook',
          (b['dismissible'], b['reason']), (False, 'guard_denied:soft_deny'))
    # A warn is not a control: the classifier verdict is dismissible and the
    # warn itself is untouched, because stage 4 writes a log field and
    # nothing else.
    b = block(cascade(), rows(clause('echo', ['restored'])), decision='warn')
    check('guard warn still allows a dismissal',
          (b['dismissible'], b['reason']), (True, 'read_only_verb:echo'))

    # -- it can never break the guard. A malformed clause row must produce a
    # named error, not an exception: a raise here lands before emit() and
    # silently disables the regex guard, which is the failure the whole
    # plan path is written around.
    b = block(cascade(), ['not-a-dict'])
    check('a malformed clause row is an error, not a raise',
          (b.get('error') or '').startswith('raise:') or b.get('skipped') is not None,
          True)
    for bad in [None, {}, {'ok': True}, {'ok': True, 'endpoint': 'cascade'}]:
        try:
            got = block(bad, rows(clause('echo')))
            ok = isinstance(got, dict) and bool(
                got.get('skipped') or got.get('error') or 'dismissible' in got)
        except Exception as e:
            ok = f'RAISED {type(e).__name__}'
        check(f'never raises on lfm2d={bad!r}', ok, True)

    print()
    if FAILURES:
        print(f'FAILED: {len(FAILURES)} case(s): {FAILURES}')
        return 1
    print('ALL OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
