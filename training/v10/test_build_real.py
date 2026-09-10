"""Pin the loader's verb filter (Amy's ruling 2026-09-04).

Each case was checked to fail under a plausible wrong implementation
before it passed: a substring match on the rendered text drops
`grep git ...`; a check on the raw bash line instead of the plan's
`name` drops `echo 'git push'`; forgetting the stat makes the drop
silent, which is the thing we refuse.
"""
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_real as br  # noqa: E402
from soak_shapes import shape  # noqa: E402


def plan(cmd):
    """The plan command dicts for `cmd`, skipped if kaish can't render it."""
    cmds = br.plan_commands(cmd)
    if not cmds:
        raise unittest.SkipTest(f'kaish did not plan: {cmd!r}')
    return cmds


def label_every_shape(by_cmd, label='informative'):
    """A labels dict covering every shape in `by_cmd`, so the only thing
    that can remove a row is the verb filter under test."""
    labels = {}
    for cmds in by_cmd.values():
        for c in cmds:
            labels[shape(br._render_command(c))] = label
    return labels


def run(*commands, cap=100):
    by_cmd = {c: plan(c) for c in commands}
    rows, stats, _ = br.build_rows(by_cmd, label_every_shape(by_cmd), cap)
    return rows, stats


class DroppedVerbs(unittest.TestCase):
    def test_git_is_dropped(self):
        rows, stats = run('git status')
        self.assertEqual(rows, [])
        self.assertEqual(stats['dropped_verb'], 1)

    def test_git_dash_c_is_dropped_too(self):
        # `git -C <dir> status` is still the git verb; a filter keyed on
        # the second word rather than the command name would keep it.
        rows, _ = run('git -C /srv/repo status')
        self.assertEqual(rows, [])

    def test_a_non_git_clause_survives_beside_a_git_one(self):
        rows, stats = run('git status; ls -la /srv')
        texts = [r['text'] for r in rows]
        self.assertEqual(len(rows), 1, texts)
        self.assertTrue(texts[0].startswith('ls'), texts)
        self.assertEqual(stats['dropped_verb'], 1)

    def test_git_as_an_argument_is_not_dropped(self):
        # The filter reads the plan's command NAME. A substring match on
        # the rendered clause would drop this one.
        rows, stats = run("grep -rn git /srv/notes")
        self.assertEqual(len(rows), 1, [r['text'] for r in rows])
        self.assertEqual(stats['dropped_verb'], 0)

    def test_git_inside_a_quoted_word_is_not_dropped(self):
        rows, _ = run("echo 'git push --force'")
        self.assertEqual(len(rows), 1, [r['text'] for r in rows])

    def test_clause_count_still_counts_the_dropped_clause(self):
        # `clauses` is the denominator every rate in the build report is
        # over; hiding the drop from it would overstate coverage.
        _, stats = run('git status; ls -la /srv')
        self.assertEqual(stats['clauses'], 2)

    def test_the_dropped_set_is_exactly_git(self):
        # Amy ruled git out, and nothing else. A future addition should
        # be a deliberate edit with its own ruling, so pin the set.
        self.assertEqual(br.DROPPED_VERBS, frozenset({'git'}))



class ReviewTarget(unittest.TestCase):
    """The review file is Amy's reviewed artifact; a rebuild used to
    clobber it silently. These fail if the path is hardcoded again."""

    def _fixture(self):
        rows = [{'text': 'ls -la d1', 'label': 'informative', 'shape': 'ls', 'n': 2}]
        examples = {'ls': {'ls -la d1': 'ls -la /srv/secret-project'}}
        return rows, examples, {'ls': 'informative'}

    def test_it_writes_where_it_is_told(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / 'review.txt'
            br.write_review(target, *self._fixture())
            self.assertTrue(target.exists())
            self.assertIn('ls -la /srv/secret-project', target.read_text())

    def test_it_does_not_touch_the_default(self):
        import tempfile
        before = br.REVIEW.stat().st_mtime_ns if br.REVIEW.exists() else None
        with tempfile.TemporaryDirectory() as d:
            br.write_review(Path(d) / 'review.txt', *self._fixture())
        after = br.REVIEW.stat().st_mtime_ns if br.REVIEW.exists() else None
        self.assertEqual(before, after)

    def test_it_is_0600_because_it_holds_real_text(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / 'review.txt'
            br.write_review(target, *self._fixture())
            self.assertEqual(target.stat().st_mode & 0o777, 0o600)

    def test_review_is_on_the_cli(self):
        # Exercises main()'s own parser rather than grepping the source,
        # so it fails if the flag is removed OR never wired to argparse.
        import contextlib, io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), self.assertRaises(SystemExit):
            br.main(['--help'])
        self.assertIn('--review', buf.getvalue())

if __name__ == '__main__':
    unittest.main()
