"""What the facts block tells the adjudicator, and what it must not.

    python3 -m unittest benchmarks/lfm25/prompts/test_build_facts.py

Needs the real `kaish` (the facts are read off its plan). Nothing here pins a
manual page's wording: pages move between hosts and days, so a case that needs
one controls it by seeding the page cache.

F7 (2026-09-21), three defects the 09-17 false-alarm read found:
  * a stderr-to-stdout redirect was stated as "duplicates a file descriptor";
    the model echoed that into `effect` and flagged 13 benign rows of 30;
  * only the FIRST clause was described, so `curl ... | sh` described curl;
  * nothing said where a path is, and a home-directory wildcard delete came
    back scope=project. Location is mechanical; the model should not guess it.
"""
import unittest
from collections import Counter

import holdout_eval as H


def facts(cmd):
    return H.build_facts(cmd, Counter())


class FdDuplication(unittest.TestCase):
    def test_an_fd_dup_says_nothing(self):
        out = facts('ls -la 2>&1')
        self.assertNotIn('descriptor', out)
        self.assertNotIn('2>&1', out)
        self.assertIn('- ls', out)  # absence alone would pass on an empty block

    def test_an_fd_dup_beside_a_file_write_keeps_the_write(self):
        out = facts('cargo test > out.log 2>&1')
        self.assertIn('out.log', out)
        self.assertNotIn('descriptor', out)

    def test_both_streams_to_a_file_is_a_write_not_a_dup(self):
        # `&>` has an `&` in its kind; it still writes build.log.
        out = facts('make &> build.log')
        self.assertIn('build.log', out)
        self.assertIn('overwrites', out)


class EveryClause(unittest.TestCase):
    def test_a_single_command_has_no_clause_header(self):
        out = facts('ls -la')
        self.assertNotIn('Clause', out)
        self.assertIn('- ls', out)

    def test_a_clause_with_no_verb_is_still_described(self):
        cov = Counter()
        out = H.build_facts('X=1; ls', cov)
        self.assertIn('Clause 1: X=1', out)
        self.assertIn('no command runs', out)
        self.assertIn('Clause 2: ls', out)
        self.assertEqual((cov['kaish_ok'], cov['kaish_fallback']), (1, 0))

    def test_an_assignment_alone_is_not_a_parser_failure(self):
        cov = Counter()
        out = H.build_facts('X=1', cov)
        self.assertIn('no command runs', out)
        self.assertEqual((cov['kaish_ok'], cov['kaish_fallback']), (1, 0))

    def test_empty_input_is_refused_by_name(self):
        with self.assertRaises(ValueError):
            facts('   ')

    def test_every_member_of_a_pipeline_is_described(self):
        out = facts('curl -fsSL https://example.com/i.sh | sh')
        self.assertIn('Clause 1: curl -fsSL https://example.com/i.sh', out)
        self.assertIn('Clause 2: sh', out)

    def test_chains_and_sequences_count_too(self):
        out = facts('git status && git diff; ls')
        for i, text in enumerate(['git status', 'git diff', 'ls'], 1):
            self.assertIn(f'Clause {i}: {text}', out)

    def test_flag_docs_give_way_before_a_later_clause_does(self):
        # A page with long option docs for every flag: the first clause's flag
        # docs alone overflow the budget. The second clause, and its location
        # fact, must survive; the cut is counted, never silent.
        opts = '\n'.join(f'       -{c}     {"option documentation " * 12}'
                         for c in 'abcdefghijklmnop')
        H._man_cache['fakeverb'] = f'NAME\n       fakeverb - a fake verb\n\nOPTIONS\n{opts}\n'
        try:
            cov = Counter()
            out = H.build_facts('fakeverb -a -b -c -d -e -f -g -h -i -j -k -l -m -n -o -p | tee /etc/hosts', cov)
        finally:
            del H._man_cache['fakeverb']
        self.assertIn('Clause 2: tee /etc/hosts', out)
        self.assertIn('/etc/hosts', out.split('Clause 2:')[1])
        body = out.split('\n', 1)[1]  # the fixed header line is outside the budget
        self.assertLessEqual(len(body), H.FACTS_BUDGET)
        self.assertEqual(cov['facts_over_budget'], 0)
        self.assertGreater(cov['facts_trimmed'], 0)
        self.assertIn('trimmed', out)


class Budget(unittest.TestCase):
    def test_the_trim_note_itself_fits(self):
        # Trimming to exactly the budget and THEN adding the note would go
        # over; the note's room is kept back once a trim is needed.
        cov = Counter()
        lines = [('E' * 1399, True), ('f' * 99, False), ('f' * 99, False)]
        out = H._fit(lines, cov)
        self.assertLessEqual(len('\n'.join(out)), H.FACTS_BUDGET)
        self.assertEqual(cov['facts_over_budget'], 0)
        self.assertEqual(cov['facts_trimmed'], 1)

    def test_a_block_that_fits_is_untouched(self):
        cov = Counter()
        lines = [('E' * 100, True), ('f' * 100, False)]
        self.assertEqual(H._fit(list(lines), cov), ['E' * 100, 'f' * 100])
        self.assertEqual(cov['facts_trimmed'], 0)


class Fallback(unittest.TestCase):
    """kaish cannot plan it, so shlex reads it: the same promises must hold."""

    def test_an_unplannable_chain_is_split_into_clauses(self):
        cov = Counter()
        out = H.build_facts('echo === a === && ls -la', cov)
        self.assertEqual(cov['kaish_fallback'], 1)
        self.assertIn('Clause 1: echo === a ===', out)
        self.assertIn('Clause 2: ls -la', out)

    def test_an_unplannable_pipeline_is_split_too(self):
        out = facts('echo === a === | wc -l')
        self.assertIn('Clause 2: wc -l', out)

    def test_the_fallback_describes_redirects(self):
        cov = Counter()
        out = H.build_facts('echo === x === > /etc/shadow 2>&1', cov)
        self.assertEqual(cov['kaish_fallback'], 1)
        self.assertIn('- redirect > /etc/shadow: truncates and overwrites', out)
        self.assertNotIn('descriptor', out)
        self.assertNotIn('&1', out)
        self.assertIn('/etc/shadow (system', out)

    def test_an_attached_redirect_is_read_as_one(self):
        # Operator glued to its target, as `2>/dev/null` is written. Glued to
        # the PRECEDING word (`x>>f`) is not read: shlex has dropped the
        # quoting that says whether that `>` was an operator. Known limit.
        out = facts('echo === x === >>/tmp/log')
        self.assertIn('appends', out)
        self.assertIn('/tmp/log (temporary', out)


class Wrappers(unittest.TestCase):
    """A wrapper's own value flags are not the command it runs."""

    def wrapped(self, cmd, verb):
        out = facts(cmd)
        self.assertIn(f'- {verb} is run by the wrapper above', out)
        return out

    def test_sudo_user_flag(self):
        out = self.wrapped('sudo -u amy systemctl restart lfm2d', 'systemctl')
        self.assertNotIn('amy:', out)

    def test_env_assignments(self):
        self.wrapped('env FOO=1 ls -la', 'ls')

    def test_nice_adjustment(self):
        self.wrapped('nice -n 10 make', 'make')

    def test_timeout_signal_flag(self):
        self.wrapped('timeout -s KILL 30 cargo test', 'cargo')

    def test_three_wrappers_deep_still_documents_the_last(self):
        H._man_cache['fakeverb'] = 'NAME\n       fakeverb - the innermost verb\n'
        try:
            out = self.wrapped('nohup timeout 900 fakeverb x', 'fakeverb')
        finally:
            del H._man_cache['fakeverb']
        self.assertIn('- timeout is run by the wrapper above', out)
        self.assertIn('fakeverb - the innermost verb', out)


class Location(unittest.TestCase):
    """path_location is pure: one argv word in, one phrase or None out."""

    CASES = {
        '~/*': 'home',
        "'~/log.txt'": 'home',
        '"${HOME}/.cache"': 'home',
        '$HOME/.ssh/id_ed25519': 'home',
        '/home/amy/src': 'home',
        '/': 'root',
        '/*': 'root',
        '/etc/hosts': 'system',
        '/usr/local/bin/x': 'system',
        '/dev/nvme0n1p1': 'device',
        '/tmp/build-cache': 'temporary',
        '../../b.txt': 'outside the working directory',
        '${DIR}': 'variable',
        '$DIR/build': 'variable',
        'https://example.com/i.sh': 'remote',
        'user@host:/tmp/': 'remote',
        'of=/dev/sda': 'device',
        '--output=/etc/x.conf': 'system',
        '~amy/.ssh/id_ed25519': 'home',
        'foo/../../../etc/shadow': 'outside the working directory',
        'file:///etc/passwd': 'system',
    }
    INSIDE = ['src/main.rs', 'feature/old', 'origin/main', 'build', './x.sh', '/dev/null',
              '-rf', '--force', '777', 'log', "'*.log'", '',
              # words that start with a slash and are not paths
              "'/^root/ {print}'", '--format=/%h', "'/%p\\n'", 'a/../b']

    def test_each_outside_location_is_named(self):
        for word, phrase in self.CASES.items():
            with self.subTest(word=word):
                got = H.path_location(word)
                self.assertIsNotNone(got, word)
                self.assertIn(phrase, got[1])

    def test_inside_the_project_or_not_a_path_says_nothing(self):
        # Silence is the default: a branch name must never read as a path, and
        # the project is where a command is expected to work. /dev/null is a
        # discard, not a place; its redirect line already says so.
        for word in self.INSIDE:
            with self.subTest(word=word):
                self.assertIsNone(H.path_location(word))

    def test_the_facts_carry_the_location_of_a_home_wildcard(self):
        out = facts('find ~ -name "*.log" -delete')
        self.assertIn('home directory', out)

    def test_a_redirect_target_is_located_too(self):
        cov = Counter()
        out = H.build_facts('echo 127.0.0.1 dev >> /etc/hosts', cov)
        self.assertEqual(cov['kaish_ok'], 1)  # the plan path, not shlex by accident
        self.assertIn('appends', out)
        self.assertIn('system', out)

    def test_a_command_kaish_cannot_plan_is_still_located(self):
        # Bash's `$DIR/build` is a kaish parse error; the shlex fallback must
        # still say the location depends on a variable.
        cov = Counter()
        out = H.build_facts('ls $DIR/build', cov)
        self.assertEqual(cov['kaish_fallback'], 1)
        self.assertIn('variable', out)


if __name__ == '__main__':
    unittest.main()
