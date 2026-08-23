"""Pin the pure functions soak_shapes.py's tables rest on. Each case was
checked to fail under a plausible wrong implementation before it passed."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import soak_shapes as ss  # noqa: E402


def sc(dc, inf, sn):
    return {'data-critical': dc, 'informative': inf, 'situation-normal': sn}


class Cascade(unittest.TestCase):
    def test_winner_is_max_severity_not_max_dc(self):
        # clause 0: dc .40 sn .10 -> sev 0.40*2+0.10 = 0.90
        # clause 1: dc .35 sn .30 -> sev 0.70+0.30 = 1.00  (wins despite lower dc)
        self.assertEqual(ss.winner([sc(.40, .50, .10), sc(.35, .35, .30)]), 1)

    def test_exact_tie_goes_to_earlier_clause(self):
        self.assertEqual(ss.winner([sc(.3, .4, .3), sc(.3, .4, .3)]), 0)

    def test_informative_carries_no_severity(self):
        self.assertEqual(ss.severity(sc(0, 1, 0)), 0.0)


class Shift(unittest.TestCase):
    prior = {'data-critical': .5, 'informative': .2, 'situation-normal': .3}

    def test_zero_tau_is_identity(self):
        s = sc(.4, .35, .25)
        out = ss.shift(s, self.prior, 0.0)
        for k in s:
            self.assertAlmostEqual(out[k], s[k])

    def test_positive_tau_shrinks_the_majority_class(self):
        s = sc(.4, .35, .25)
        out = ss.shift(s, self.prior, 1.0)
        self.assertLess(out['data-critical'], s['data-critical'])
        self.assertAlmostEqual(sum(out.values()), 1.0)


class Argv0(unittest.TestCase):
    def test_strips_env_and_sudo(self):
        self.assertEqual(ss.argv0('FOO=1 sudo rm -r /x'), 'rm')

    def test_sed_inplace_vs_read(self):
        self.assertEqual(ss.argv0("sed -i 's/a/b/' f"), 'sed -i')
        self.assertEqual(ss.argv0("sed -n '5p' f"), 'sed -n/other')
        self.assertEqual(ss.argv0('sed --in-place s/a/b/ f'), 'sed -i')

    def test_echo_banner_is_its_own_bucket(self):
        self.assertEqual(ss.argv0('echo "=== step 3 ==="'), 'echo ===')
        self.assertEqual(ss.argv0('echo hi'), 'echo')

    def test_keyword_and_assignment(self):
        self.assertEqual(ss.argv0('for f in a b'), 'kw:for')
        self.assertEqual(ss.argv0('X=1'), 'assign')


class Shape(unittest.TestCase):
    def test_stderr_merge_is_not_a_write_redirect(self):
        self.assertEqual(ss.shape('cargo test 2>&1'), 'cargo test')

    def test_write_redirect_kinds(self):
        self.assertTrue(ss.shape('cat a > b').endswith('>'))
        self.assertTrue(ss.shape('cat a >> b').endswith('>>'))

    def test_subcommand_only_for_known_verbs(self):
        self.assertEqual(ss.shape('git push origin main'), 'git push')
        self.assertEqual(ss.shape('ls foo bar'), 'ls')


class Payload(unittest.TestCase):
    def test_payload_shapes(self):
        self.assertTrue(ss.is_payload("python3 -c 'print(1)'"))
        self.assertTrue(ss.is_payload("cat <<'EOF'\nx\nEOF"))
        self.assertTrue(ss.is_payload('echo x | bash'))
        self.assertFalse(ss.is_payload('python3 script.py'))


if __name__ == '__main__':
    unittest.main()
