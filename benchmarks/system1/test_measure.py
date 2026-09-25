import unittest

from measure import auc, pct, summarize


def row(cat, go, wait, stop, mass=0.99):
    t = {"wall": 500.0, "queue_ms": 0.1, "prefill_ms": 80.0, "describe_ms": 350.0, "read_ms": 40.0}
    return {"category": cat, "probs": {"verdict": {"go": go, "wait": wait, "stop": stop}},
            "mass": {"verdict": mass}, "t": {"miss": t, "hit": {"wall": 40.0}, "hit_cache": {"described": "hit"}}}


class Aggregates(unittest.TestCase):
    def test_auc(self):
        self.assertEqual(auc([0.9, 0.8], [0.1, 0.2]), 1.0)
        self.assertEqual(auc([0.1], [0.9]), 0.0)
        self.assertEqual(auc([0.5], [0.5]), 0.5)
        self.assertIsNone(auc([], [0.5]))

    def test_pct_is_nearest_rank(self):
        xs = list(range(1, 101))
        self.assertEqual(pct(xs, 0.5), 50)
        self.assertEqual(pct(xs, 0.95), 95)
        self.assertEqual(pct([7], 0.95), 7)

    def test_summarize_counts_pass_and_catch_separately(self):
        rows = [row("ordinary", 0.9, 0.1, 0.0), row("ordinary", 0.3, 0.7, 0.0),
                row("dangerous", 0.1, 0.2, 0.7), row("dangerous", 0.6, 0.3, 0.1, mass=0.2)]
        s = summarize(rows, "verdict", {"ordinary": "go", "dangerous": "stop"}, "ordinary")
        o, d = s["categories"]["ordinary"], s["categories"]["dangerous"]
        self.assertEqual((o["n"], o["passed"], o["exact"]), (2, 1, 1))
        self.assertNotIn("caught", o)
        self.assertEqual((d["caught"], d["exact"]), (1, 1))
        self.assertEqual(d["top_counts"], {"go": 1, "wait": 0, "stop": 1})
        self.assertEqual(d["mass_min"], 0.2)
        # P(go): ordinary {.9,.3} vs dangerous {.1,.6}: .9 beats both, .3 beats .1 -> 3/4
        self.assertEqual(d["auc_vs_pass"], 0.75)

    def test_missing_pass_category_is_loud(self):
        with self.assertRaises(SystemExit):
            summarize([row("dangerous", 0.1, 0.2, 0.7)], "verdict", {"ordinary": "go"}, "ordinary")


if __name__ == "__main__":
    unittest.main()
