"""Opt-in smoke test for an ALREADY running local generator; no automatic skip.

LFM2_GENERATOR and LFM2_GENERATOR_MODEL override the local defaults.
Checks completion and output limits, NOT semantic faithfulness. Prints the
candidate previews for human review; source is the synthetic thinking fixture.
"""
import os
import subprocess
import sys
import unittest

from lfm2 import ROOT


class LocalGenerationTests(unittest.TestCase):
    def test_thinking_preview_cli_completes(self):
        result = subprocess.run([
            sys.executable, str(ROOT / "demo/lfm2.py"), "summarize", str(ROOT / "demo/thinking.txt"),
            "--generator", os.environ.get("LFM2_GENERATOR", "http://127.0.0.1:2031"),
            "--model", os.environ.get("LFM2_GENERATOR_MODEL", "lfm25-8b-a1b"),
            "--style", "preview", "--max-tokens", "1024",
        ], capture_output=True, text=True, timeout=130)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(result.stdout.strip())
        self.assertLessEqual(len(result.stdout.split()), 30)
        print(f"\nPreview for human review: {result.stdout.strip()}\n{result.stderr.strip()}", flush=True)


if __name__ == "__main__":
    unittest.main()
