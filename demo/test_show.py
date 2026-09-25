"""The prop the matinee needs is a file the image ships, the spec a benchmark
measured byte for byte, and on the example manifest's boot menu."""
import hashlib
import json
import re
import unittest
from pathlib import Path

import show

REPO = show.HERE.parent
SPECS = show.HERE / 'specs'
RESULTS = REPO / 'benchmarks' / 'system1' / 'results'
MANIFEST = REPO / 'lfm2d' / 'deploy' / 'k8s-system1.yaml'
# Containerfile.rocm: COPY demo/specs /etc/lfm2d/demo-specs
BOOT = re.compile(r'--opinion-spec=/etc/lfm2d/demo-specs/(\S+)')


class ShowPropTests(unittest.TestCase):
    def test_prop_is_a_demo_spec_file(self):
        self.assertTrue((SPECS / f'{show.SPEC}.json').is_file(),
                        f'{show.SPEC} is not in {SPECS}: the boot name is the file stem')

    def test_prop_bytes_are_a_measured_spec(self):
        spec_id = hashlib.sha256((SPECS / f'{show.SPEC}.json').read_bytes()).hexdigest()
        measured = {json.loads(p.read_text())['spec_id'] for p in RESULTS.glob('*.json')}
        self.assertIn(spec_id, measured,
                      f'{show.SPEC} was edited after it was measured, or never measured')

    def test_example_manifest_boots_the_prop_and_only_shipped_files(self):
        booted = BOOT.findall(MANIFEST.read_text())
        self.assertIn(f'{show.SPEC}.json', booted)
        for name in booted:
            self.assertTrue((SPECS / name).is_file(), f'{MANIFEST.name} boots {name}, which the image lacks')


if __name__ == '__main__':
    unittest.main()
