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
CONTAINERFILE = REPO / 'lfm2d' / 'Containerfile.rocm'
CONTAINERIGNORE = REPO / '.containerignore'
SHIPPED = '/etc/lfm2d/demo-specs/'
BOOT = re.compile(r'--opinion-spec[=\s]+([^\s"\']+)')


def booted_paths(text):
    """Every --opinion-spec path in a manifest's args, comments ignored."""
    return [m.group(1) for l in text.splitlines() for m in BOOT.finditer(l.split('#', 1)[0])]


class ShowPropTests(unittest.TestCase):
    def test_prop_is_a_demo_spec_file(self):
        self.assertTrue((SPECS / f'{show.SPEC}.json').is_file(),
                        f'{show.SPEC} is not in {SPECS}: the boot name is the file stem')

    def test_prop_bytes_are_the_pinned_measured_spec(self):
        spec_id = hashlib.sha256((SPECS / f'{show.SPEC}.json').read_bytes()).hexdigest()
        self.assertEqual(spec_id, show.SPEC_ID, f'{show.SPEC} is not the bytes show.py pins')
        measured = {json.loads(p.read_text()).get('spec_id') for p in RESULTS.glob('*.json')}
        self.assertIn(show.SPEC_ID, measured, 'show.SPEC_ID was never measured in benchmarks/system1')

    def test_the_image_ships_demo_specs(self):
        self.assertIn(f'COPY demo/specs {SHIPPED.rstrip("/")}', CONTAINERFILE.read_text())
        self.assertIn('!demo/specs/', CONTAINERIGNORE.read_text().splitlines())

    def test_example_manifest_boots_the_prop_and_only_shipped_files(self):
        booted = booted_paths(MANIFEST.read_text())
        self.assertIn(f'{SHIPPED}{show.SPEC}.json', booted)
        for path in booted:
            self.assertTrue(path.startswith(SHIPPED), f'{MANIFEST.name} boots {path}, outside the image specs')
            self.assertTrue((SPECS / path[len(SHIPPED):]).is_file(), f'{MANIFEST.name} boots {path}, which the image lacks')

    def test_booted_paths_reads_both_spellings_and_skips_comments(self):
        text = ('  # - --opinion-spec=/etc/lfm2d/demo-specs/commented.json\n'
                '  - --opinion-spec=/etc/lfm2d/demo-specs/a.json\n'
                '  - "--opinion-spec /etc/lfm2d/demo-specs/b.json"\n')
        self.assertEqual(booted_paths(text), ['/etc/lfm2d/demo-specs/a.json', '/etc/lfm2d/demo-specs/b.json'])


class MenuCheckTests(unittest.TestCase):
    URL = 'http://daemon'

    def test_prop_missing_from_the_menu(self):
        self.assertIn('not on', show.menu_problem([], self.URL))

    def test_prop_booted_from_other_bytes(self):
        menu = [{'spec': show.SPEC, 'id': '0' * 64}]
        self.assertIn('not the measured', show.menu_problem(menu, self.URL))

    def test_prop_booted_from_the_measured_bytes(self):
        self.assertIsNone(show.menu_problem([{'spec': show.SPEC, 'id': show.SPEC_ID}], self.URL))


if __name__ == '__main__':
    unittest.main()
