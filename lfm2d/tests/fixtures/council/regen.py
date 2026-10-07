#!/usr/bin/env python3
"""Regenerate components.json from the vendored council-api.openapi.yaml.

The yaml is kaijutsu's `docs/council-api.openapi.yaml`, copied verbatim; see
PROVENANCE.md for the commit. The tests read `components.json` (the schemas,
parsed once here so the crate needs no YAML dependency). Run from anywhere:

    python3 lfm2d/tests/fixtures/council/regen.py
"""
import json
import pathlib

import yaml

here = pathlib.Path(__file__).parent
doc = yaml.safe_load((here / "council-api.openapi.yaml").read_text())
out = {"openapi": doc["openapi"], "version": doc["info"]["version"], "components": doc["components"]}
(here / "components.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
print("wrote components.json for contract", out["version"])
