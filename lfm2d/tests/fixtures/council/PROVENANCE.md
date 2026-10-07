# Council contract fixtures

`council-api.openapi.yaml` is kaijutsu's `docs/council-api.openapi.yaml`,
copied verbatim (contract 0.2.5-draft, kaijutsu commit
`bc07c2426ea47ab9adcb3a817652d504c4441c8f`, the last commit touching the
file as of 2026-10-07). It is the oracle `tests/council_contract.rs`
validates our wire bodies against. Where our code and this file disagree,
the code is wrong until the contract is changed upstream.

`components.json` is generated from it by `regen.py`. Update both together,
and record the new kaijutsu commit here.

Amy, 2026-10-07: comply with the wire contract; app behaviour the contract
describes (control text, bit-identical snapshots, ...) is not a gate yet.
