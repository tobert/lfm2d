#!/usr/bin/env python3
"""Synthetic form-coverage slice: BARE build/test/lint verbs.

Why (kaijutsu-lead, 2026-08-25, measured on live v10): bare `cargo test`
argmaxes data-critical 0.394 -- their single most-run command, so in
escalate mode it asks a human every time -- while `cargo test 2>&1` is
situation-normal 0.985 and `cargo test -p kaijutsu-kernel` 0.951. Same
for `cargo build` (dc 0.515), `make test` (0.700), `npm run build`
(0.972), `go build` (0.866), `go test ./...` (0.572). The live corpus
almost always carries `2>&1` or a `-p` on these (the plan renderer keeps
the redirect), so the two-token form is degenerate-short and untaught.
Rule 16 refined: build / test / lint / run against already-resolved
dependencies is situation-normal; pure reads (fmt --check, --version,
--help, dry runs) are informative.

Deterministic, seeded, committed (gitignore-negated: synthetic, no real text).

    python3 training/v10/slice_short_forms.py            # write
    python3 training/v10/slice_short_forms.py --check
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / 'short_forms.jsonl'

SN = [
    'cargo test', 'cargo build', 'cargo check', 'cargo run', 'cargo clippy', 'cargo fmt', 'cargo doc', 'cargo bench',
    'cargo test --release', 'cargo build --release', 'cargo test --all', 'cargo test --workspace', 'cargo test --lib',
    'cargo build --workspace', 'cargo clippy --all-targets', 'cargo nextest run', 'cargo insta test',
    'make', 'make test', 'make build', 'make lint', 'make check', 'make all', 'make -j8', 'make install PREFIX=~/.local',
    'npm test', 'npm run build', 'npm run lint', 'npm run test', 'npm run dev', 'pnpm test', 'pnpm build', 'yarn test', 'yarn build',
    'go build', 'go build ./...', 'go test ./...', 'go test', 'go vet ./...', 'go run .', 'go generate ./...',
    # `pytest -q` is v9 row r046 (informative, contested) -- left to v9 so the merge gate stays honest
    'pytest', 'pytest -x', 'python -m pytest', 'python3 -m pytest tests/', 'tox', 'ruff check .', 'ruff format .',
    'mypy .', 'black .', 'uv run pytest', 'uv sync',
    'mvn test', 'mvn package', 'gradle build', './gradlew test', 'dotnet build', 'dotnet test', 'bundle exec rspec', 'rake test',
    'zig build', 'zig build test', 'cmake --build build', 'ninja -C build', 'meson compile -C build',
    'kaish --help >/dev/null && cargo test', 'timeout 600 cargo test', 'nice cargo build',
]
INF = [
    'cargo --version', 'cargo fmt --check', 'cargo tree', 'cargo metadata --format-version 1', 'cargo test --list',
    'make -n', 'make --dry-run test', 'npm --version', 'npm ls', 'go version', 'go env GOPATH', 'go list ./...',
    'pytest --collect-only -q', 'pytest --version', 'ruff --version', 'zig version', 'cmake --version',
]


def rows():
    out = []
    for t in SN:
        out.append({'text': t, 'label': 'situation-normal', 'note': 'rule 16 refined: build/test/lint against resolved deps'})
    for t in INF:
        out.append({'text': t, 'label': 'informative', 'note': 'read-only tool invocation'})
    return out


def main(argv=None):
    body = ''.join(json.dumps(r) + '\n' for r in rows())
    if '--check' in (argv or sys.argv[1:]):
        same = OUT.exists() and OUT.read_text() == body
        print('CHECK: ' + ('identical' if same else 'DIFFERS'))
        return 0 if same else 1
    OUT.write_text(body)
    print(f'wrote {OUT}: {len(SN)} situation-normal, {len(INF)} informative')
    return 0


if __name__ == '__main__':
    sys.exit(main())
