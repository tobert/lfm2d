#!/usr/bin/env python3
"""Slice 3 bulk labeling: chunk the ranked shape sample for the labelers,
then merge the families' votes and surface the splits.

Three families label blind (the pilot's trio: deepseek-v4-pro,
gemini-3.5-flash, qwen3-235b), each seeing labeler_prompt_v10.txt verbatim
plus one chunk of shapes. This tool never calls a model: the calls are
driven through kaibo `oneshot` with the prompt and the chunk ATTACHED, and
the model's reply is saved as `raw/<family>_c<N>.txt`.

Data discipline (ruling (b), 2026-08-24):
  - chunks carry REAL clause text -> `bulk/chunks/*.jsonl`, 0600, gitignored
    (kaibo can only attach files under the repo root, which is why they
    live here and not in ~/.cache like the sample they are cut from).
  - raw replies carry shape keys + labels ONLY -> `bulk/raw/*.txt`,
    committed beside the scorer (commit-the-scorer).
  - the merged votes file carries shape keys, counts, votes, consensus --
    no example text -> `bulk_votes.json`, committed.

Agreement is MEASURED, not declared (measure-disagreement-dont-declare-it):
a 2-1 split is a candidate rubric question for Amy, and a 3-way split or
an escape-hatch vote (`mixed`/`undecidable`) is flagged louder. The
twelve pilot golds ride along as instruction-following canaries.

    python3 training/v10/bulk_label.py chunk --size 50
    python3 training/v10/bulk_label.py score
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
BULK = HERE / 'bulk'
DEFAULT_SAMPLE = Path.home() / '.cache/claude-hooks/v10-shape-sample.jsonl'
GOLD = HERE / 'gold_pilot.json'

LABELS = ('informative', 'situation-normal', 'data-critical', 'mixed', 'undecidable')
RANK = {'informative': 0, 'situation-normal': 1, 'data-critical': 2}
FAMILIES = ('deepseek-v4-pro', 'gemini-3.5-flash', 'qwen3-235b')


# ---------------------------------------------------------------- pure half

def chunk_rows(rows, size):
    """Split rows into consecutive chunks of `size` (last one shorter)."""
    if size < 1:
        raise ValueError(f'chunk size must be >= 1, got {size}')
    return [rows[i:i + size] for i in range(0, len(rows), size)]


def plan_chunks(rows, size, exclude=(), start=1, only=None):
    """[(chunk_name, rows)] for the rows whose shape is not in `exclude`
    (and, when `only` is given, is in it), numbered from `start`. A later
    round labels only the delta of a re-cut sample and keeps the earlier
    chunks' numbering; a shape voted in two chunks takes the LATER vote,
    since a re-vote is cast on the current examples."""
    skip = set(exclude)
    keep = [r for r in rows if r['shape'] not in skip and (only is None or r['shape'] in set(only))]
    return [(f'chunk_{start + i:02d}', c) for i, c in enumerate(chunk_rows(keep, size))]


class ManifestError(ValueError):
    pass


def update_manifest(manifest, planned):
    """Record chunk -> shape keys. A chunk name already present must carry
    the same shapes: raw votes are keyed by chunk name, so silently
    renumbering would pair votes with the wrong shapes."""
    out = dict(manifest)
    for name, rows in planned:
        shapes = [r['shape'] for r in rows]
        if name in out and out[name] != shapes:
            raise ManifestError(f'{name} already in manifest with different shapes; pick --start past it')
        out[name] = shapes
    return out


def chunk_text(rows):
    """The labeler's input: one JSON line per shape with exactly the fields
    the prompt's Input section promises (shape, clauses, examples)."""
    return ''.join(
        json.dumps({'shape': r['shape'], 'clauses': r['clauses'], 'examples': r['examples']}) + '\n'
        for r in rows
    )


class VoteError(ValueError):
    pass


def parse_votes(text, expected_shapes):
    """Parse a labeler reply into {shape: label}. Loud on anything off-spec:
    a shape we never asked about, a label outside the vocabulary, a shape
    voted twice, or a shape left unvoted. Tolerates code fences and blank
    lines, nothing else."""
    expected = set(expected_shapes)
    votes = {}
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith('```'):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise VoteError(f'line {lineno}: not JSON: {line[:80]!r} ({e})')
        if not isinstance(obj, dict) or set(obj) != {'shape', 'label'}:
            raise VoteError(f'line {lineno}: expected {{shape,label}}, got {line[:80]!r}')
        shape, label = obj['shape'], obj['label']
        if shape not in expected:
            raise VoteError(f'line {lineno}: unexpected shape {shape!r}')
        if label not in LABELS:
            raise VoteError(f'line {lineno}: bad label {label!r} for {shape!r}')
        if shape in votes:
            raise VoteError(f'line {lineno}: duplicate vote for {shape!r}')
        votes[shape] = label
    missing = [s for s in expected_shapes if s not in votes]
    if missing:
        raise VoteError(f'{len(missing)} shape(s) unvoted: {missing[:5]}{"..." if len(missing) > 5 else ""}')
    return votes


def collect_votes(manifest, chunk_votes, in_sample):
    """Fold one family's per-chunk votes into {shape: label}. Chunks are
    walked in NAME order (zero-padded, so lexical == numeric), never dict
    order: a re-vote in a later chunk must beat the original whatever a
    JSON round-trip or a hand edit did to key order. Returns (votes,
    orphans, revotes) — orphans are shapes no longer in the sample."""
    votes, orphans, revotes = {}, [], 0
    for name in sorted(manifest):
        for shape, label in chunk_votes.get(name, {}).items():
            if shape not in in_sample:
                orphans.append(shape)
                continue
            if shape in votes:
                revotes += 1
            votes[shape] = label
    return votes, orphans, revotes


def merge(rows, votes_by_family, gold):
    """One record per sample row: the families' votes, the consensus if any,
    and how the votes fell. `kind` is one of
      unanimous  - every family present agrees
      majority   - a strict majority agrees (a 2-1 split)
      split      - no label has a majority
      incomplete - fewer than two families voted
    `escape` marks any mixed/undecidable vote so it can be surfaced apart."""
    out = []
    for r in rows:
        shape = r['shape']
        votes = {f: v[shape] for f, v in votes_by_family.items() if shape in v}
        counts = Counter(votes.values())
        n = len(votes)
        if n < 2:
            kind, consensus = 'incomplete', None
        else:
            label, top = counts.most_common(1)[0]
            if top == n:
                kind, consensus = 'unanimous', label
            elif top * 2 > n:
                kind, consensus = 'majority', label
            else:
                kind, consensus = 'split', None
        rec = {
            'rank': r['rank'], 'shape': shape, 'clauses': r['clauses'], 'share': r['share'],
            'votes': votes, 'consensus': consensus, 'kind': kind,
            'escape': any(v not in RANK for v in votes.values()),
        }
        if shape in gold:
            rec['gold'] = gold[shape]
            rec['gold_hits'] = {f: v == gold[shape] for f, v in votes.items()}
        out.append(rec)
    return out


def summarize(records, families=None):
    """Aggregates for the report: kinds, consensus label mix, family-vs-gold
    canary, and coverage share by consensus. The canary's denominator is
    every gold shape in the sample — a family that never voted on one is
    a miss, not a shrunken denominator."""
    kinds = Counter(r['kind'] for r in records)
    labels = Counter(r['consensus'] for r in records if r['consensus'])
    share = Counter()
    for r in records:
        share[r['consensus'] or f'({r["kind"]})'] += r['share']
    if families is None:
        families = sorted({f for r in records for f in r['votes']})
    gold_total = sum('gold' in r for r in records)
    gold_hits = Counter()
    for r in records:
        for f, hit in r.get('gold_hits', {}).items():
            gold_hits[f] += int(hit)
    return {
        'shapes': len(records),
        'kinds': dict(kinds),
        'consensus_labels': dict(labels),
        'share_by_consensus': {k: round(v, 5) for k, v in share.items()},
        'gold_canary': {f: f'{gold_hits[f]}/{gold_total}' for f in families},
        'escape_votes': sum(r['escape'] for r in records),
    }


# ------------------------------------------------------------------ CLI half

def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def load_manifest(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def cmd_chunk(args):
    rows = load_jsonl(args.sample)
    exclude = {r['shape'] for f in args.exclude for r in load_jsonl(f)}
    only = [l.rstrip('\n') for l in open(args.only) if l.strip()] if args.only else None
    planned = plan_chunks(rows, args.size, exclude, args.start, only)
    manifest = update_manifest(load_manifest(args.manifest), planned)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    os.chmod(out, 0o700)
    for name, rows_i in planned:
        p = out / f'{name}.jsonl'
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, 'w') as fh:
            fh.write(chunk_text(rows_i))
        os.chmod(p, 0o600)
        print(f'{p.relative_to(HERE.parent.parent)}  ranks {rows_i[0]["rank"]}-{rows_i[-1]["rank"]}  ({len(rows_i)} shapes)')
    Path(args.manifest).write_text(json.dumps(manifest, indent=1) + '\n')
    print(f'{len(planned)} chunk(s) covering {sum(len(c) for _, c in planned)} shapes '
          f'({len(exclude)} excluded); real text, 0600, gitignored; manifest -> {args.manifest}')


def cmd_score(args):
    rows = load_jsonl(args.sample)
    in_sample = {r['shape'] for r in rows}
    manifest = load_manifest(args.manifest)
    if not manifest:
        raise SystemExit(f'no manifest at {args.manifest}; run `chunk` first')
    gold = json.loads(Path(args.gold).read_text())['gold']
    raw = Path(args.raw)
    votes_by_family = {}
    orphans = set()
    revotes = {}
    for fam in args.families:
        chunk_votes = {}
        for name, shapes in manifest.items():
            p = raw / f'{fam}_{name.replace("chunk_", "c")}.txt'
            if not p.exists():
                print(f'  {fam}: {name} missing ({p.name})', file=sys.stderr)
                continue
            try:
                chunk_votes[name] = parse_votes(p.read_text(), shapes)
            except VoteError as e:
                raise SystemExit(f'{p}: {e}')
        fam_votes, fam_orphans, fam_revotes = collect_votes(manifest, chunk_votes, in_sample)
        orphans.update(fam_orphans)
        if fam_revotes:
            revotes[fam] = fam_revotes
        if fam_votes:
            votes_by_family[fam] = fam_votes
    records = merge(rows, votes_by_family, gold)
    summary = summarize(records, list(votes_by_family))
    if orphans:
        summary['orphan_shapes'] = len(orphans)  # voted on, but no longer in the sample (re-cut key)
    if revotes:
        summary['revotes'] = revotes  # a later chunk superseded an earlier vote
    print(json.dumps(summary, indent=2))

    def show(rec):
        v = ' / '.join(f'{f.split("-")[0]}={l}' for f, l in rec['votes'].items())
        g = f'  gold={rec["gold"]}' if 'gold' in rec else ''
        return f'  #{rec["rank"]:<4} {rec["shape"]!r:<32} {rec["clauses"]:>5}  {v}{g}'

    def section(title, recs):
        if recs:
            print(f'\n{title} ({len(recs)}):')
            for rec in recs:
                print(show(rec))

    section('3-WAY SPLITS', [r for r in records if r['kind'] == 'split'])
    section('2-1 SPLITS', [r for r in records if r['kind'] == 'majority'])
    section('ESCAPE-HATCH VOTES on otherwise unanimous shapes',
            [r for r in records if r['escape'] and r['kind'] == 'unanimous'])
    section('GOLD MISSES', [r for r in records if 'gold_hits' in r and not all(r['gold_hits'].values())])
    section('INCOMPLETE', [r for r in records if r['kind'] == 'incomplete'])

    if args.out:
        payload = {
            'date': args.date, 'prompt': 'training/v10/labeler_prompt_v10.txt',
            'sample': str(args.sample).replace(str(Path.home()), '~'),
            'families': list(votes_by_family), 'summary': summary, 'shapes': records,
        }
        Path(args.out).write_text(json.dumps(payload, indent=1) + '\n')
        print(f'\nwrote {args.out}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sample', default=str(DEFAULT_SAMPLE))
    ap.add_argument('--size', type=int, default=50)
    ap.add_argument('--manifest', default=str(BULK / 'manifest.json'))
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('chunk')
    c.add_argument('--out', default=str(BULK / 'chunks'))
    c.add_argument('--exclude', nargs='*', default=[], help='earlier sample file(s); their shapes are already labeled')
    c.add_argument('--start', type=int, default=1, help='first chunk number (continue an earlier round)')
    c.add_argument('--only', help='file of shape keys, one per line: re-vote just these')
    c.set_defaults(fn=cmd_chunk)
    s = sub.add_parser('score')
    s.add_argument('--raw', default=str(BULK / 'raw'))
    s.add_argument('--gold', default=str(GOLD))
    s.add_argument('--families', nargs='+', default=list(FAMILIES))
    s.add_argument('--out', default=str(HERE / 'bulk_votes.json'))
    s.add_argument('--date', default='2026-08-25')
    s.set_defaults(fn=cmd_score)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == '__main__':
    main()
