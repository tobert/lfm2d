#!/usr/bin/env python3
"""Synthetic form-coverage slice for the severe forms the corpus barely
contains -- the ones that set the pass-through gate's zero-miss floor.

Why: across candidates A-D the floor was set by shred (0.47 on B-e2,
0.04 on D-e2), `git reset --hard` (0.18 / 0.06), `find / -delete`
(0.076), `rm -r <dir>` (0.23): the same recipe on the same data lands
these anywhere, because the corpus holds a handful of each form and the
fit is not bit-reproducible on this GPU. A gate that hinges on one
under-taught form is measuring the draw, not the head. PLAN slice 4
already asks for the dd family (Amy 2026-08-24: "we should add some dd
to the training so it comes out at data critical ... dd's args
nonstandard"); this spells that family and the other floor-setters,
with the target -- not the verb -- carrying the label.

Never the probe texts themselves (build_v10.py quarantines those);
variants over other paths, refs and devices. Deterministic, seeded,
committed (gitignore-negated: synthetic, no real text).

    python3 training/v10/slice_severe_forms.py            # write
    python3 training/v10/slice_severe_forms.py --check
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / 'severe_forms.jsonl'
SEED = 20260825

DEVICES = ['/dev/sdb', '/dev/sdc1', '/dev/nvme0n1', '/dev/nvme1n1p2', '/dev/vda', '/dev/mmcblk0']
SRC_DIRS = ['src/model', 'crates/d1/src', 'lib/core', 'docs/api', 'tests/integration', 'app/views']
DEV_FILES = ['src/config.rs', 'lib/util.py', 'notes/todo.md', 'crates/d1/src/mod.rs', 'app/main.go', 'docs/f1.md']
SECRET_FILES = ['~/.ssh/id_rsa', '~/.gnupg/secring.gpg', '/etc/wireguard/wg0.conf', 'secrets/prod.env',
                '/var/lib/mysql/ibdata1', '~/.config/gcloud/credentials.db']
REFS = ['HEAD~1', 'HEAD~5', 'origin/main', 'a1b2c3d', 'v1.4.0', 'HEAD^']
REMOTES = ['origin main', 'origin develop', 'upstream main', 'origin release/2.1', 'origin feature/x']
BRANCHES = ['feature-x', 'fix/set-o-unknown-option', 'spike-router', 'wip-2026-08', 'deps/chumsky-0.13',
            'old-main', 'exp/quantize', 'hotfix-9']

DC = [
    ('shred -u {f}', 'DEV_FILES', 'shred: unrecoverable by design, any target'),
    ('shred -u {f}', 'SECRET_FILES', 'shred: unrecoverable by design'),
    ('shred -n 3 -z {f}', 'SECRET_FILES', 'shred'),
    ('sudo shred -u {f}', 'SECRET_FILES', 'shred, privileged'),
    ('truncate -s 0 {f}', 'SECRET_FILES', 'wholesale truncation of primary state'),
    ('sudo truncate -s 0 {f}', 'SECRET_FILES', 'wholesale truncation'),
    ('> {f}', 'SECRET_FILES', 'truncation via redirect'),
    ('find {d} -name "*.rs" -delete', 'SRC_DIRS', 'recursive delete via find'),
    ('find {d} -type f -delete', 'SRC_DIRS', 'recursive delete via find'),
    ('find / -name "*.bak" -delete', None, 'recursive delete from root'),
    ('find {d} -name "*.py" -exec rm {{}} \\;', 'SRC_DIRS', 'recursive delete via find -exec'),
    ('git reset --hard {r}', 'REFS', 'history rewrite: Amy ruled above situation-normal'),
    ('git reset --hard', None, 'discards every uncommitted change'),
    ('git push --force {rm}', 'REMOTES', 'published history rewrite'),
    ('git push -f {rm}', 'REMOTES', 'published history rewrite, short flag'),
    ('git push --force-with-lease {rm}', 'REMOTES', 'rule 12: lease does not un-rewrite history'),
    ('rm -r {d}', 'SRC_DIRS', 'recursive delete of a source tree, no backout in the text'),
    ('rm -rf {d}', 'SRC_DIRS', 'recursive force delete of a source tree'),
    ('rm -rf {d}/', 'SRC_DIRS', 'recursive force delete, trailing slash'),
    ('dd if=/dev/zero of={dev} bs=1M', 'DEVICES', 'overwrites the raw device'),
    ('dd if=/dev/urandom of={dev} bs=4M status=progress', 'DEVICES', 'overwrites the raw device'),
    ('sudo dd if=/dev/zero of={dev} bs=1M count=100', 'DEVICES', 'overwrites the raw device head'),
    ('dd if=image.iso of={dev} bs=4M', 'DEVICES', 'writes an image over the raw device'),
    ('mkfs.ext4 {dev}', 'DEVICES', 'formats a filesystem'),
    ('sudo mkfs.xfs -f {dev}', 'DEVICES', 'formats a filesystem'),
    ('mkfs.vfat -F 32 {dev}', 'DEVICES', 'formats a filesystem'),
    ('wipefs -a {dev}', 'DEVICES', 'erases filesystem signatures'),
    ('sudo wipefs --all {dev}', 'DEVICES', 'erases filesystem signatures'),
    ('blkdiscard {dev}', 'DEVICES', 'discards every block'),
    ('sgdisk --zap-all {dev}', 'DEVICES', 'destroys the partition table'),
    ('parted {dev} mklabel gpt', 'DEVICES', 'rewrites the partition table'),
    # 2026-08-25: E-e2 read the short spelling as situation-normal on 3/3 live
    # instances (gold: data-critical, interlock-twin) while v9 carries only the
    # long `--delete --force` form. Case is semantic: -D is not -d.
    ('git branch -D {b}', 'BRANCHES', 'interlock-twin: -D bypasses the unmerged-work refusal that -d keeps (pilot gold)'),
    ('git branch -D {b} 2>&1', 'BRANCHES', 'interlock-twin, with the stderr merge real sessions add'),
    ('git branch -fD {b}', 'BRANCHES', 'interlock-twin, combined flags'),
    ('git branch -D {b} && git worktree prune', 'BRANCHES', 'interlock-twin inside a cleanup chain'),
    ('git worktree remove --force ~/src/wt/{b}', 'BRANCHES', '--force disables the dirty-tree refusal (rule 11)'),
]
# Contrast rows: same verbs where the TARGET makes them recoverable or
# observational, so the target carries the label, not the verb.
CONTRAST = [
    ('dd if={dev} of=backup.img bs=4M', 'DEVICES', 'situation-normal', 'reads the device into a new artifact (rule 2)'),
    ('dd if={dev} of=/dev/null bs=1M count=10', 'DEVICES', 'informative', 'reads into the null sink'),
    ('find {d} -name "*.rs" -print', 'SRC_DIRS', 'informative', 'find without a deleting action'),
    ('find {d} -type f -newer Cargo.toml', 'SRC_DIRS', 'informative', 'find without a deleting action'),
    ('git reset --soft {r}', 'REFS', 'situation-normal', 'keeps the working tree and index'),
    ('git reset {f}', 'DEV_FILES', 'situation-normal', 'unstages one path'),
    ('git push {rm}', 'REMOTES', 'situation-normal', 'plain push: new refs, no rewrite (pilot)'),
    ('truncate -s 0 /tmp/scratch/f1.txt', None, 'situation-normal', 'tmp is derived (rule 10)'),
    ('shred --help', None, 'informative', 'help text'),
    ('mkfs.ext4 -n {dev}', 'DEVICES', 'informative', 'dry run: -n formats nothing'),
    ('lsblk {dev}', 'DEVICES', 'informative', 'reads the device table'),
    ('parted {dev} print', 'DEVICES', 'informative', 'prints the partition table'),
    ('git branch -d {b}', 'BRANCHES', 'situation-normal', 'rule 11: -d refuses unmerged work'),
    ('git branch -d {b} 2>&1', 'BRANCHES', 'situation-normal', 'rule 11 interlock kept'),
    ('git branch --delete {b}', 'BRANCHES', 'situation-normal', 'long spelling of -d, interlock kept'),
    ('git worktree remove ~/src/wt/{b}', 'BRANCHES', 'situation-normal', 'porcelain interlock (Amy\'s ruling)'),
]
POOLS = {'DEVICES': DEVICES, 'SRC_DIRS': SRC_DIRS, 'DEV_FILES': DEV_FILES, 'SECRET_FILES': SECRET_FILES,
         'REFS': REFS, 'REMOTES': REMOTES, 'BRANCHES': BRANCHES}
KEYS = {'DEVICES': 'dev', 'SRC_DIRS': 'd', 'DEV_FILES': 'f', 'SECRET_FILES': 'f', 'REFS': 'r', 'REMOTES': 'rm',
        'BRANCHES': 'b'}
PER_FORM = 2


def rows():
    rng = random.Random(SEED)
    out, seen = [], set()

    def emit(form, pool, label, note, n):
        picks = rng.sample(POOLS[pool], n) if pool else [None]
        for v in picks:
            text = form.format(**{KEYS[pool]: v}) if pool else form
            if text in seen:
                continue
            seen.add(text)
            out.append({'text': text, 'label': label, 'note': note})

    for form, pool, note in DC:
        emit(form, pool, 'data-critical', note, PER_FORM)
    for form, pool, label, note in CONTRAST:
        emit(form, pool, label, note, PER_FORM)
    return out


def main(argv=None):
    body = ''.join(json.dumps(r) + '\n' for r in rows())
    if '--check' in (argv or sys.argv[1:]):
        same = OUT.exists() and OUT.read_text() == body
        print('CHECK: ' + ('identical' if same else 'DIFFERS'))
        return 0 if same else 1
    OUT.write_text(body)
    labels = {}
    for r in rows():
        labels[r['label']] = labels.get(r['label'], 0) + 1
    print(f'wrote {OUT}: {labels}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
