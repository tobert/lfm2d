#!/usr/bin/env python3
"""Regenerate lfm2d/tests/fixtures/lfm25-json-prefix.txt.

`lfm2d/tests/adjudicator.rs::json_prefix_matches_independent_hugging_face_template_render`
asserts that our Rust `PromptSpec::render_prefix` matches what the checkpoint's
OWN chat template produces. That is only worth anything if the fixture is
rendered by the template rather than by us, so the fixture must never be
hand-edited -- editing it turns the test into a comparison of our code with
itself.

Until 2026-09-16 the fixture carried a comment saying how it had been generated
and no committed script, so the claim had no regenerator (see the
commit-the-scorer rule). This is that script.

The template comes from the checkpoint we actually serve. Two sources, in order
of preference:

  --props URL   the raw `chat_template` the llama.cpp server read out of the
                GGUF (default: the local adjudicator at :2031). This is the
                exact artifact in production, with no Hub version skew.
  --hf REPO     a HuggingFace repo id or local path, rendered through
                transformers.AutoTokenizer.apply_chat_template.

SELF-CHECK FIRST. `--verify-against-git HEAD` renders the prompt as it exists at
that git revision and requires the result to byte-match the fixture committed
there. Run that before regenerating: if the script cannot reproduce the artifact
it is replacing, it has no business replacing it.

    tests/reference/dump_json_prefix.py --verify-against-git HEAD
    tests/reference/dump_json_prefix.py --write
"""
import argparse, json, subprocess, sys, urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROMPT = 'lfm2d/prompts/shell-severity-json-v1.json'
FIXTURE = 'lfm2d/tests/fixtures/lfm25-json-prefix.txt'
BOS = '<|startoftext|>'


def system_message(spec):
    """Mirror of PromptSpec::render_prefix's system assembly (adjudicator.rs).

    Deliberately duplicated rather than shelled out to: the point of the fixture
    is an INDEPENDENT render, so this must not call the code under test. Kept
    narrow on purpose -- schema only, no tools, which is what this prompt uses.
    """
    system = spec['system']
    schema = spec.get('output_schema')
    if schema is not None:
        if spec.get('tools'):
            sys.exit('choose output_schema or tools, not both')
        # serde_json::Value is backed by a BTreeMap unless the `preserve_order`
        # feature is on, so Rust emits OBJECT keys sorted, recursively, whatever
        # order the file uses. ARRAYS keep their order -- which is why
        # `required` is what carries field order into the grammar and
        # `properties` order in the file is inert. sort_keys mirrors that.
        system += ('\nReturn exactly one JSON object matching this schema: '
                   + json.dumps(schema, separators=(',', ':'), ensure_ascii=False,
                                sort_keys=True))
    return system


def template_from_props(url):
    with urllib.request.urlopen(url.rstrip('/') + '/props', timeout=10) as r:
        props = json.load(r)
    tpl = props.get('chat_template')
    if not tpl:
        sys.exit('server at %s reports no chat_template' % url)
    return tpl


def render_with_jinja(template, system):
    """Compile with transformers' environment, not a bare jinja2 one.

    The checkpoint's template uses `{% generation %}`, which is a transformers
    extension; a plain jinja2.Environment dies with "unknown tag 'generation'".
    Using the same compiler transformers uses is also the point -- we want the
    template evaluated the way the reference implementation evaluates it.
    """
    from transformers.utils.chat_template_utils import _compile_jinja_template
    compiled = _compile_jinja_template(template)
    return compiled.render(
        messages=[{'role': 'system', 'content': system}],
        bos_token=BOS, add_generation_prompt=False, tools=None,
    )


def render_with_hf(repo, system):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(repo)
    return tok.apply_chat_template(
        [{'role': 'system', 'content': system}],
        tokenize=False, add_generation_prompt=False)


def render(args, spec):
    system = system_message(spec)
    if args.hf:
        return render_with_hf(args.hf, system)
    return render_with_jinja(template_from_props(args.props), system)


def at_revision(rev, path):
    return subprocess.run(['git', '-C', str(REPO), 'show', f'{rev}:{path}'],
                          capture_output=True, text=True, check=True).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--props', default='http://127.0.0.1:2031',
                    help='llama.cpp server serving the adjudicator checkpoint')
    ap.add_argument('--hf', help='HuggingFace repo id or local path instead of --props')
    ap.add_argument('--write', action='store_true', help='overwrite the fixture')
    ap.add_argument('--verify-against-git', metavar='REV',
                    help='render REV\'s prompt and require a byte match with REV\'s fixture')
    a = ap.parse_args()

    if a.verify_against_git:
        rev = a.verify_against_git
        spec = json.loads(at_revision(rev, PROMPT))
        want = at_revision(rev, FIXTURE)
        got = render(a, spec)
        if got == want:
            print('OK: reproduces %s:%s byte for byte (%d bytes)' % (rev, FIXTURE, len(got)))
            return 0
        print('MISMATCH against %s:%s' % (rev, FIXTURE), file=sys.stderr)
        for i, (x, y) in enumerate(zip(want, got)):
            if x != y:
                print('  first difference at byte %d: want %r got %r' % (i, x, y),
                      file=sys.stderr)
                print('  want: %r' % want[max(0, i - 40):i + 40], file=sys.stderr)
                print('  got : %r' % got[max(0, i - 40):i + 40], file=sys.stderr)
                break
        else:
            print('  identical prefix, lengths %d vs %d' % (len(want), len(got)), file=sys.stderr)
        return 1

    spec = json.loads((REPO / PROMPT).read_text())
    out = render(a, spec)
    if a.write:
        (REPO / FIXTURE).write_text(out)
        print('wrote %s (%d bytes)' % (FIXTURE, len(out)))
    else:
        sys.stdout.write(out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
