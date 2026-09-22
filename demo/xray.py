#!/usr/bin/env python3
"""xray: the data under the description. Every other act prints what the
model WROTE for each field; the write is a greedy argmax, and the argmax
throws the judgement away. xray asks every choice field of the spec in turn
and puts the full distribution under each written value: per option the
renormalised prob, the raw first-token logprob, the token ids, the raw mass
on the answer set. Then it shows the exact bytes the last read continued
(`rendered: true`) and checks them against the daemon's own sha256.

    python3 xray.py --url http://127.0.0.1:18171 --spec command-verdict-enum-v1

Try `git clean -fdx`: the model says it deletes user data, then writes
`undo: easy`, and its `scope` was a coin flip it wrote as a certainty.
Type an item, press enter, nothing runs. Ctrl-D exits.
"""
import argparse, hashlib, json, math, re, sys, time, urllib.error, urllib.request

GREEN, RED, YELLOW, CYAN, MAGENTA, DIM, BOLD, OFF = (
    '\033[32m', '\033[31m', '\033[33m', '\033[36m', '\033[35m', '\033[2m', '\033[1m', '\033[0m')
CONTROL = re.compile(r'<\|[a-z_]+\|>|</?think>')


def rpc(url, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url.rstrip('/') + path, data,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def bar(p, color, width=16):
    filled = round(p * width)
    return f'{color}{"█" * filled}{"░" * (width - filled)}{OFF}'


def show_prompt(rendered, full):
    """The rendered bytes, control tokens in magenta. By default from the
    last user turn on: the spec prefix is the same on every request."""
    text = rendered
    if not full:
        at = rendered.rfind('<|im_start|>user')
        if at > 0:
            print(f'{DIM}  … {len(rendered[:at].encode())} bytes of spec prefix (--full-prompt shows them){OFF}')
            text = rendered[at:]
    painted = CONTROL.sub(lambda m: f'{MAGENTA}{m.group(0)}{OFF}{DIM}', text)
    for line in painted.split('\n'):
        print(f'  {DIM}│ {line}{OFF}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://127.0.0.1:18171')
    ap.add_argument('--spec', required=True)
    ap.add_argument('--tie', type=float, default=0.15,
                    help='top-two margin (renormalised) below which a field is flagged as a near tie')
    ap.add_argument('--full-prompt', action='store_true', help='print the spec prefix too')
    ap.add_argument('--no-prompt', action='store_true',
                    help='skip the rendered view (a daemon without `rendered: true`)')
    a = ap.parse_args()

    menu = {m['spec']: m for m in rpc(a.url, '/v1/opinion/specs')}
    if a.spec not in menu:
        sys.exit(f'{a.spec!r} not on the menu: {sorted(menu)}')
    fields = menu[a.spec]['fields']
    choices = [f for f in fields if f['kind'] == 'choice']
    if not choices:
        sys.exit(f'{a.spec!r} has no choice field to x-ray')
    print(f'{BOLD}xray — what the model wrote, and the distribution it wrote it from{OFF}')
    print(f'{DIM}spec {a.spec} · choice fields {", ".join(f["field"] for f in choices)} · '
          f'snapshot {menu[a.spec]["snapshot_id"][:12]}{OFF}\n')

    while True:
        try:
            item = input(f'{BOLD}❯ {OFF}')
        except EOFError:
            print()
            return
        if not item.strip():
            continue
        reads, total_ms = [], 0.0
        for i, f in enumerate(choices):
            body = {'spec': a.spec, 'state': {'command': item},
                    'questions': [{'field': f['field']}]}
            if not a.no_prompt and i == len(choices) - 1:
                body['rendered'] = True
            t0 = time.perf_counter()
            try:
                resp = rpc(a.url, '/v1/opinion', body)
            except urllib.error.HTTPError as e:
                msg = e.read().decode()[:300]
                if 'unknown field `rendered`' in msg:
                    sys.exit(f'{RED}this daemon predates `rendered: true`: {msg}\n'
                             f'rerun with --no-prompt{OFF}')
                print(f'{RED}{e.code} {msg}{OFF}')
                break
            total_ms += (time.perf_counter() - t0) * 1000
            reads.append((f, resp))
        else:
            render(item, reads, choices, a, total_ms)


def render(item, reads, choices, a, total_ms):
    # What the model wrote for a field is in a LATER field's description:
    # the read for `verdict` describes effect/scope/undo on the way there.
    last = reads[-1][1]
    written = {d['field']: d['value'] for _, r in reads for d in r['described']}
    for d in last['described']:
        if d['field'] not in {c['field'] for c in choices}:
            print(f'  {DIM}{d["field"]:8s}{OFF} {json.dumps(d["value"])}')
    for f, resp in reads:
        ans = resp['answers'][0]
        opts = ans['options']
        top = max(opts, key=lambda o: o['prob'])
        # The write is a greedy first-token pick (under the grammar and the
        # repetition penalty), so compare it with the first-token top.
        first_top = max(opts, key=lambda o: o['first_logprob'])
        wrote = written.get(f['field'])
        mass = math.exp(ans['sequence_mass'])
        flags = []
        if wrote is None:
            head = f'{DIM}(nothing after it is a choice field: the read is all there is){OFF}'
        else:
            head = f'wrote {BOLD}{json.dumps(wrote)}{OFF}'
            if wrote != first_top['option']:
                flags.append(f'{RED}wrote ≠ first-token top ({first_top["option"]}){OFF}')
        if ans['margin'] < a.tie:
            flags.append(f'{YELLOW}near tie: margin {ans["margin"]:.3f}{OFF}')
        print(f'\n  {BOLD}{CYAN}{f["field"]}{OFF}  {head}  {"  ".join(flags)}')
        for o in opts:
            mark = '◀' if o['option'] == wrote else ' '
            color = GREEN if o is top else DIM
            print(f'    {mark} {o["option"]:16s} {bar(o["prob"], color)} {o["prob"]:6.1%}'
                  f'   {DIM}first_logprob {o["first_logprob"]:+8.3f} · logprob {o["logprob"]:+8.3f}'
                  f' · tokens {o["tokens"]}{OFF}')
        print(f'      {DIM}mass on these options {mass:.2%} (seq {ans["sequence_mass"]:+.4f},'
              f' first {ans["first_token_mass"]:+.4f}) · cache {resp["cache"]["described"]}'
              f' · {resp["describe_ms"] + resp["prefill_ms"] + resp["read_ms"]:.0f} ms server{OFF}')
    rendered = last.get('rendered')
    if rendered is None and not a.no_prompt:
        # Asked and not returned: a daemon that ignores the flag would let
        # this view silently skip the one check it exists to make.
        sys.exit(f'{RED}asked for `rendered` and the response has none{OFF}')
    if rendered is not None:
        digest = hashlib.sha256(rendered.encode()).hexdigest()
        claimed = last['answers'][0]['rendered_sha256']
        if digest != claimed:
            # The one thing this view promises; a mismatch is a daemon bug.
            sys.exit(f'{RED}rendered text hashes to {digest}, daemon says {claimed}{OFF}')
        print(f'\n  {BOLD}the bytes the {reads[-1][0]["field"]!r} options continue{OFF}  '
              f'{GREEN}sha256 {digest[:12]} ✓ matches rendered_sha256{OFF}')
        show_prompt(rendered, a.full_prompt)
    print(f'  {DIM}{len(reads)} reads · {total_ms:.0f} ms round trip{OFF}\n')


if __name__ == '__main__':
    main()
