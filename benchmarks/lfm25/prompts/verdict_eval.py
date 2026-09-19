#!/usr/bin/env python3
"""Score a verdict prompt (allow / ask / review) on OUR stack, not llama.cpp.

Amy, 2026-09-17: the LLM must not answer in the classifier's vocabulary. The
classifier says what a command DOES; the LLM decides what the harness should do
with it. So the corpus labels are mapped to a gold verdict here, at scoring time,
and never shown to the model.

One run = one prompt file = one daemon. The prompt prefix is frozen when the
daemon starts, so an arm is a process. The harness spawns lfm2d from --binary,
waits for /readyz, sends every row through POST /v1/adjudicate with the per-token
distributions switched on, and stops it.

What is recorded per row: the bytes SENT and the bytes GENERATED, verbatim, so
that everything downstream is a replay and not a second rendering (row_record
says why neither can be rebuilt afterwards); then the validated report, the
verdict, and the RAW top-k log-probabilities at the verdict's first token -- raw meaning the model's own
distribution over the full vocabulary before the grammar mask. With them comes
the mass on the verdict words, so "the model was never choosing among them" stays
visible (docs/field-requests.md decision 5). It also counts FORCED steps, where
the grammar overrode a token the model was nearly certain of.

Scoring. `--ask-labels` names the corpus labels whose gold verdict is ask; every
other label's gold verdict is allow. `review` has no gold. It counts as FLAGGED,
with ask, because both interrupt the operator; its use is reported on its own
line. Every recall is printed beside its false-alarm count, because recall
without one is meaningless (the `anti` arm, docs/lfm25-prompt-experiments.md).
A row the model did not finish is its own outcome and never a wrong answer.

Facts are the campaign's own: holdout_eval.build_facts, label-blind, from the
kaish plan and the manual pages.

--classifier-url adds the severity classifier's scores as EVIDENCE. Its labels
and their order are read from GET /v1/models at run time, never written here.
The classifier's own flag rate is scored beside the model's, from the same
scores, so "the LLM with the classifier" is always read against "the classifier
alone". Requests are small batches with a pause between them: a burst against
the live daemon has opened the advisory hook's breaker before.
"""
import argparse, hashlib, json, os, statistics, subprocess, sys, time
import urllib.error, urllib.request
from collections import Counter
from pathlib import Path

# --- repo-relative paths -------------------------------------------------
# Run outputs hold per-row verdicts, i.e. corpus rows, and corpora do not live
# in this repo. So an output directory is never defaulted into the tree: set
# LFM2D_EVAL_OUT, or pass --out. Failing loudly beats writing rows into git.
REPO = Path(__file__).resolve().parents[3]
TRAINING = Path(os.environ.get('LFM2D_TRAINING_DIR', REPO / 'training/v10'))


def eval_out(explicit=None):
    if explicit:
        return Path(explicit)
    env = os.environ.get('LFM2D_EVAL_OUT')
    if not env:
        raise SystemExit(
            'set LFM2D_EVAL_OUT (or pass --out) to a directory OUTSIDE this '
            'repo: these runs write per-row verdicts and corpora are not '
            'committed here')
    return Path(env)
# -------------------------------------------------------------------------

import holdout_eval as H  # build_facts only; its model plumbing is llama.cpp's

FLAGGED = ('ask', 'review')


def render_input(text, facts, scores=None, order=None):
    evidence = ''
    if scores is not None:
        lines = '\n'.join(f'- {label}: {scores[label]:.2f}' for label in order)
        evidence = ('Scores from the command classifier. It grades what a command does, '
                    f'from least severe to most severe:\n{lines}\n')
    return f'{facts}{evidence}Command:\n{text}'


def classify_all(url, texts, batch=16, pause=0.5):
    def call(path, payload=None):
        req = urllib.request.Request(url.rstrip('/') + path,
                                     None if payload is None else json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)
    heads = [m for m in call('/v1/models') if m['kind'] == 'classifier']
    if len(heads) != 1:
        raise SystemExit(f'{url} serves {len(heads)} classifiers; expected exactly one')
    order, out = heads[0]['labels'], {}
    for i in range(0, len(texts), batch):
        part = texts[i:i + batch]
        for text, res in zip(part, call('/v1/classify', {'inputs': part})):
            out[text] = res['scores']
        time.sleep(pause)
    return heads[0], order, out


# A step is FORCED when the grammar made the model emit a token it gave less
# than this probability. One forced step can derail everything after it: with
# compact separators the model was forced off its own `": "` style, read the
# opening quote as content, and wrote `": "` into every field.
FORCED = -4.6  # ln(0.01)


def piece(text):
    """A byte-level BPE piece as the text it stands for."""
    return text.replace('Ġ', ' ').replace('Ċ', '\n')


def verdict_step(steps, keys=('"verdict": "', '"verdict":"')):
    """The decode step that chose the verdict's first token: the first step
    after the generated text ends with the key, in either separator style.
    None if it never got there."""
    out = ''
    for i, s in enumerate(steps):
        if out.endswith(keys):
            return i
        out += piece(s['text'])
    return None


def row_record(row, gold, classifier, sent, resp, seconds):
    """One rows.jsonl record.

    `input` is the bytes sent and `output` the bytes generated, on every outcome
    that produced them. They are what a replay stands on, and neither can be
    rebuilt afterwards: build_facts reads the host's manual pages, which move,
    and `report` is the PARSED text, stored with its keys sorted, so it has lost
    both the emission order and the model's own escaping. Both of those have
    forked a replay before (docs: replay-in-emission-order).
    """
    rec = {'text': row['text'], 'label': row['label'], 'classifier': classifier,
           'gold': gold, 'input': sent, 'seconds': seconds}
    if 'http_error' in resp:
        rec.update(outcome='error', error=resp)
        return rec
    # Kept on an unfinished row too: a budget failure is not a wrong answer, and
    # the partial text is the only evidence of which one it was.
    rec['output'] = resp['output']
    if resp.get('report') is None:
        rec.update(outcome='unfinished', finish_reason=resp['finish_reason'],
                   report_error=resp.get('report_error'),
                   completion_tokens=resp['completion_tokens'])
        return rec
    steps = resp['distributions']
    at = verdict_step(steps)
    forced = [i for i, st in enumerate(steps) if st['logprob'] < FORCED]
    rec.update(outcome='answered', report=resp['report'],
               forced_steps=len(forced), first_forced=forced[:4],
               verdict=resp['report']['verdict'],
               completion_tokens=resp['completion_tokens'],
               # The token counts make a parity check possible: an examiner that
               # renders the same row must reach the same total, and if it does
               # not then the two are reading different bytes before any
               # arithmetic is blamed.
               prompt_tokens=resp['prompt_tokens'], cached_tokens=resp['cached_tokens'],
               prefill_ms=resp['prefill_ms'], decode_ms=resp['decode_ms'],
               verdict_top=None if at is None else
               [[piece(t['text'] or ''), round(t['logprob'], 4)]
                for t in steps[at]['top_logprobs']])
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--binary', type=Path, required=True)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--tokenizer', type=Path, required=True)
    ap.add_argument('--prompt', type=Path, required=True)
    ap.add_argument('--data', type=Path, default=TRAINING / 'val_F.jsonl')
    ap.add_argument('--ask-labels', required=True,
                    help='comma list of corpus labels whose gold verdict is ask; all others are allow')
    ap.add_argument('--classifier-url', help='add this lfm2d classifier\'s scores to the evidence')
    ap.add_argument('--name', help='output subdirectory; default is the prompt file\'s stem')
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--port', type=int, default=18153)
    ap.add_argument('--device', default='rocm')
    ap.add_argument('--max-tokens', type=int, default=256)
    a = ap.parse_args()
    out = eval_out(a.out) / (a.name or a.prompt.stem)
    out.mkdir(parents=True, exist_ok=False)
    ask_labels = set(a.ask_labels.split(','))

    rows = [json.loads(l) for l in a.data.read_text().splitlines() if l.strip()]
    if a.limit:
        rows = rows[:a.limit]
    seen = {r['label'] for r in rows}
    if not ask_labels <= seen:
        raise SystemExit(f'--ask-labels {sorted(ask_labels - seen)} do not occur in {a.data.name}')
    cov = Counter()
    facts = {r['text']: H.build_facts(r['text'], cov) for r in rows}
    gold = Counter('ask' if r['label'] in ask_labels else 'allow' for r in rows)
    print('rows %d gold %s facts %s' % (len(rows), dict(gold), dict(cov)), flush=True)
    head, order, scores = None, None, {}
    if a.classifier_url:
        head, order, scores = classify_all(a.classifier_url, sorted({r['text'] for r in rows}))
        print('classifier %s %s labels %s' % (head['id'], head['weight_hash'][:12], order), flush=True)

    address = f'http://127.0.0.1:{a.port}'

    def rpc(path, payload=None):
        req = urllib.request.Request(address + path,
                                     None if payload is None else json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=130) as r:
            body = r.read().decode()
            return json.loads(body) if body.startswith(('{', '[')) else body

    command = [str(a.binary), '--adjudicator-model', str(a.model),
               '--adjudicator-tokenizer', str(a.tokenizer),
               '--adjudicator-prompt', str(a.prompt), '--adjudicator-context', '4096',
               '--device', a.device, '--bind-addr', f'127.0.0.1:{a.port}', '--threads', '8']
    results = []
    with (out / 'daemon.log').open('x') as log:
        proc = subprocess.Popen(command, stdout=log, stderr=log)
        try:
            for _ in range(240):
                if proc.poll() is not None:
                    raise SystemExit(f'daemon exited {proc.returncode}; see {out}/daemon.log')
                try:
                    if rpc('/readyz') == 'ready':
                        break
                except (urllib.error.URLError, TimeoutError, ConnectionError):
                    pass
                time.sleep(.5)
            else:
                raise SystemExit('daemon did not become ready')
            info = rpc('/v1/adjudicator')
            print('prefix', json.dumps(info), flush=True)
            with (out / 'rows.jsonl').open('x') as sink:
                for n, r in enumerate(rows, 1):
                    began = time.time()
                    sent = render_input(r['text'], facts[r['text']], scores.get(r['text']), order)
                    try:
                        resp = rpc('/v1/adjudicate', {
                            'input': sent,
                            'max_tokens': a.max_tokens,
                            'distributions': {'top_k': 8}})
                    except urllib.error.HTTPError as e:
                        resp = {'http_error': e.code, 'body': e.read().decode()[:400]}
                    rec = row_record(r, 'ask' if r['label'] in ask_labels else 'allow',
                                     scores.get(r['text']), sent, resp,
                                     round(time.time() - began, 3))
                    results.append(rec)
                    sink.write(json.dumps(rec) + '\n')
                    sink.flush()
                    if n % 50 == 0:
                        done = Counter(x.get('verdict', x['outcome']) for x in results)
                        print(f'  {n}/{len(rows)} {dict(done)}', flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()

    answered = [x for x in results if x['outcome'] == 'answered']
    table = Counter((x['gold'], x['verdict']) for x in answered)
    by_label = Counter((x['label'], x['verdict']) for x in answered)
    g_ask = [x for x in answered if x['gold'] == 'ask']
    g_allow = [x for x in answered if x['gold'] == 'allow']
    caught = sum(x['verdict'] in FLAGGED for x in g_ask)
    false_alarms = sum(x['verdict'] in FLAGGED for x in g_allow)
    summary = {
        'prompt': a.prompt.name,
        'prompt_sha256': hashlib.sha256(a.prompt.read_bytes()).hexdigest(),
        'data': a.data.name, 'ask_labels': sorted(ask_labels), 'prefix': info,
        'rows': len(results), 'outcomes': dict(Counter(x['outcome'] for x in results)),
        'gold_by_verdict': {f'{g}->{v}': c for (g, v), c in sorted(table.items())},
        'label_by_verdict': {f'{l}->{v}': c for (l, v), c in sorted(by_label.items())},
        'flagged_of_gold_ask': [caught, len(g_ask)],
        'false_alarms_of_gold_allow': [false_alarms, len(g_allow)],
        'review_used': sum(x['verdict'] == 'review' for x in answered),
        'rows_with_a_forced_step_after_the_first': sum(any(i > 0 for i in x['first_forced']) for x in answered),
        'forced_steps_p50': statistics.median(x['forced_steps'] for x in answered) if answered else None,
        'flag_precision': round(caught / (caught + false_alarms), 3) if caught + false_alarms else None,
        'completion_tokens_p50': statistics.median(x['completion_tokens'] for x in answered) if answered else None,
        'seconds_p50': statistics.median(x['seconds'] for x in results),
    }
    if head:
        # The classifier alone, as a flagger: its top label is one of --ask-labels.
        top = lambda x: max(x['classifier'], key=x['classifier'].get)
        summary['classifier'] = {'id': head['id'], 'weight_hash': head['weight_hash']}
        summary['classifier_alone_flagged_of_gold_ask'] = [sum(top(x) in ask_labels for x in g_ask), len(g_ask)]
        summary['classifier_alone_false_alarms_of_gold_allow'] = [sum(top(x) in ask_labels for x in g_allow), len(g_allow)]
    (out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'prefix'}, indent=1))
    print('wrote', out)


if __name__ == '__main__':
    main()
