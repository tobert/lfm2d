#!/usr/bin/env python3
"""Opinion-read latency while a generative adjudication runs, and what a
read served at the generation's pauses costs the generation, against one
running lfm2d. Prints aggregates only; the inputs are synthetic.

usage: interleave_latency.py BASE_URL LABEL N

The daemon serves the fixture spec lfm2d/tests/fixtures/specs/
email-triage-v1.json (`--opinion-spec`), `--adjudicator-context=4096`.
Three phases of N each: opinion reads on an idle worker; generations
(a 309-token single-line email, three prefill chunks past the spec's
prefix, max_tokens 256) alone; the same generations with one opinion read
fired at a random point inside each.
Every read has distinct input, so each is a full miss (prefill +
description + scoring). The generation's text with a read must equal its
text alone (asserted). Compare two daemons (before/after) by running this
against each in turn, alternating, on an otherwise quiet GPU.
"""
import json, random, statistics, sys, threading, time, urllib.request

BASE, LABEL, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
SPEC = "email-triage-v1"

SENTENCES = [
    "I have been a customer for six years and I have never written in before",
    "last month I ordered a standing desk, two monitor arms and a replacement power supply",
    "the desk arrived with a cracked leg and the courier left it in the rain by the side gate",
    "the monitor arms arrived a week later in a box that had clearly been opened and taped shut again",
    "one arm is missing the clamp and the other has a stripped screw that will not tighten",
    "the power supply never arrived at all even though the tracking page says it was delivered",
    "I called on the fourth and was told someone would call me back within two business days",
    "nobody called, so I called again on the ninth and was put on hold for forty minutes",
]


def post(path, body, timeout=120):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.load(r)
            status = r.status
    except urllib.error.HTTPError as e:
        out, status = json.load(e), e.code
    return status, out, (time.perf_counter() - t) * 1000


def long_email(i):
    rng = random.Random(i)
    s = SENTENCES[:]
    rng.shuffle(s)
    return f"Email: Case {i}: " + "; ".join(s * 2) + ". Please tell me what happens next."


def generation(i):
    return {"spec": SPEC, "input": long_email(i), "max_tokens": 256, "timeout_ms": 120000}


def read(tag):
    return {"spec": SPEC, "state": {"input": f"Hi, order {tag} shows delivered but nothing came. What now?"},
            "questions": [{"field": "verdict"}], "timeout_ms": 120000}


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def summary(name, xs):
    return f"{name}: n={len(xs)} p50={statistics.median(xs):.0f} p90={pct(xs, 90):.0f} max={max(xs):.0f} ms"


# Warm the kernels once.
post("/v1/adjudicate", generation(10_000))
post("/v1/opinion", read("warm"))

run = f"{LABEL}-{int(time.time())}"
# 1. Reads on an idle worker (distinct inputs: every read a full miss).
idle = []
for i in range(N):
    s, out, ms = post("/v1/opinion", read(f"{run}-idle-{i}"))
    assert s == 200, out
    idle.append(ms)

# 2. Generations alone.
alone_wall, alone_tokens, alone_decode, alone_out = [], [], [], []
for i in range(N):
    s, out, ms = post("/v1/adjudicate", generation(i))
    assert s == 200, out
    alone_wall.append(ms)
    alone_tokens.append(out["completion_tokens"])
    alone_decode.append(out["decode_ms"])
    alone_out.append(out["output"])

# 3. The same generations, each with one read fired at a random point in it.
during, during_queue, with_wall, with_decode, read_work = [], [], [], [], []
for i in range(N):
    result = {}
    t = threading.Thread(target=lambda: result.update(g=post("/v1/adjudicate", generation(i))))
    t.start()
    time.sleep(random.Random(i).uniform(0.2, 0.9 * statistics.median(alone_wall) / 1000))
    s, out, ms = post("/v1/opinion", read(f"{run}-during-{i}"))
    assert s == 200, out
    during.append(ms)
    during_queue.append(out["queue_ms"])
    read_work.append(out["prefill_ms"] + out["describe_ms"] + out["read_ms"])
    t.join()
    gs, gout, gms = result["g"]
    assert gs == 200, gout
    assert gout["completion_tokens"] == alone_tokens[i] and gout["output"] == alone_out[i], i
    with_wall.append(gms)
    with_decode.append(gout["decode_ms"])

print(f"== {LABEL}")
print(summary("read latency, idle worker", idle))
print(summary("read latency, fired during a generation", during))
print(summary("  of which queue_ms", during_queue))
print(summary("  of which read work (prefill+describe+read)", read_work))
print(summary("generation wall, alone", alone_wall))
print(summary("generation wall, with one read", with_wall))
extra = [w - a for w, a in zip(with_wall, alone_wall)]
print(summary("  generation wall added by the read", extra))
print(summary("  added beyond the read's own work", [e - w for e, w in zip(extra, read_work)]))
print(f"  tokens per generation: {statistics.median(alone_tokens)}; "
      f"decode tok/s alone p50={statistics.median([t / d * 1000 for t, d in zip(alone_tokens, alone_decode)]):.1f}")
