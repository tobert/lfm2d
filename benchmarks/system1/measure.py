#!/usr/bin/env python3
"""Measure one spec on one set, through the daemon, and print aggregates only.

Uploads the spec's exact bytes, reads its menu, and asks every choice field of
every row with /v1/opinion (use_cache: true: a fresh input misses, as in
production), then repeats the same request at once to time a described-cache
hit. Per-row reads go to OUT_DIR/rows-<spec id>-<set>.jsonl (outside the
repo); the summary (aggregates, no rows) goes to stdout and, with --summary,
to a JSON file.

    python3 benchmarks/system1/measure.py --url http://host:8088 \
        --spec demo/web/static/life-decision-v1.json \
        --set ~/exomemory/lfm2d/system1-post-2026-09-25/actions-tune.jsonl \
        --field verdict --expect ordinary=go --expect think_twice=wait \
        --expect dangerous=stop --pass ordinary --out ~/exomemory/...

The pass category is the one the spec should leave alone. Pass-through is the
share of pass rows whose top option is the expected one; every other category
reports how often its top option is NOT the pass option (caught) and how often
it is its own expected option (exact), and the AUC of P(pass option) between
the pass rows and that category.
"""

import argparse
import hashlib
import json
import math
import pathlib
import statistics
import sys
import time
import urllib.request


def call(url, path, body=None, raw=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(url + path, data=data, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.load(r)
    return out, (time.perf_counter() - t0) * 1000


def auc(pos, neg):
    """P(a random pos scores above a random neg), ties count half."""
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    i = min(len(xs) - 1, max(0, math.ceil(q * len(xs)) - 1))
    return xs[i]


def top(probs):
    return max(probs, key=probs.get)


def summarize(rows, field, expect, pass_cat):
    pass_opt = expect[pass_cat]
    by = {}
    for r in rows:
        by.setdefault(r["category"], []).append(r)
    if pass_cat not in by:
        raise SystemExit(f"no rows in pass category {pass_cat!r}")
    p_pass = {c: [r["probs"][field][pass_opt] for r in rs] for c, rs in by.items()}
    cats = {}
    for c, rs in sorted(by.items()):
        tops = [top(r["probs"][field]) for r in rs]
        dist = {o: tops.count(o) for o in rs[0]["probs"][field]}
        masses = [r["mass"][field] for r in rs]
        entry = {
            "n": len(rs),
            "expected": expect.get(c),
            "top_counts": dist,
            "exact": sum(t == expect.get(c) for t in tops),
            "mean_prob": {o: round(statistics.fmean(r["probs"][field][o] for r in rs), 4) for o in dist},
            "mass_min": round(min(masses), 4),
            "mass_median": round(statistics.median(masses), 4),
        }
        if c == pass_cat:
            entry["passed"] = sum(t == pass_opt for t in tops)
        else:
            entry["caught"] = sum(t != pass_opt for t in tops)
            entry["auc_vs_pass"] = round(auc(p_pass[pass_cat], p_pass[c]), 4)
        cats[c] = entry
    miss = [r["t"]["miss"] for r in rows]
    hit = [r["t"]["hit"] for r in rows]

    def lat(ts, k):
        xs = [t[k] for t in ts]
        return {"p50": round(pct(xs, 0.5), 1), "p95": round(pct(xs, 0.95), 1), "max": round(max(xs), 1)}

    return {
        "field": field,
        "pass_category": pass_cat,
        "pass_option": pass_opt,
        "categories": cats,
        "latency_ms": {
            "miss_wall": lat(miss, "wall"),
            "miss_prefill": lat(miss, "prefill_ms"),
            "miss_describe": lat(miss, "describe_ms"),
            "miss_read": lat(miss, "read_ms"),
            "hit_wall": lat(hit, "wall"),
        },
        "cache_on_repeat": sorted({json.dumps(r["t"]["hit_cache"], sort_keys=True) for r in rows}),
    }


def measure(url, spec_bytes, rows, questions):
    out = []
    for i, row in enumerate(rows):
        body = {"spec": hashlib.sha256(spec_bytes).hexdigest(), "state": {"input": row["input"]},
                "questions": questions, "use_cache": True}
        r, wall = call(url, "/v1/opinion", body)
        r2, wall2 = call(url, "/v1/opinion", body)
        if r2["answers"] != r["answers"]:
            raise SystemExit(f"row {i}: the cached repeat answered differently")
        out.append({
            **row,
            "described": {d["field"]: d["value"] for d in r["described"]},
            "probs": {a["field"]: {o["option"]: o["prob"] for o in a["options"]} for a in r["answers"]},
            "mass": {a["field"]: math.exp(a["sequence_mass"]) for a in r["answers"]},
            "rendered_sha256": {a["field"]: a.get("rendered_sha256") for a in r["answers"]},
            "snapshot_id": r["snapshot_id"],
            "prefix_tokens": r["prefix_tokens"],
            "t": {"miss": {"wall": wall, **{k: r[k] for k in ("queue_ms", "prefill_ms", "describe_ms", "read_ms")},
                           "cache": r["cache"], "described_tokens": r["described_tokens"]},
                  "hit": {"wall": wall2}, "hit_cache": r2["cache"]},
        })
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(rows)}", file=sys.stderr)
    return out


def cold_repeat(url, spec_id, rows, questions):
    """Two uncached reads of each row; the engine is greedy, so they must match exactly."""
    same = 0
    for row in rows:
        body = {"spec": spec_id, "state": {"input": row["input"]}, "questions": questions, "use_cache": False}
        a, _ = call(url, "/v1/opinion", body)
        b, _ = call(url, "/v1/opinion", body)
        same += a["described"] == b["described"] and a["answers"] == b["answers"]
    return {"rows": len(rows), "bit_identical": same}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--spec", required=True, type=pathlib.Path)
    ap.add_argument("--set", required=True, type=pathlib.Path)
    ap.add_argument("--field", required=True, help="the choice field the summary scores")
    ap.add_argument("--expect", action="append", required=True, help="category=option")
    ap.add_argument("--pass", dest="pass_cat", required=True)
    ap.add_argument("--out", required=True, type=pathlib.Path, help="per-row reads (outside the repo)")
    ap.add_argument("--summary", type=pathlib.Path)
    ap.add_argument("--cold-repeat", type=int, default=0, metavar="N",
                    help="read the first N rows twice with use_cache: false and count bit-identical answers")
    ap.add_argument("--ask-only", action="store_true", help="ask only --field, as a one-question consumer would")
    a = ap.parse_args()
    expect = dict(e.split("=", 1) for e in a.expect)

    spec_bytes = a.spec.read_bytes()
    spec_id = hashlib.sha256(spec_bytes).hexdigest()
    reg, _ = call(a.url, "/v1/opinion/specs", raw=spec_bytes)
    if reg["id"] != spec_id:
        raise SystemExit(f"daemon id {reg['id']} != sha256 of the file {spec_id}")
    menu, _ = call(a.url, "/v1/opinion/specs")
    entry = next(m for m in menu if m["id"] == spec_id)
    fields = {f["field"]: f for f in entry["fields"]}
    if fields.get(a.field, {}).get("kind") != "choice":
        raise SystemExit(f"{a.field!r} is not a choice field of this spec: {sorted(fields)}")
    for c, o in expect.items():
        if o not in fields[a.field]["options"]:
            raise SystemExit(f"{o!r} (for {c}) is not an option of {a.field}: {fields[a.field]['options']}")
    questions = [{"field": f["field"]} for f in entry["fields"] if f["kind"] == "choice"]
    if a.ask_only:
        questions = [{"field": a.field}]
    info, _ = call(a.url, "/v1/adjudicator")

    rows = [json.loads(line) for line in a.set.read_text().splitlines() if line.strip()]
    t0 = time.time()
    got = measure(a.url, spec_bytes, rows, questions)
    cold = cold_repeat(a.url, spec_id, rows[: a.cold_repeat], questions) if a.cold_repeat else None
    a.out.mkdir(parents=True, exist_ok=True)
    rows_path = a.out / f"rows-{spec_id[:12]}-{a.set.stem}.jsonl"
    rows_path.write_text("".join(json.dumps(r) + "\n" for r in got))

    summary = {
        "spec_file": a.spec.name, "spec_id": spec_id, "set": a.set.name,
        "set_sha256": hashlib.sha256(a.set.read_bytes()).hexdigest(),
        "snapshot_ids": sorted({r["snapshot_id"] for r in got}),
        "rendered_sha256_count": len({h for r in got for h in r["rendered_sha256"].values()}),
        "prefix_tokens": sorted({r["prefix_tokens"] for r in got}),
        "adjudicator": {k: info[k] for k in ("model_id", "weight_hash", "device", "candle_rev", "dtype", "sampling")},
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seconds": round(time.time() - t0, 1),
        "questions": [q["field"] for q in questions],
        "cold_repeat": cold,
        **summarize(got, a.field, expect, a.pass_cat),
    }
    text = json.dumps(summary, indent=1)
    print(text)
    if a.summary:
        a.summary.parent.mkdir(parents=True, exist_ok=True)
        a.summary.write_text(text + "\n")


if __name__ == "__main__":
    main()
