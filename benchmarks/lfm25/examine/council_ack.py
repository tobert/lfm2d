#!/usr/bin/env python3
"""Council: do acknowledgement turns stop each context answering one verdict whatever the action?

Pre-registered in benchmarks/lfm25/council/ack-v1.json (scenario, conditions, metrics, success bar), committed
before any read. Against a live lfm2d, per condition (lone user turns / "Acknowledged." / "Acknowledged, I will use
this in my deliberations."), per spec (the demo's cold and describe-first specs): one multi-context read per action,
loglinear, `rendered: true` so every rendered prompt is hashed into the results. Field names come from the specs.
The scorer (score()) is pure and unit-tested in test_council_ack.py. Take the heavy lock for the whole run:

  flock -w 900 ~/.cache/zorak-heavy.lock timeout 3600 python3 benchmarks/lfm25/examine/council_ack.py \
      --base http://127.0.0.1:8095 --out benchmarks/lfm25/results/2026-10-03-council-ack.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SCENARIO = HERE.parent / "council" / "ack-v1.json"
SPECS = [ROOT / "demo/web/static/council-verdict-v1.json", ROOT / "demo/web/static/council-describe-v1.json"]
ACKS = {"lone": None, "ack": "Acknowledged.", "ack_deliberate": "Acknowledged, I will use this in my deliberations."}


def plan(prereg: Path | None):
    """(scenario path, spec paths, {condition: (opening messages, ack)}): ack-v1's own run when `prereg` is None,
    else a later pre-registration that names its scenario, specs, slot 0 and conditions (e.g. ack-v2.json)."""
    if prereg is None:
        return SCENARIO, SPECS, {c: ([], a, None) for c, a in ACKS.items()}
    p = json.loads(prereg.read_text())
    opening = p.get("opening", [])  # conversation items [1] and [2]
    slot0 = p.get("slot0")  # the system turn's thinking-out-loud, item [0]
    conds = {c: (opening if v["opening"] else [], v["ack"], slot0 if v.get("slot0") else None)
             for c, v in p["conditions"].items()}
    return ROOT / p["scenario"], [ROOT / s for s in p["specs"]], conds


def system_turn(reviewer: str, tab: dict, system_slot0: dict | None) -> str:
    """[0]: the reviewer framing (with slot 0's thinking-out-loud in place of the sentence it would contradict),
    then the source's name and preamble."""
    if system_slot0 is not None:
        if system_slot0["drop"] not in reviewer:
            raise ValueError("slot 0 replaces a sentence the reviewer framing does not hold")
        reviewer = reviewer.replace(system_slot0["drop"], system_slot0["text"])
    return f"{reviewer}\n\nTHIS SOURCE: {tab['name']}\n{tab['preamble']}"


def call(base, method, path, body=None, raw=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"content-type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        sys.exit(f"{method} {path}: {e.code} {e.read().decode(errors='replace')}")


def messages(tab: dict, ack: str | None, opening: list[dict] = ()) -> list[dict]:
    out = list(opening)
    for m in tab["messages"]:
        out.append({"role": "user", "content": m})
        if ack is not None:
            out.append({"role": "assistant", "content": ack})
    return out


def top(options: list[str], probs: list[float]) -> str:
    best = 0
    for i in range(1, len(probs)):
        if probs[i] > probs[best]:
            best = i
    return options[best]


def score(rows: list[dict], tabs: list[str], n_actions: int) -> dict:
    """rows: {condition, spec, action, rules, per: {tab: top verdict}, pooled: top verdict}."""
    out = {}
    for cond, spec in sorted({(r["condition"], r["spec"]) for r in rows}):
        sub = [r for r in rows if r["condition"] == cond and r["spec"] == spec]
        assert len(sub) == n_actions, (cond, spec, len(sub))
        locked = {}
        for t in tabs:
            c = Counter(r["per"][t] for r in sub)
            mode, n = c.most_common(1)[0]
            locked[t] = {"mode": mode, "share": n / len(sub), "counts": dict(c)}
        hits = sum(r["pooled"] == r["rules"] for r in sub)
        out.setdefault(spec, {})[cond] = {
            "locked": locked,
            "pooled_matches_hint": hits,
            "pooled_counts": dict(Counter(r["pooled"] for r in sub)),
            "success": all(v["share"] * len(sub) <= 14 for v in locked.values()) and hits >= 10,
        }
    return out


def run(base: str, prereg: Path | None = None) -> dict:
    scenario, spec_paths, conditions = plan(prereg)
    s = json.loads(scenario.read_text())
    tabs = [t["name"] for t in s["tabs"]]
    specs = []
    for path in spec_paths:
        raw = path.read_bytes()
        spec = json.loads(raw)
        field = spec["output_schema"]["required"][-1]
        specs.append((path.stem, call(base, "POST", "/v1/opinion/specs", raw=raw)["id"], field))
    rows, mass, rendered, ms = [], [], [], []
    for cond, (opening, ack, system_slot0) in conditions.items():
        ids = [call(base, "POST", "/v1/contexts", {
            "system": system_turn(s["reviewer"], t, system_slot0),
            "messages": messages(t, ack, opening), "pin": True})["id"] for t in s["tabs"]]
        for name, sid, field in specs:
            for a in s["actions"]:
                t0 = time.perf_counter()
                r = call(base, "POST", "/v1/opinion", {
                    "spec": sid, "state": {"input": a["text"]}, "questions": [{"field": field}], "contexts": ids,
                    "pool": {"method": "loglinear"}, "rendered": True, "timeout_ms": 120000})
                ms.append((time.perf_counter() - t0) * 1000)
                per, probs = {}, {}
                for t, rd in zip(tabs, r["reads"]):
                    ans = rd["answers"][-1]
                    opts = [o["option"] for o in ans["options"]]
                    per[t] = top(opts, [o["prob"] for o in ans["options"]])
                    probs[t] = [round(o["prob"], 4) for o in ans["options"]]
                    mass.append(math.exp(ans["sequence_mass"]))
                    rendered.append(hashlib.sha256(rd["rendered"].encode()).hexdigest())
                pooled = r["pooled"][0]
                rows.append({"condition": cond, "spec": name, "action": a["text"], "rules": a["rules"], "per": per,
                             "probs": probs, "pooled": top(pooled["options"], pooled["probs"]),
                             "pooled_probs": [round(x, 4) for x in pooled["probs"]],
                             "described": {t: rd["described"] for t, rd in zip(tabs, r["reads"])}})
            print(f"{cond:15} {name:22} done", flush=True)
        for cid in ids:
            call(base, "DELETE", f"/v1/contexts/{cid}")
    info = call(base, "GET", "/v1/adjudicator")
    return {
        "date": time.strftime("%Y-%m-%d"),
        "daemon": {k: info.get(k) for k in ("model_id", "weight_hash", "backend", "device", "candle_rev", "dtype")},
        "scenario_sha256": hashlib.sha256(scenario.read_bytes()).hexdigest(),
        "prereg": str(prereg.relative_to(ROOT)) if prereg else None,
        "rendered_sha256": hashlib.sha256("".join(rendered).encode()).hexdigest(),
        "mass_min": min(mass), "mass_median": sorted(mass)[len(mass) // 2],
        "read_ms_median": sorted(ms)[len(ms) // 2],
        "score": score(rows, tabs, len(s["actions"])),
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", default="http://127.0.0.1:8095")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prereg", type=Path, default=None, help="a later pre-registration (default: ack-v1's run)")
    args = ap.parse_args()
    result = run(args.base, args.prereg.resolve() if args.prereg else None)
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("mass_min", "mass_median", "read_ms_median")}))
    for spec, conds in result["score"].items():
        for cond, v in conds.items():
            locked = " ".join(f"{t[0]}:{x['mode'][:3]}{x['share']:.2f}" for t, x in v["locked"].items())
            print(f"{spec:22} {cond:15} pooled={v['pooled_matches_hint']:2}/18 {v['pooled_counts']}  {locked}  "
                  f"success={v['success']}")


if __name__ == "__main__":
    main()
