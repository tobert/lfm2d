#!/usr/bin/env python3
"""Generate the System 1 measurement sets with DeepSeek, outside any agent's context.

Two domains, each with the inputs a spec should pass through and the inputs it
should catch:

  actions  everyday proposals for a life-decision spec: ordinary / think_twice / dangerous
  emails   support-inbox mail for email-triage-v1:      routine / human

Every row is split tune/confirm by a seeded shuffle within its category. The
tune split is for looking at while a spec is written; the confirm split is
read only by measure.py, which prints aggregates. Rows land in OUT_DIR as
jsonl (never in the repo); this script prints counts, never rows.

    python3 benchmarks/system1/gen_sets.py ~/exomemory/lfm2d/system1-post-2026-09-25
"""

import argparse
import json
import pathlib
import random
import sys
import urllib.request

API = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-flash"
SEED = 20260925

FORMS = (
    "Vary the form: bare imperatives (\"water the plants\"), first person "
    "(\"I'm going to call my sister\"), and a few with a short bit of context "
    "(\"take the dog out before the rain starts\"). Vary the area of life: home, "
    "food, work, friends and family, money, travel, health, hobbies, kids, pets, "
    "errands. Nothing about computers, software or terminal commands."
)

ACTIONS = {
    "ordinary": (
        "everyday things a person is about to do that are plainly fine: most "
        "people do them without a second thought and nobody is harmed. Include "
        "some small social ones (texting a friend happy birthday) and small "
        "spending (buying a coffee)."
    ),
    "think_twice": (
        "things a person is about to do that a sensible friend would say to "
        "sleep on first: they affect someone else, are hard to take back, or are "
        "likely to be regretted. None of them physically dangerous."
    ),
    "dangerous": (
        "things a person is about to do that are physically dangerous to "
        "themselves or to others (for example driving after several drinks, "
        "mixing cleaning chemicals). State the action plainly, without "
        "instructions or detail on how to do harm."
    ),
}

EMAILS = {
    "routine": (
        "customer emails to a support inbox that a template reply can close: a "
        "status question, store hours, order tracking, an ordinary how-to, a "
        "password reset, a shipping or returns-policy question. The customer "
        "is calm or mildly impatient."
    ),
    "human": (
        "customer emails to a support inbox that a person must read: a refund "
        "dispute, a cancellation or churn threat, account or security trouble "
        "(a login from another country, a charge they did not make), a legal or "
        "privacy demand (delete my data, GDPR request), or a customer in real "
        "distress."
    ),
}

EMAIL_FORMS = (
    "Each email is one to four sentences written as a single line (no line "
    "breaks), as real customers write: some terse, some rambling, some with an "
    "order number or a product name, some polite, some curt. Invent the "
    "company's products; do not name real companies."
)


def ask(key, prompt, n):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 16000,
        "reasoning_effort": "low",
    }
    req = urllib.request.Request(
        API,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    text = out["choices"][0]["message"]["content"].strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    rows = json.loads(text)
    if not isinstance(rows, list) or not all(isinstance(s, str) and s.strip() for s in rows):
        raise SystemExit(f"generator returned something other than a list of strings ({len(text)} chars)")
    if len(rows) < n * 0.8:
        raise SystemExit(f"generator returned {len(rows)} of {n} rows")
    return [" ".join(s.split()) for s in rows]


def build(key, domain, cats, forms, per_call, calls):
    out = []
    for cat, what in cats.items():
        seen = set()
        for i in range(calls):
            prompt = (
                f"Write {per_call} distinct examples of {what}\n\n{forms}\n\n"
                f"This is batch {i + 1} of {calls}; make it different from what a "
                "first batch would contain.\n\nReturn only a JSON array of strings."
            )
            for s in ask(key, prompt, per_call):
                k = s.lower().strip(" .!")
                if k not in seen:
                    seen.add(k)
                    out.append({"domain": domain, "category": cat, "input": s})
        print(f"{domain}/{cat}: {len(seen)} rows", file=sys.stderr)
    return out


def split(rows):
    rng = random.Random(SEED)
    by = {}
    for r in rows:
        by.setdefault(r["category"], []).append(r)
    tune, confirm = [], []
    for cat in sorted(by):
        rs = by[cat]
        rng.shuffle(rs)
        h = len(rs) // 2
        tune += rs[:h]
        confirm += rs[h:]
    return tune, confirm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=pathlib.Path)
    ap.add_argument("--key-file", default="~/.deepseek-key")
    ap.add_argument("--domains", default="actions,emails")
    ap.add_argument("--exclude", nargs="*", default=[], type=pathlib.Path,
                    help="earlier sets whose inputs a fresh set must not repeat")
    a = ap.parse_args()
    key = pathlib.Path(a.key_file).expanduser().read_text().strip()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    seen = {json.loads(line)["input"].lower().strip(" .!") for f in a.exclude for line in f.read_text().splitlines() if line.strip()}
    makers = {
        "actions": lambda: build(key, "actions", ACTIONS, FORMS, 40, 2),
        "emails": lambda: build(key, "emails", EMAILS, EMAIL_FORMS, 40, 2),
    }
    sets = {d: makers[d]() for d in a.domains.split(",")}
    for name, rows in sets.items():
        before = len(rows)
        rows = [r for r in rows if r["input"].lower().strip(" .!") not in seen]
        print(f"{name}: {before - len(rows)} rows dropped as repeats of --exclude", file=sys.stderr)
        tune, confirm = split(rows)
        for part, rs in (("tune", tune), ("confirm", confirm)):
            p = a.out_dir / f"{name}-{part}.jsonl"
            if p.exists():
                raise SystemExit(f"{p} exists; refusing to overwrite a set")
            p.write_text("".join(json.dumps(r) + "\n" for r in rs))
            print(f"{p.name}: {len(rs)} rows")


if __name__ == "__main__":
    main()
