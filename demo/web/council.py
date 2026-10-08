# SPDX-License-Identifier: MIT
# Ported from the megakernel council, council/server.py (~/src/megakernel-qwen38-flashnext-strixhalo, MIT,
# 2026-10-03), onto lfm2d's /council/v1 (the council API contract, lfm2d/src/council_api.rs).
"""The council: several held contexts judge each proposed agent action, as a client of an lfm2d daemon.

demo/web/server.py mounts this module under /council/api/* and serves its page (static/council.html) at /council. It
starts on the first request there: the other demos never pay for it. Standard library only.

On the left, tabs: one per context (default three, from council_scenario.py: Memory, User, Session; at most 8, the
most one read takes). Each tab is a small chat: a system turn (the scenario's REVIEWER framing, the tab's name and
preamble), then messages the user adds, edits and deletes, and an ask box that runs System 2 in that tab (a streamed
POST /v1/chat; the reply joins the tab). Each tab is a context the client names: a UUID minted when the tab is made and kept for the tab's life. Its
head is held on the daemon as a pinned context, the whole of it sent by PUT /council/v1/contexts/{uuid} {system, turns,
pin: true} (an ask's reply is an assistant turn with its reasoning); the daemon refuses control-token text in any turn
rather than escaping it, and that refusal is shown to the page. After any change the tab is PUT again under the same
UUID: the daemon feeds from the first difference (the page shows tokens kept and fed), and there is nothing to retire
or to count. Removing a tab is a DELETE.

On the right, decisions: an action (typed, or one of the scenario's) is ONE POST /council/v1/decisions with `contexts`
= the included tabs' contexts in tab order, the current spec's id, `ask` = its verdict question, pooled by the daemon
(linear | loglinear, uniform | mass weights). This module recomputes the daemon's pool from the reads' raw log
probabilities (council_pool.py) and stops the job loudly when it differs by more than float rounding (the contract
re-derives every number in float64), then computes what the page shows: both pools (the ternary plot's two stars), the
pooled verdict (the top option, ties to the earlier one: the daemon never picks), and leave-one-out.

Specs: two, consumer-owned, in static/council-specs/ as council specs (POSTed at boot; the daemon compiles them onto
the engine, and the spec id is the hash of the spec's canonical JSON). council-describe-v3, the default, has
each context describe the action first and then reads the verdict, so every description comes from inside its own
context; council-verdict-v3 asks the verdict cold, at the first slot, like the megakernel's letters. Both name the
agent as the proposer ("nobody here requested it") and label the input "Action proposed by the agent". Question ids and
option names come from the held spec the daemon answers, never from this file; the two specs must ask the same options. The
options run from routine to loudest, so the last one is the louder ask the page alarms on. The engine-form v1, v2 and v3 specs
stay in static/ (the benchmarks and the daemon's compile tests name them) and are not loaded.

The fence: every read sends the action set off on its own lines (FENCE, the fence the 2026-10-03 speaker-and-quoting
run pre-registered in benchmarks/lfm25/council/speaker-v1-prereg.json): on an unseen scenario describe-first passed the
bar only fenced (docs/lfm25-adjudicator.md, "Speaker and quoting"). The fence is the read's rendering: decisions and
the page keep the action as typed. An action holding a line that opens or closes a fence is refused (400).

Backfill: when the set of contexts changes (an edit, an ask's reply, an include toggle, a tab added or removed) or the
spec does, the last BACKFILL_K decisions are re-read under the new set, one read each, and each card whose pooled
verdict changed is marked with the flip (old -> new) and what caused it. A pool change re-pools the stored reads (no
read). Replay re-reads one decision under the same contexts and spec and compares the bits: a repeated read is
identical (invariant 17), so a mismatch is a bug, shown as one.

P(loudest): the pooled argmax picked the loudest option once in 24 reads of actions the hint says to report, yet
the pooled probability of the loudest option ranked every such action above every other (docs/lfm25-adjudicator.md,
"Speaker and quoting"). So each decision also carries `loud_p` (that probability, under the pool the page uses),
`loud_rank` (1 = highest among the current decisions, ties to the earlier decision) and `loud_elevated` (rank_loud's
rule). It is a rank within this page's decisions, not a calibrated probability, and it is recomputed after every
change to the decisions or the pool; a `loud` event carries every decision's three fields.

Restarts: a daemon restart forgets every context and held spec. A read that meets a 404 (its `param` names the spec or
the context) PUTs every tab again, or POSTs the specs again, and retries once; the client's UUIDs and the spec ids
come back the same.

Ask: a context holds assistant turns as text, but the generator does not: /v1/chat never continues from a held context
and never takes an assistant turn as text (invariant 16), and the council API has no generation. So an ask starts a
fresh chat from the tab's system turn and its messages (all user turns) plus the question, and when the tab already
holds a reply from an earlier ask, the ask continues that chat `from` its checkpoint, but only while the tab still holds
exactly what that chat held (later notes are appended as user turns). A tab whose reply the daemon can no longer
continue (edited since, evicted, restarted) refuses the ask with a 400 that says so: delete the reply, or add the
question as a note. The reply itself joins the context through the PUT, as an assistant turn with its reasoning.

Trust: an action is the read's `state`, fenced. The daemon refuses control-token text in it (400) instead of escaping
it, so this module needs no denylist; the refusal goes to the page as the daemon wrote it.

One engine thread owns the daemon calls; every change runs there in order. The page gets server-sent events.

  GET  /council/api/state                 the snapshot
  GET  /council/api/events                SSE: hello (the snapshot), then status, tab, tab_removed, token, ask_done,
                                          aborted, decision, backfill, replay, reset, spec, error, repinned, loud
  POST /council/api/decide {action}       one decision, answered when read
  POST /council/api/scenario              queue the scripted actions (202)
  POST /council/api/pool {method?, weights?}   re-pools every decision
  POST /council/api/spec {spec}           switch spec (a file name from the snapshot's specs); a backfill
  POST /council/api/tabs {name, preamble?, color?}                  add a tab (pinned, included)
  POST /council/api/tabs/{id} {name?, preamble?, color?, include?}  change a tab
  POST /council/api/tabs/{id}/remove
  POST /council/api/tabs/{id}/messages {role, content}              add a message
  POST /council/api/tabs/{id}/messages/{i} {content}                edit one
  POST /council/api/tabs/{id}/messages/{i}/remove
  POST /council/api/tabs/{id}/ask {text}  System 2 in the tab (202; tokens stream as events)
  POST /council/api/decisions/{id}/replay re-read under the same contexts and spec; {match}
  POST /council/api/stop                  stop an ask
  POST /council/api/reset                 back to the seed tabs and the first spec, no decisions
"""
from __future__ import annotations

import copy
import itertools
import json
import math
import queue
import re
import statistics
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid
from http import HTTPStatus
from pathlib import Path

import council_pool
import council_scenario as scenario

STATIC = Path(__file__).resolve().parent / "static"
# Describe-first first: the default, and what reset returns to.
SPEC_FILES = ["council-describe-v3.json", "council-verdict-v3.json"]
SPEC_DIR = "council-specs"  # under static/
# How a read renders an action (as benchmarks/lfm25/council/speaker-v1-prereg.json's "fence"); every read is built
# through fenced(), so decide, backfill and replay send the same bytes.
FENCE = "```\n{action}\n```"
_FENCE_LINE = re.compile(r"^[ \t]*```", re.MULTILINE)  # a line that would open or close a fence
MAX_TABS = 8  # a decision reads after 1 to 8 contexts (identity.limits.contexts_per_decision may lower it)
BACKFILL_K = 12
MAX_BODY = 64 * 1024
MAX_ACTION = 2000
MAX_NAME = 32
MAX_TEXT = 8000
MAX_MESSAGES = 64
MAX_DECISIONS = 500
HISTORY = 8  # earlier reads kept per decision
ASK_TOKENS = 2048
ASK_TIMEOUT_MS = 600_000
READ_TIMEOUT_MS = 120_000
WAIT_S = 900.0  # a request that waits for its job (behind at most an ask's reply)
BOOT_RETRY_S = 5.0
# Loglinear by default (Amy, 2026-10-03: "loglinear seems fine"): every context is asked and its probabilities
# are its confidence; a product lets a confident context dominate and a flat one barely count. Each context's own
# odds always show beside the pool.
DEFAULT_POOL = {"method": "loglinear", "weights": "uniform"}
# Elevated P(loudest), ranked within the page's decisions, never against a fixed threshold (rank within, don't
# threshold across): at least `min_n` decisions, in the top 1/`quarter` by loud_p (rank <= ceil(n / quarter)), and
# loud_p at least `median_x` times the median loud_p of all current decisions. A page-relative rank, not a calibrated
# probability: the same action can be elevated on one page and not on another.
LOUD_RULE = {"min_n": 4, "quarter": 4, "median_x": 2.0}
MIN_MASS = 0.5  # under this much raw mass on the options, the page flags a read (as tail.html does)
# One per tab: each is its identity everywhere on the page. None is green, amber or red, the verdict colors.
PALETTE = ["#22e4ff", "#ff4fd8", "#a98bff", "#4d9dff", "#e6e6f0", "#ff8fb8", "#d4a8ff", "#9ff3ff"]
TRUST = ("An action is the read's input, fenced on its own lines, rendered after each context; an action with a "
         "line that would open or close the fence is refused. The daemon refuses control-token text in it, "
         "and in every tab's turns, rather than escaping it (a 400 shown here as it came), so no action or tab text "
         "can forge a chat turn.")
_COLOR = re.compile(r"#[0-9a-fA-F]{6}")
# The daemon pools in float64 from the same raw numbers; a Python exp() is not guaranteed to round as Rust's does, so
# the check is a tolerance, far under any difference the page could show and far over rounding.
POOL_TOL = 1e-9
# lfm2d/src/council.rs read_numbers: the engine's logprobs are f32, so an answer set holding all the mass can sum a hair
# past 1. The daemon reads a log-mass in (0, MASS_NOISE] as 0 and derives the read's probabilities from that.
MASS_NOISE = 1e-6


class Refused(ValueError):
    """A request this module won't take (400); nothing changed."""


class Missing(LookupError):
    """No such tab or decision (404)."""


class ApiError(Exception):
    """The daemon answered with an error status."""

    def __init__(self, status: int, message: str, kind: str = "", param: str = ""):
        super().__init__(f"{status} {message}")
        self.status, self.message, self.kind, self.param = status, message, kind, param


class Unreachable(Exception):
    """The daemon did not answer at all."""


class Drifted(Exception):
    """The daemon read some tabs at a head the page never heard of (a PUT it took whose answer was lost)."""

    def __init__(self, tabs: list[dict]):
        super().__init__(f"the daemon holds other content for {[t['name'] for t in tabs]}")
        self.tabs = tabs


class Daemon:
    """A small JSON client of one lfm2d daemon."""

    def __init__(self, base: str, timeout: float = 180.0):
        self.base, self.timeout = base.rstrip("/"), timeout

    def _open(self, method: str, path: str, data: bytes | None, timeout: float):
        req = urllib.request.Request(self.base + path, data, method=method,
                                     headers={"content-type": "application/json"})
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            with e:
                raw = e.read()
            kind = param = ""
            try:
                err = json.loads(raw)["error"]
                msg, kind, param = err["message"], err.get("type", ""), err.get("param", "")
            except (ValueError, KeyError, TypeError):
                msg = raw.decode("utf-8", "replace")[:500]
            raise ApiError(e.code, msg, kind, param) from None
        except (urllib.error.URLError, OSError) as e:
            raise Unreachable(f"{method} {path}: {e}") from None

    def call(self, method: str, path: str, body=None):
        data = json.dumps(body).encode() if body is not None else None
        with self._open(method, path, data, self.timeout) as r:
            return r.status, json.loads(r.read() or b"null")

    def get(self, path: str):
        return self.call("GET", path)[1]

    def post(self, path: str, body):
        return self.call("POST", path, body)[1]

    def put(self, path: str, body):
        return self.call("PUT", path, body)[1]

    def delete(self, path: str):
        return self.call("DELETE", path)[1]

    def stream(self, path: str, body, stop: threading.Event):
        """POST and yield (event, data) off a server-sent-event answer; leaving the loop (or `stop`) hangs up, and
        the daemon cancels a chat turn whose client went."""
        with self._open("POST", path, json.dumps(body).encode(), 60.0) as r:
            event, data = "message", []
            while not stop.is_set():
                line = r.readline()
                if not line:
                    return
                line = line.decode("utf-8").rstrip("\r\n")
                if line == "":
                    if data:
                        yield event, json.loads("\n".join(data))
                    event, data = "message", []
                elif line.startswith(":"):
                    continue  # keep-alive
                else:
                    k, _, v = line.partition(":")
                    v = v[1:] if v.startswith(" ") else v
                    if k == "event":
                        event = v
                    elif k == "data":
                        data.append(v)


class Job:
    def __init__(self, kind: str, args: dict, wait: bool):
        self.kind, self.args = kind, args
        self.done = threading.Event() if wait else None
        self.result = None
        self.error: BaseException | None = None


def _check_action(action) -> str:
    if not isinstance(action, str) or not action.strip():
        raise Refused("action must be a non-empty string")
    if len(action) > MAX_ACTION:
        raise Refused(f"the action is too long ({len(action)} characters, at most {MAX_ACTION})")
    if _FENCE_LINE.search(action):
        raise Refused("the action has a line starting with ``` : a read quotes the action inside a ``` fence, and "
                      "that line would close the fence early; write it without a line that starts with ```")
    return action.strip()


def fenced(action: str) -> str:
    """The action as a read's input: fenced on its own lines (FENCE)."""
    return FENCE.replace("{action}", action)


def _check_name(name) -> str:
    if not isinstance(name, str) or not name.strip():
        raise Refused("name must be a non-empty string")
    if len(name.strip()) > MAX_NAME:
        raise Refused(f"name is too long (at most {MAX_NAME} characters)")
    return name.strip()


def _check_text(value, what: str, limit: int = MAX_TEXT) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Refused(f"{what} must be a non-empty string")
    if len(value) > limit:
        raise Refused(f"{what} is too long (at most {limit} characters)")
    return value


def _fields(body: dict, required: set, optional: set = frozenset()) -> dict:
    unknown = set(body) - required - optional
    if unknown:
        raise Refused(f"unknown fields {sorted(unknown)}")
    missing = required - set(body)
    if missing:
        raise Refused(f"{', '.join(sorted(missing))} is required")
    return body


def _refused_by_daemon(e: ApiError, what: str) -> Refused:
    return Refused(f"the daemon refused {what}: {e.message}")


def _close(a: float, b: float) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= POOL_TOL


def _all_close(xs, ys) -> bool:
    return len(xs) == len(ys) and all(_close(x, y) for x, y in zip(xs, ys))


def _read_mass(logprobs: list[float]) -> float:
    """A read's log-mass as the daemon derives it from its option logprobs (council.rs read_numbers)."""
    mx = max(logprobs)
    lse = mx + math.log(sum(math.exp(v - mx) for v in logprobs))
    return 0.0 if 0.0 < lse <= MASS_NOISE else lse


class Council:
    def __init__(self, daemon: Daemon, seeds: list[dict] | None = None, actions: list[dict] | None = None,
                 spec_files: list[str] | None = None, static: Path = STATIC):
        self.daemon = daemon
        self.seeds = scenario.TABS if seeds is None else seeds
        self.actions = scenario.ACTIONS if actions is None else actions
        self.spec_files = SPEC_FILES if spec_files is None else spec_files
        self.static = static
        self.lock = threading.RLock()
        self.gate = threading.Lock()
        self.jobs: queue.Queue[Job] = queue.Queue()
        self.subs: list[queue.Queue] = []
        self.status = {"phase": "starting", "detail": "not started"}
        self.booted = False
        self.started = False
        self.model: dict | None = None
        self.max_tabs = MAX_TABS
        self.specs: dict[str, dict] = {}  # file -> {file, id, body, field, options, describe, input_label}
        self.spec = self.spec_files[0]
        self.options: list[str] = []
        self.tabs: list[dict] = []
        self.decisions: list[dict] = []
        self.pool = dict(DEFAULT_POOL)
        self.causes: list[dict] = []  # context or spec changes since the last backfill
        self.tab_seq = itertools.count(1)
        self.dec_seq = itertools.count(1)
        self.stop_flag = threading.Event()

    # --- events -------------------------------------------------------------------

    def emit(self, ev: dict) -> None:
        with self.lock:
            for q in list(self.subs):
                try:
                    q.put_nowait(ev)
                except queue.Full:  # this browser stopped reading: disconnect it rather than buffer without end
                    q.dropped = True
                    self.subs.remove(q)

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=4096)
        q.dropped = False
        with self.lock:
            self.subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.lock:
            if q in self.subs:
                self.subs.remove(q)

    def set_status(self, phase: str, detail: str = "") -> None:
        self.status = {"phase": phase, "detail": detail}
        self.emit({"type": "status", **self.status})

    # --- jobs ---------------------------------------------------------------------

    def start(self) -> None:
        """Start the engine thread once (the first request under /council/api does)."""
        with self.gate:
            if self.started:
                return
            self.started = True
        threading.Thread(target=self.run, name="council", daemon=True).start()

    def submit(self, kind: str, wait: bool = False, **args) -> Job:
        job = Job(kind, args, wait)
        with self.gate:
            if self.status["phase"] == "failed":
                job.error = RuntimeError(f"the council is down: {self.status['detail']}")
                return job
            self.jobs.put(job)
        if wait and not job.done.wait(WAIT_S):
            job.error = TimeoutError(f"{kind} did not finish in {WAIT_S:g} s")
        return job

    def quiet(self) -> bool:
        """Booted, nothing queued and nothing running."""
        return self.booted and self.jobs.unfinished_tasks == 0

    def run(self) -> None:
        while True:
            try:
                self.boot()
                break
            except (Unreachable, ApiError) as e:
                if isinstance(e, ApiError) and e.status not in (502, 503, 504):
                    self.fail(e)
                    return
                self.set_status("waiting", f"the daemon is not answering yet ({e}); retrying")
                time.sleep(BOOT_RETRY_S)
            except Exception as e:  # noqa: BLE001 -- a boot that can't work: fail loudly, once
                self.fail(e)
                return
        while True:
            job = self.jobs.get()
            try:
                job.result = getattr(self, f"do_{job.kind}")(**job.args)
            except Exception as e:  # noqa: BLE001 -- the job's error goes to its waiter and to the page
                if not isinstance(e, (Refused, Missing)):
                    traceback.print_exc()
                job.error = e
                if not job.done:
                    self.emit({"type": "error", "job": job.kind, "message": f"{type(e).__name__}: {e}"})
            finally:
                if self.jobs.qsize() == 0:
                    self.set_status("idle")
                if job.done:
                    job.done.set()
                self.jobs.task_done()

    def fail(self, e: BaseException) -> None:
        traceback.print_exc()
        with self.gate:
            self.set_status("failed", f"{type(e).__name__}: {e}")
        while True:  # nothing will run what is queued: fail it loudly
            try:
                job = self.jobs.get_nowait()
            except queue.Empty:
                return
            job.error = RuntimeError(f"the council is down: {self.status['detail']}")
            self.emit({"type": "error", "job": job.kind, "message": str(job.error)})
            if job.done:
                job.done.set()
            self.jobs.task_done()

    # --- boot ---------------------------------------------------------------------

    def boot(self) -> None:
        """The daemon answers and can do what the page shows; the specs are held and ask the same options; then the
        seed tabs are built."""
        self.set_status("starting", "asking the daemon")
        ident = self.daemon.get("/council/v1/identity")
        if "leave_one_out" not in ident.get("capabilities", []):
            raise RuntimeError(f"the daemon's capabilities are {ident.get('capabilities')}: the page shows "
                               f"leave-one-out, which this daemon does not serve")
        self.model = {"model_id": ident["model"], "weight_hash": ident["weight_hash"], "backend": ident.get("device"),
                      "device": ident.get("device")}
        self.max_tabs = min(MAX_TABS, ident["limits"]["contexts_per_decision"])
        self.upload_specs()
        first = self.specs[self.spec_files[0]]
        for f in self.spec_files[1:]:
            if self.specs[f]["options"] != first["options"]:
                raise RuntimeError(f"{f} asks {self.specs[f]['options']}, {self.spec_files[0]} asks "
                                   f"{first['options']}: switching specs would change what a verdict is")
        self.options = list(first["options"])
        with self.lock:  # a boot retried after some seeds were put: those are no tab's, and are seeded again
            left, self.tabs = self.tabs, []
        for t in left:
            self.free(t["context"])
        self.seed_tabs()
        self.booted = True
        self.set_status("idle")

    def upload_specs(self) -> None:
        """Hold every spec (the same spec is the same id) and read its question off the held spec: the LAST choice
        question; the text questions before it are the description (the daemon describes text questions only)."""
        specs = {}
        for f in self.spec_files:
            body = json.loads((self.static / SPEC_DIR / f).read_text())
            held = self.daemon.post("/council/v1/specs", body)
            qs = held["spec"]["questions"]
            choices = [i for i, x in enumerate(qs) if x["type"] == "choice"]
            if not choices:
                raise RuntimeError(f"{f} has no choice question to ask")
            q = choices[-1]
            options = [c["option"] for c in qs[q]["criteria"]]
            if len(options) < 2:
                raise RuntimeError(f"{f}: question {qs[q]['id']!r} has fewer than two options")
            specs[f] = {"file": f, "id": held["spec_id"], "field": qs[q]["id"], "options": options,
                        "input_label": held["spec"]["input_label"], "describe": [x["id"] for x in qs[:q] if x["type"] == "text"]}
        with self.lock:
            self.specs = specs

    # --- tabs ---------------------------------------------------------------------

    def new_tab(self, name: str, color: str, preamble: str = "", include: bool = True,
                messages: list[dict] | None = None) -> dict:
        """A tab not yet held: its context is a UUID the client chooses, once, for the tab's life."""
        return {"id": f"t{next(self.tab_seq)}", "name": name, "color": color, "include": include,
                "preamble": preamble, "messages": messages or [], "context": str(uuid.uuid4()), "head": None,
                "n_tokens": 0, "pin_ms": 0.0, "cached_tokens": 0, "fed_tokens": 0, "chat": None}

    def seed_tabs(self) -> None:
        for k, s in enumerate(self.seeds):
            tab = self.new_tab(s["name"], s.get("color", PALETTE[k % len(PALETTE)]), s.get("preamble", ""),
                               s.get("include", True), [dict(m) for m in s.get("messages", [])])
            self.pin_new(tab)
            with self.lock:
                self.tabs.append(tab)
            self.emit({"type": "tab", "tab": self.tab_view(tab)})

    def head(self, tab: dict) -> tuple[str, list[dict]]:
        """A tab's head: the system turn (the reviewer framing, the tab's name and preamble) and its messages, an
        ask's reply with the reasoning it was generated with."""
        system = scenario.REVIEWER + f"\n\nTHIS SOURCE: {tab['name']}"
        if tab["preamble"].strip():
            system += f"\n{tab['preamble'].strip()}"
        msgs = []
        for m in tab["messages"]:
            x = {"role": m["role"], "content": m["content"]}
            if m["role"] == "assistant" and m.get("thinking") is not None:
                x["thinking"] = m["thinking"]
            msgs.append(x)
        return system, msgs

    def pin(self, tab: dict) -> None:
        """Hold the tab's head pinned: PUT the whole context under the tab's UUID. The daemon feeds from the first
        difference with what it holds, and answers how much it kept and fed."""
        self.set_status("pinning", tab["name"])
        system, msgs = self.head(tab)
        turns = [{"role": m["role"], "content": m["content"],
                  **({"reasoning": m["thinking"]} if "thinking" in m else {})} for m in msgs]
        t0 = time.perf_counter()
        try:
            ctx = self.daemon.put(f"/council/v1/contexts/{tab['context']}",
                                  {"system": system, "turns": turns, "pin": True})
        except ApiError as e:
            if e.status in (400, 413, 507):
                raise _refused_by_daemon(e, f"tab {tab['name']!r}") from None
            raise
        if ctx.get("pinned") is not True:
            raise RuntimeError(f"the daemon did not pin the context of tab {tab['name']!r}")
        with self.lock:
            tab["head"], tab["n_tokens"] = ctx["head"], ctx["tokens"]
            tab["pin_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            tab["cached_tokens"], tab["fed_tokens"] = ctx["kept"], ctx["fed"]

    def pin_new(self, tab: dict) -> None:
        """Pin a tab the page holds nothing for yet. A PUT whose outcome is unknown may have left the daemon holding
        the UUID, which no tab would ever free: free it. A refusal held nothing."""
        try:
            self.pin(tab)
        except Refused:
            raise
        except BaseException:
            self.free(tab["context"])
            raise

    def repin_all(self) -> None:
        """Hold every tab again after the daemon lost them (a restart forgets every context). Says why first, so a
        PUT that fails part way is not left unexplained."""
        self.emit({"type": "repinned", "tabs": len(self.tabs),
                   "message": f"the daemon no longer held the tabs' contexts (a restart?): PUT {len(self.tabs)} "
                              f"again"})
        for tab in self.tabs:
            self.pin(tab)
            self.emit({"type": "tab", "tab": self.tab_view(tab)})

    def free(self, context: str) -> None:
        """Drop a tab's context; a DELETE that fails is reported, not raised: the change that dropped the tab
        stands. A 404 is already the goal."""
        try:
            self.daemon.delete(f"/council/v1/contexts/{context}")
        except ApiError as e:
            if e.status == 404:
                return
            self.free_failed(context, e)
        except Exception as e:  # noqa: BLE001 -- reported to the page with the id, so it can be freed by hand
            self.free_failed(context, e)

    def free_failed(self, context: str, e: BaseException) -> None:
        traceback.print_exc()
        self.emit({"type": "error", "job": "free", "message": f"context {context} stays pinned on the daemon "
                   f"(DELETE failed: {type(e).__name__}: {e}); delete it by hand or restart the daemon"})

    def tab(self, tid: str) -> dict:
        for t in self.tabs:
            if t["id"] == tid:
                return t
        raise Missing(f"no tab {tid!r}")

    def tab_view(self, tab: dict) -> dict:
        v = json.loads(json.dumps(tab))
        v["chat"] = bool(tab.get("chat"))
        return v

    def changed(self, what: str, tab: dict | None = None, **extra) -> None:
        """A change to the contexts' set or the spec: note its cause and queue a backfill (it runs after this job)."""
        cause = {"what": what, "t": time.time(), **extra}
        if tab is not None:
            cause.update({"tab": tab["id"], "name": tab["name"], "color": tab["color"]})
        with self.lock:
            self.causes.append(cause)
        self.submit("backfill")

    def do_tab_add(self, name: str, preamble: str = "", color: str | None = None) -> dict:
        if len(self.tabs) >= self.max_tabs:
            raise Refused(f"at most {self.max_tabs} tabs: a read takes at most {self.max_tabs} contexts")
        used = {t["color"] for t in self.tabs}
        color = color or next((c for c in PALETTE if c not in used), PALETTE[len(self.tabs) % len(PALETTE)])
        tab = self.new_tab(name, color, preamble)
        self.pin_new(tab)
        with self.lock:
            self.tabs.append(tab)
        self.emit({"type": "tab", "tab": self.tab_view(tab)})
        self.changed("add", tab)
        return self.tab_view(tab)

    def do_tab_update(self, tid: str, **kw) -> dict:
        tab = self.tab(tid)
        repin = any(k in kw and kw[k] != tab[k] for k in ("name", "preamble"))
        include = "include" in kw and kw["include"] != tab["include"]
        saved = dict(tab)
        with self.lock:
            tab.update(kw)
        if repin:
            try:
                self.pin(tab)
            except BaseException:
                with self.lock:
                    tab.clear()
                    tab.update(saved)
                raise
        self.emit({"type": "tab", "tab": self.tab_view(tab)})
        if repin and tab["include"]:
            self.changed("rename" if "name" in kw and kw["name"] != saved["name"] else "preamble", tab)
        elif include:
            self.changed("include" if tab["include"] else "exclude", tab)
        return self.tab_view(tab)

    def do_tab_remove(self, tid: str) -> dict:
        tab = self.tab(tid)
        with self.lock:
            self.tabs.remove(tab)
        self.emit({"type": "tab_removed", "id": tid})
        self.free(tab["context"])
        if tab["include"]:
            self.changed("remove", tab)
        return {"removed": tid}

    def edit_messages(self, tab: dict, change, what: str) -> dict:
        """Apply `change` to the tab's messages, pin again, and restore them if the pin fails."""
        saved = [dict(m) for m in tab["messages"]]
        with self.lock:
            change(tab["messages"])
        try:
            self.pin(tab)
        except BaseException:
            with self.lock:
                tab["messages"][:] = saved
            raise
        self.emit({"type": "tab", "tab": self.tab_view(tab)})
        if tab["include"]:
            self.changed(what, tab)
        return self.tab_view(tab)

    def do_message_add(self, tid: str, role: str, content: str) -> dict:
        tab = self.tab(tid)
        if len(tab["messages"]) >= MAX_MESSAGES:
            raise Refused(f"tab {tab['name']!r} has {MAX_MESSAGES} messages, the most a tab holds")
        return self.edit_messages(tab, lambda ms: ms.append({"role": role, "content": content}), "add message")

    def _index(self, tab: dict, i: int) -> int:
        if not 0 <= i < len(tab["messages"]):
            raise Refused(f"no message {i} in tab {tab['name']!r} (it has {len(tab['messages'])})")
        return i

    def do_message_edit(self, tid: str, index: int, content: str) -> dict:
        tab = self.tab(tid)
        i = self._index(tab, index)

        def change(ms):
            ms[i] = {"role": ms[i]["role"], "content": content}  # an edited reply drops the reasoning it no longer matches
        return self.edit_messages(tab, change, "edit")

    def do_message_remove(self, tid: str, index: int) -> dict:
        tab = self.tab(tid)
        i = self._index(tab, index)
        return self.edit_messages(tab, lambda ms: ms.pop(i), "delete message")

    # --- System 2 in a tab ----------------------------------------------------------

    def ask_route(self, tab: dict) -> dict:
        """How an ask in this tab reaches /v1/chat: a fresh chat when the tab holds no reply, else a continuation of
        the chat that wrote its replies, while the tab still holds exactly what that chat held. Refused otherwise:
        /v1/chat never takes an assistant turn as text."""
        system, msgs = self.head(tab)
        if not any(m["role"] == "assistant" for m in msgs):
            return {"system": system, "messages": msgs}
        chat = tab.get("chat")
        if chat and chat["system"] == system and msgs[:len(chat["messages"])] == chat["messages"] \
                and all(m["role"] == "user" for m in msgs[len(chat["messages"]):]):
            return {"from": chat["checkpoint"], "messages": msgs[len(chat["messages"]):]}
        raise Refused(f"tab {tab['name']!r} holds a reply that no chat on the daemon continues (it was written by "
                      f"hand, edited since, or the daemon restarted), and /v1/chat never takes a reply back as text: "
                      f"delete the reply to ask here, or add your question as a note")

    def stop(self) -> None:
        self.stop_flag.set()

    def do_ask(self, tid: str, text: str) -> dict:
        """An ask has no waiter (the request was answered 202), so every way it ends tells the page: ask_done, or
        aborted with why, including a refusal before anything streamed."""
        try:
            tab = self.tab(tid)
            if len(tab["messages"]) + 2 > MAX_MESSAGES:
                raise Refused(f"tab {tab['name']!r} is full ({MAX_MESSAGES} messages)")
            route = self.ask_route(tab)
            if self.stop_flag.is_set():
                raise RuntimeError("stopped before it started; the question was dropped")
        except BaseException as e:
            self.emit({"type": "aborted", "tab": tid, "reason": str(e)})
            raise
        return self.ask_in(tab, tid, text, route)

    def ask_in(self, tab: dict, tid: str, text: str, route: dict) -> dict:
        body = {**route, "messages": route["messages"] + [{"role": "user", "content": text}],
                "max_tokens": ASK_TOKENS, "timeout_ms": ASK_TIMEOUT_MS, "stream": True}
        with self.lock:
            tab["messages"].append({"role": "user", "content": text})
        n0 = len(tab["messages"]) - 1
        self.emit({"type": "tab", "tab": self.tab_view(tab)})
        self.set_status("thinking", tab["name"])
        done, t0 = None, time.perf_counter()
        try:
            try:
                for name, data in self.daemon.stream("/v1/chat", body, self.stop_flag):
                    if name == "token":
                        self.emit({"type": "token", "tab": tid, "piece": data.get("text", "")})
                    elif name == "done":
                        done = data
                    elif name == "error":
                        err = data.get("error", {})
                        raise ApiError(data.get("status", 500), err.get("message", json.dumps(data)))
            except ApiError as e:
                if e.status == 404 and "from" in route:
                    tab["chat"] = None
                    raise Refused(f"the daemon no longer holds the chat behind tab {tab['name']!r}'s replies "
                                  f"(evicted or restarted), and /v1/chat never takes a reply back as text: delete "
                                  f"the reply to ask here") from None
                if e.status == 400:
                    raise _refused_by_daemon(e, "the question") from None
                raise
            if done is None:
                raise RuntimeError("the reply was stopped; the question was dropped")
            if done["finish_reason"] != "stop" or done.get("checkpoint") is None or done.get("content") is None:
                raise RuntimeError(f"the reply ended by {done['finish_reason']} after {done['completion_tokens']} "
                                   f"tokens; the question was dropped")
            reply = {"role": "assistant", "content": done["content"]}
            if done.get("thinking") is not None:
                reply["thinking"] = done["thinking"]
            with self.lock:
                tab["messages"].append(reply)
            self.pin(tab)
            system, msgs = self.head(tab)
            tab["chat"] = {"checkpoint": done["checkpoint"], "system": system, "messages": msgs}
        except BaseException as e:
            with self.lock:
                del tab["messages"][n0:]
            self.emit({"type": "tab", "tab": self.tab_view(tab)})
            self.emit({"type": "aborted", "tab": tid, "reason": str(e)})
            raise
        s = time.perf_counter() - t0
        self.emit({"type": "tab", "tab": self.tab_view(tab)})
        self.emit({"type": "ask_done", "tab": tid, "tokens": done["completion_tokens"], "s": round(s, 2),
                   "continued": "from" in route})
        if tab["include"]:
            self.changed("ask", tab)
        return {"tokens": done["completion_tokens"]}

    # --- reads --------------------------------------------------------------------

    def included(self) -> list[dict]:
        tabs = [t for t in self.tabs if t["include"]]
        if not tabs:
            raise Refused("no tab is included: include at least one context to decide")
        return tabs

    def decision(self, spec: dict, action: str, ids: list[str]) -> tuple[dict, float]:
        body = {"state": fenced(action), "spec_id": spec["id"], "ask": [spec["field"]],
                "contexts": [{"id": c} for c in ids], "pool": dict(self.pool), "timeout_ms": READ_TIMEOUT_MS}
        t0 = time.perf_counter()
        r = self.daemon.post("/council/v1/decisions", body)
        return r, (time.perf_counter() - t0) * 1000

    def read_one(self, action: str, tabs: list[dict]) -> dict:
        """One decision across these tabs for this action. A 404 for a lost spec posts the specs again, one for a
        lost context puts every tab again, each at most once: a restart loses both, and the daemon names the spec
        first, so a read after a restart can need both before it answers. A read of a head the page did not put
        puts those tabs again, once, and reads again."""
        recovered = set()
        while True:
            spec = self.specs[self.spec]
            ids = [t["context"] for t in tabs]
            try:
                r, wall = self.decision(spec, action, ids)
                return self.record(r, wall, spec, tabs, ids)
            except Drifted as e:
                if "drift" in recovered:
                    raise RuntimeError(f"{e}, again after putting them again") from None
                recovered.add("drift")
                names = [t["name"] for t in e.tabs]
                self.emit({"type": "repinned", "tabs": len(e.tabs),
                           "message": f"the daemon read {names} at a head this page did not put (a PUT it took whose "
                                      f"answer was lost?): PUT again"})
                for tab in e.tabs:
                    self.pin(tab)
                    self.emit({"type": "tab", "tab": self.tab_view(tab)})
            except ApiError as e:
                if e.status == 400:
                    raise _refused_by_daemon(e, "the decision (the action, or the pool over these reads)") from None
                if e.status != 404:
                    raise
                if e.param == "spec_id" and "specs" not in recovered:
                    recovered.add("specs")
                    self.upload_specs()
                    self.emit({"type": "repinned", "tabs": 0,
                               "message": "the daemon no longer held the specs (a restart?): posted them again"})
                elif e.param == "id" and "contexts" not in recovered:  # council_api.rs context_gone
                    recovered.add("contexts")
                    self.repin_all()
                else:
                    raise

    def record(self, r: dict, wall: float, spec: dict, tabs: list[dict], ids: list[str]) -> dict:
        """The stored read: each context's raw numbers, checked against the daemon's pool."""
        field, options = spec["field"], spec["options"]
        if [x["context"] for x in r["reads"]] != ids:
            raise RuntimeError(f"/council/v1/decisions answered reads for {[x['context'] for x in r['reads']]}, "
                               f"asked {ids}")
        # lfm2d names the context's head a read started from as its snapshot (council_decision.rs: the contract's
        # spec layer is not held there), so it says what was read, whatever the page believes it put
        drifted = [t for t, x in zip(tabs, r["reads"]) if x["snapshot"] != t["head"]]
        if drifted:
            raise Drifted(drifted)
        per = []
        for t, x in zip(tabs, r["reads"]):
            a = x["answers"].get(field)
            if a is None or a["type"] != "choice" or set(a["logprobs"]) != set(options):
                raise RuntimeError(f"a read answered {sorted(x['answers'])} (logprobs {sorted((a or {}).get('logprobs', []))}), "
                                   f"asked {field!r} over {options}")
            logprobs = [a["logprobs"][o] for o in options]
            if not _close(a["mass"], _read_mass(logprobs)):
                raise RuntimeError(f"a read's mass is {a['mass']}; its own logprobs give {_read_mass(logprobs)}")
            mass = math.exp(a["mass"])
            wire = [math.exp(v - a["mass"]) for v in logprobs]
            if not all(_close(wire[i], a["probabilities"][o]) for i, o in enumerate(options)):
                raise RuntimeError(f"a read's probabilities are {a['probabilities']}; its own logprobs give {wire}")
            probs = council_pool.option_probs(logprobs)  # renormalized over the options, as the pool sees them
            described = [{"field": q, "value": x["described"][q]} for q in spec["describe"]
                         if q in x.get("described", {})]
            per.append({"tab": t["id"], "name": t["name"], "color": t["color"], "logprobs": logprobs, "mass": mass,
                        "probs": probs, "verdict": options[council_pool.argmax(probs)], "described": described,
                        "context_tokens": x.get("tokens"), "rendered_sha256": x["rendered_sha256"],
                        "ms": {"total": x.get("ms")}})
        got = r["answers"].get(field)
        if got is None or got["type"] != "choice" or set(got["probabilities"]) != set(options):
            raise RuntimeError(f"/council/v1/decisions pooled {sorted(r['answers'])}, asked {field!r} over {options}")
        if {k: r["pool"][k] for k in ("method", "weights")} != self.pool:
            raise RuntimeError(f"/council/v1/decisions pooled under {r['pool']}, asked {self.pool}")
        mine = council_pool.pool([p["logprobs"] for p in per], [p["mass"] for p in per], self.pool["method"],
                                 self.pool["weights"])
        # every number the daemon pooled must be ours, to float rounding
        theirs_loo = got.get("leave_one_out") or {}
        problems = []
        if not all(_close(mine["probs"][i], got["probabilities"][o]) for i, o in enumerate(options)):
            problems.append("probabilities")
        if got["agree"] != mine["agree"]:
            problems.append("agree")
        if not _close(got["spread"], mine["spread"]):
            problems.append("spread")
        if not _all_close(r["pool"]["normalized"].get(field, []), mine["weights"]):
            problems.append("weights")
        want_loo = {c: row for c, row in zip(ids, mine["leave_one_out"])}
        if set(theirs_loo) != set(want_loo) or any(
                (theirs_loo[c] is None) != (row is None)
                or (row is not None and not all(_close(row[i], theirs_loo[c][o]) for i, o in enumerate(options)))
                for c, row in want_loo.items()):
            problems.append("leave_one_out")
        if problems:
            raise RuntimeError(f"the daemon pooled {got} / {r['pool']}; council_pool.py gives {mine} from the same "
                               f"reads, differing in {problems}: the pooling is not the one this page explains")
        read = {"spec": spec["file"], "spec_id": spec["id"], "field": field, "options": options,
                "tabs": [t["id"] for t in tabs], "contexts": ids, "heads": [t["head"] for t in tabs], "per": per,
                "ms": round(wall, 1), "queue_ms": r.get("queue_ms"), "usage": r.get("usage"), "t": time.time()}
        return self.derive(read, self.pool)

    def derive(self, read: dict, pool: dict) -> dict:
        """The pooled verdict under `pool`, both pools' stars, and leave-one-out, from the stored reads. ValueError
        where `pool` is undefined for them."""
        lg, ms, options = [p["logprobs"] for p in read["per"]], [p["mass"] for p in read["per"]], read["options"]
        m, w = pool["method"], pool["weights"]
        pooled = council_pool.pool(lg, ms, m, w)
        read["pool"] = dict(pool)
        read["pooled"] = {"probs": pooled["probs"], "weights": pooled["weights"], "agree": pooled["agree"],
                          "spread": pooled["spread"],
                          "verdict": options[council_pool.argmax(pooled["probs"])]}
        read["stars"] = {m: pooled["probs"]}
        for k in council_pool.METHODS:
            if k != m:
                try:  # the method not asked can be undefined where the asked one is not (log-linear, all vetoed)
                    read["stars"][k] = council_pool.pool(lg, ms, k, w)["probs"]
                except ValueError:
                    read["stars"][k] = None
        read["loo"] = [{"tab": p["tab"], "probs": x,
                        "verdict": None if x is None else options[council_pool.argmax(x)]}
                       for p, x in zip(read["per"], pooled["leave_one_out"])]
        return read

    def rank_loud(self) -> None:
        """Set every decision's loud_p, loud_rank and loud_elevated (LOUD_RULE) from its current pooled read, and send
        the page all of them. The loudest option is the menu's last; loud_p is its pooled probability."""
        with self.lock:
            ds = self.decisions
            for d in ds:
                p = d["read"]["pooled"]["probs"][-1]
                if not (isinstance(p, float) and math.isfinite(p)):
                    raise RuntimeError(f"decision {d['id']}'s pooled probability of {d['read']['options'][-1]!r} is "
                                       f"{p!r}: nothing to rank")
                d["loud_p"] = p
            n = len(ds)
            order = sorted(range(n), key=lambda k: (-ds[k]["loud_p"], ds[k]["n"]))
            top = math.ceil(n / LOUD_RULE["quarter"])
            median = statistics.median(d["loud_p"] for d in ds) if ds else 0.0
            for rank, k in enumerate(order, 1):
                d = ds[k]
                d["loud_rank"] = rank
                # loud_p > 0 too: under a median of 0, "twice the median" would otherwise hold for nothing at all
                d["loud_elevated"] = (n >= LOUD_RULE["min_n"] and rank <= top
                                      and d["loud_p"] >= LOUD_RULE["median_x"] * median and d["loud_p"] > 0)
            view = [{"id": ds[k]["id"], **{f: ds[k][f] for f in ("loud_p", "loud_rank", "loud_elevated")}}
                    for k in order]
        self.emit({"type": "loud", "n": n, "median": median, "decisions": view})

    def do_decide(self, action: str, source: str = "typed", rules: str | None = None, note: str | None = None) -> dict:
        self.set_status("reading", action[:60])
        read = self.read_one(action, self.included())
        n = next(self.dec_seq)
        d = {"id": f"d{n}", "n": n, "action": action, "source": source, "rules": rules, "note": note, "t": time.time(),
             "read": read, "history": [], "flip": None, "replay": None}
        with self.lock:
            self.decisions.append(d)
            del self.decisions[:-MAX_DECISIONS]
        self.rank_loud()
        self.emit({"type": "decision", "decision": d})
        return d

    def do_backfill(self) -> dict:
        with self.lock:
            causes, self.causes = self.causes, []
        if not causes:
            return {"reread": 0}  # an earlier backfill already covered these changes
        recent = self.decisions[-BACKFILL_K:]
        try:
            tabs = self.included()
        except Refused as e:
            self.emit({"type": "backfill", "cause": causes, "reread": 0, "flips": [], "decisions": [], "note": str(e)})
            return {"reread": 0}
        held = [(t["context"], t["head"]) for t in tabs]
        todo = [d for d in recent if list(zip(d["read"]["contexts"], d["read"]["heads"])) != held
                or d["read"]["spec"] != self.spec]
        if not todo:
            self.emit({"type": "backfill", "cause": causes, "reread": 0, "flips": [], "decisions": []})
            return {"reread": 0}
        reads = []
        try:
            for k, d in enumerate(todo):
                self.set_status("backfill", f"re-reading {k + 1} of {len(todo)} under the new contexts")
                reads.append(self.read_one(d["action"], tabs))
        except BaseException:
            with self.lock:
                self.causes[:0] = causes  # the next change's backfill re-reads these too
            raise
        flips = []
        with self.lock:
            for d, new in zip(todo, reads):
                old = d["read"]
                d["history"] = (d["history"] + [old])[-HISTORY:]
                d["read"] = new
                a, b = old["pooled"]["verdict"], new["pooled"]["verdict"]
                d["flip"] = {"from": a, "to": b, "cause": causes[-1]} if a != b else None
                d["replay"] = None
                if a != b:
                    flips.append({"id": d["id"], "from": a, "to": b})
        self.rank_loud()
        self.emit({"type": "backfill", "cause": causes, "reread": len(todo), "flips": flips,
                   "decisions": [json.loads(json.dumps(d)) for d in todo]})
        return {"reread": len(todo), "flips": flips}

    def do_pool(self, method: str, weights: str) -> dict:
        new = {"method": method, "weights": weights}
        with self.lock:
            if new == self.pool:
                return {"pool": dict(self.pool)}
            # every stored read under the new pool before any of it is kept: one it cannot pool refuses the switch
            redone = []
            for d in self.decisions:
                try:
                    redone.append((d, self.derive(copy.deepcopy(d["read"]), new),
                                   [self.derive(copy.deepcopy(h), new) for h in d["history"]]))
                except ValueError as e:
                    raise Refused(f"decision {d['id']} ({d['action']!r}) cannot be pooled {method} with {weights} "
                                  f"weights ({e}); the pool stays {self.pool['method']} with "
                                  f"{self.pool['weights']} weights") from None
            self.pool = new
            flips = []
            cause = {"what": "pool", "pool": dict(self.pool), "t": time.time()}
            for d, read, history in redone:
                a = d["read"]["pooled"]["verdict"]
                d["read"], d["history"] = read, history
                b = d["read"]["pooled"]["verdict"]
                d["flip"] = {"from": a, "to": b, "cause": cause} if a != b else None
                if a != b:
                    flips.append({"id": d["id"], "from": a, "to": b})
            self.rank_loud()
            changed = json.loads(json.dumps(self.decisions))
        self.emit({"type": "backfill", "cause": [cause], "reread": 0, "flips": flips, "decisions": changed,
                   "pool": dict(self.pool)})
        return {"pool": dict(self.pool), "flips": flips}

    def do_spec(self, spec: str) -> dict:
        if spec not in self.specs:
            raise Refused(f"spec must be one of {list(self.specs)}")
        if spec == self.spec:
            return {"spec": spec}
        with self.lock:
            self.spec = spec
        self.emit({"type": "spec", "spec": spec})
        self.changed("spec", spec=spec)
        return {"spec": spec}

    def do_replay(self, did: str) -> dict:
        d = next((x for x in self.decisions if x["id"] == did), None)
        if d is None:
            raise Missing(f"no decision {did!r}")
        tabs = self.included()
        if list(zip(d["read"]["contexts"], d["read"]["heads"])) != [(t["context"], t["head"]) for t in tabs] \
                or d["read"]["spec"] != self.spec:
            raise Refused("its contexts or spec changed since it was read (it is older than the backfill reaches): "
                          "replay re-reads under the same contexts and spec only")
        self.set_status("reading", f"replaying {did}")
        new = self.read_one(d["action"], tabs)
        diffs = [p["name"] for p, q in zip(d["read"]["per"], new["per"])
                 if (p["logprobs"], p["mass"], p["described"], p["rendered_sha256"])
                 != (q["logprobs"], q["mass"], q["described"], q["rendered_sha256"])]
        match = not diffs
        with self.lock:
            d["replay"] = {"match": match, "t": time.time(), "diffs": diffs, "ms": new["ms"]}
        self.rank_loud()  # the stored read stands, so the ranks do too; sent again so the page is never behind
        self.emit({"type": "replay", "id": did, "match": match, "diffs": diffs, "ms": new["ms"]})
        if not match:
            self.emit({"type": "error", "job": "replay", "message": f"replay of {did} differs from its read in "
                       f"{diffs}: a repeated read is identical (invariant 17), so this is a bug"})
        return {"id": did, "match": match, "diffs": diffs}

    def do_reset(self) -> dict:
        old = [t["context"] for t in self.tabs]
        with self.lock:
            self.tabs, self.decisions, self.causes = [], [], []
            self.spec = self.spec_files[0]
        try:
            self.seed_tabs()
        finally:  # a seed that fails leaves the tabs seeded so far, and the old ones are no one's either way
            self.rank_loud()
            self.emit({"type": "reset", **self.snapshot()})
            for c in old:
                self.free(c)  # the seed tabs are new UUIDs: the old ones are no one's
        return {"tabs": len(self.tabs)}

    # --- views --------------------------------------------------------------------

    def snapshot(self) -> dict:
        with self.lock:
            return json.loads(json.dumps({
                "status": self.status, "tabs": [self.tab_view(t) for t in self.tabs], "decisions": self.decisions,
                "pool": self.pool, "spec": self.spec, "options": self.options,
                "specs": [{k: s[k] for k in ("file", "id", "field", "options", "describe", "input_label")}
                          for s in self.specs.values()],
                "palette": PALETTE, "backfill_k": BACKFILL_K, "max_tabs": self.max_tabs, "max_decisions": MAX_DECISIONS,
                "min_mass": MIN_MASS, "loud_rule": LOUD_RULE, "model": self.model, "about": scenario.ABOUT, "trust": TRUST,
                "synthetic": scenario.SYNTHETIC, "actions": self.actions}))

    # --- HTTP ---------------------------------------------------------------------

    def handle(self, h, method: str) -> None:
        """Answer a request under /council/api for demo/web/server.py's handler `h`."""
        self.start()
        path = h.path.split("?")[0][len("/council/api"):]
        if method == "GET":
            if path == "/state":
                return send_json(h, self.snapshot())
            if path == "/events":
                return self.events(h)
            return send_json(h, {"error": {"message": "not found"}}, HTTPStatus.NOT_FOUND)
        try:
            b = read_body(h)
            if not self.booted and path != "/stop":
                return send_json(h, {"error": {"message": f"the council is {self.status['phase']}: "
                                                          f"{self.status['detail']}"}},
                                 HTTPStatus.SERVICE_UNAVAILABLE)
            self.route(h, path, b)
        except Refused as e:
            send_json(h, {"error": {"message": str(e)}}, HTTPStatus.BAD_REQUEST)
        except Missing as e:
            send_json(h, {"error": {"message": str(e)}}, HTTPStatus.NOT_FOUND)
        except (ApiError, Unreachable) as e:
            send_json(h, {"error": {"message": f"the daemon: {e}"}}, HTTPStatus.BAD_GATEWAY)
        except Exception as e:  # noqa: BLE001 -- a failed job: say what, keep serving
            send_json(h, {"error": {"message": f"{type(e).__name__}: {e}"}}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def events(self, h) -> None:
        q = self.subscribe()
        try:
            h.send_response(HTTPStatus.OK)
            h.send_header("content-type", "text/event-stream")
            h.send_header("cache-control", "no-store")
            h.end_headers()
            h.close_connection = True
            h.wfile.write(f"data: {json.dumps({'type': 'hello', **self.snapshot()})}\n\n".encode())
            h.wfile.flush()
            while not q.dropped:
                try:
                    ev = q.get(timeout=15)
                    h.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                except queue.Empty:
                    h.wfile.write(b": keepalive\n\n")
                h.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.unsubscribe(q)

    def run_job(self, h, kind: str, **args) -> None:
        job = self.submit(kind, wait=True, **args)
        if job.error is not None:
            raise job.error
        send_json(h, job.result)

    def route(self, h, path: str, b: dict) -> None:
        if path == "/decide":
            _fields(b, {"action"})
            self.run_job(h, "decide", action=_check_action(b["action"]))
        elif path == "/scenario":
            _fields(b, set())
            for a in self.actions:
                self.submit("decide", action=_check_action(a["text"]), source="scenario", rules=a.get("rules"),
                            note=a.get("note"))
            send_json(h, {"queued": len(self.actions)}, HTTPStatus.ACCEPTED)
        elif path == "/pool":
            _fields(b, set(), {"method", "weights"})  # either left out keeps its current setting
            method, weights = b.get("method", self.pool["method"]), b.get("weights", self.pool["weights"])
            if method not in council_pool.METHODS:
                raise Refused(f"pool method must be one of {list(council_pool.METHODS)}")
            if weights not in council_pool.WEIGHT_NAMES:
                raise Refused(f"pool weights must be one of {list(council_pool.WEIGHT_NAMES)}")
            self.run_job(h, "pool", method=method, weights=weights)
        elif path == "/spec":
            _fields(b, {"spec"})
            if b["spec"] not in self.specs:
                raise Refused(f"spec must be one of {list(self.specs)}")
            self.run_job(h, "spec", spec=b["spec"])
        elif path == "/tabs":
            _fields(b, {"name"}, {"preamble", "color"})
            args = {"name": _check_name(b["name"])}
            if "preamble" in b:
                args["preamble"] = "" if b["preamble"] == "" else _check_text(b["preamble"], "preamble")
            if "color" in b:
                args["color"] = _color(b["color"])
            self.run_job(h, "tab_add", **args)
        elif path == "/stop":
            self.stop()
            send_json(h, {"ok": True})
        elif path == "/reset":
            _fields(b, set())
            self.run_job(h, "reset")
        elif m := _REPLAY.fullmatch(path):
            _fields(b, set())
            self.run_job(h, "replay", did=m.group(1))
        elif m := _TAB.fullmatch(path):
            self.tab_route(h, m, b)
        else:
            send_json(h, {"error": {"message": "not found"}}, HTTPStatus.NOT_FOUND)

    def tab_route(self, h, m: re.Match, b: dict) -> None:
        tid, tail, idx, remove = m.group(1), m.group(2), m.group(3), m.group(4)
        if tail is None:
            _fields(b, set(), {"name", "preamble", "color", "include"})
            if not b:
                raise Refused("nothing to change: give name, preamble, color or include")
            kw = {}
            if "name" in b:
                kw["name"] = _check_name(b["name"])
            if "preamble" in b:
                kw["preamble"] = "" if b["preamble"] == "" else _check_text(b["preamble"], "preamble")
            if "color" in b:
                kw["color"] = _color(b["color"])
            if "include" in b:
                if not isinstance(b["include"], bool):
                    raise Refused("include must be true or false")
                kw["include"] = b["include"]
            self.run_job(h, "tab_update", tid=tid, **kw)
        elif tail == "/remove":
            _fields(b, set())
            self.run_job(h, "tab_remove", tid=tid)
        elif tail == "/ask":
            _fields(b, {"text"})
            text = _check_text(b["text"], "text", MAX_ACTION)
            tab = self.tab(tid)  # an unknown tab is a 404 now, not an error event later
            if len(tab["messages"]) + 2 > MAX_MESSAGES:
                raise Refused(f"tab {tab['name']!r} is full ({MAX_MESSAGES} messages)")
            self.ask_route(tab)  # a tab whose reply can't be continued is a 400 now
            self.stop_flag.clear()  # a stop from here on stops this ask, queued or streaming
            job = self.submit("ask", tid=tid, text=text)
            if job.error is not None:
                raise job.error
            send_json(h, {"queued": "ask"}, HTTPStatus.ACCEPTED)
        elif idx is None:
            _fields(b, {"role", "content"})
            if b["role"] not in ("user", "assistant"):
                raise Refused("role must be user or assistant")
            self.run_job(h, "message_add", tid=tid, role=b["role"], content=_check_text(b["content"], "content"))
        elif remove:
            _fields(b, set())
            self.run_job(h, "message_remove", tid=tid, index=_msg_index(idx))
        else:
            _fields(b, {"content"})
            self.run_job(h, "message_edit", tid=tid, index=_msg_index(idx), content=_check_text(b["content"], "content"))


_TAB = re.compile(r"/tabs/([^/]+)(/remove|/ask|/messages(?:/([^/]+)(/remove)?)?)?")
_REPLAY = re.compile(r"/decisions/([^/]+)/replay")


def _color(v) -> str:
    if not isinstance(v, str) or not _COLOR.fullmatch(v):
        raise Refused("color must be #rrggbb")
    return v.lower()


def _msg_index(s: str) -> int:
    if not (s.isascii() and s.isdigit()) or len(s) > 6:  # "²".isdigit() is True
        raise Refused(f"message index must be a number, got {s!r}")
    return int(s)


def send_json(h, obj, status=HTTPStatus.OK) -> None:
    body = json.dumps(obj).encode()
    h.send_response(status)
    h.send_header("content-type", "application/json")
    h.send_header("content-length", str(len(body)))
    h.send_header("cache-control", "no-store")
    h.end_headers()
    h.wfile.write(body)


def read_body(h) -> dict:
    cl = (h.headers.get("content-length") or "0").strip()
    if not (cl.isascii() and cl.isdigit()):
        h.close_connection = True
        raise Refused(f"Content-Length must be a non-negative integer, got {cl[:20]!r}")
    n = int(cl)
    if n > MAX_BODY:
        h.close_connection = True  # the body stays unread
        raise Refused(f"the body is too large ({n} bytes, at most {MAX_BODY})")
    raw = h.rfile.read(n) if n else b"{}"
    try:
        b = json.loads(raw or b"{}")
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise Refused(f"the body is not JSON: {e}") from None
    if not isinstance(b, dict):
        raise Refused("the body must be a JSON object")
    return b

