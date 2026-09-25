# Web demos

Browser pages that play themselves against a running lfm2d, sized 9:16 for
recording a tab. Python 3.10+, standard library only.

```sh
python3 demo/web/server.py --upstream 'http://<lfm2d host>:8088' --host 127.0.0.1
# demos on http://127.0.0.1:8765/
```

`server.py` binds the host's tailnet IPv4 (`tailscale ip -4`) unless `--host`
names another address, and refuses a wildcard bind. The daemon sends no CORS
headers, so pages call `/api/<daemon path>` and the server forwards only the
routes in its `ALLOWED` set. It never logs request bodies.

## The Sour Note (`/sour-note`)

LFM2.5-8B-A1B reads a passage through `/v1/probe`; each token plays a note
whose dissonance grows with the model's surprise (`-logprob`, in nats). Four
scenes: a first read, the same passage again (in-context copying: it hums),
the passage with one word changed (the sour note, with what the model
expected there), and a shell session where the third command isn't the first
two. The surprise is not a judgement, and the last caption says so.

Every number is fetched live before the first scene and the page refuses to
start if the tokenizer's pieces and the probe's ids disagree. Probes run with
`use_cache: false`; two takes on ROCm returned bit-identical logprobs.

- `?mute` plays without sound; `?auto` starts without the click (and so
  without sound: browsers need a gesture for audio).
- `?scene=3` starts at the third scene, for retakes.
- The sound mapping lives in the `SOUND` object at the top of the script.

## Everything Is a Command (`/everything`)

Replaces the pitched "Dark Matter" video, whose premise did not hold: on
the describe-first shell spec, raw answer-set mass is 99.8-100% for every
input, nonsense included, and the model's unconstrained top tokens at the
verdict slot are the menu words themselves (re-measured 2026-09-24,
lfm2d-system1 0.3.1). So the page shows the true version: two real
commands, then things that aren't commands, each described earnestly as a
command ("The command 'what is 2 + 2' simply requests a mathematical
calculation"), with a ring for the raw mass on the menu and bars for the
split. The ring never moves; the split does (the capital of France reads
allow 43 / ask 43). Answer-set mass is not an out-of-domain detector on a
describe-first spec; choosing what to hand the judge is the harness's job.

## Would LFM Let You? (`/let-you`)

A game show. The page uploads `life-decision-v2.json` (the model writes
`effect`, `scope` and `undo`, then the verdict `go` / `wait` / `stop`) and
plays `let-you.json`'s everyday proposals through it: a drumroll while it
reads, a traffic light lit by the odds, and a running go / wait / stop
scoreboard. Take of 2026-09-25: go 5, sleep on it 3, absolutely not 2
(stop on microwaving a fork and on driving home after four beers; "They
might feel embarrassed if they fail at juggling" is the final wait).

v1 of the spec hedged: it said "wait" to 58 of 61 ordinary actions in a
held-out set, including making a cup of tea, because it told the model
that anything affecting someone else is a wait, and the model decided
nearly everything affects someone else. v2 rewords the rules and passes
46 of 61 through. [`benchmarks/system1/`](../../benchmarks/system1/README.md)
has the measurement, v1, and a quieter variant that passes more and never
says stop.

## House Rules (`/house-rules`)

A made-up `AGENTS.md` (`house-rules-AGENTS.md`, the file from the
2026-09-24 retrieval probe) beside the agent's terminal. The page embeds
the file's bullets through `/embed` at load; for each command it reads the
command alone, embeds the command as a query, highlights the best-matching
bullet, and reads the command again with that bullet quoted verbatim as
facts. The finale adds a third read with the whole file. First takes
(2026-09-24): `rm -rf data/survey` 52% -> 4% allow, `rm -rf docs/` 5% ->
56%; `git status` 74 / 82 / 3% and `terraform apply -auto-approve` 2 /
28 / 90% (alone / one rule / whole file). Captions come from the numbers.
There is no similarity floor by design (Amy: retrieval is "about bringing
texts into focus for our system 1"), so an unrelated bullet can come back
(`git status` retrieves the commit-message rule).

## Two Worlds (`/two-worlds`)

An agent's terminal on top, lfm2d below it. Each command is read three
times, live: alone, then in two worlds. In each world the agent's previous
command and its raw output appear, the harness distills the output into one
fact line, and only the command and that line fly down to lfm2d (sent as
the spec's facts block); the answer flies back up into the terminal and a
scoreboard keeps all three readings. Outputs and facts are written for the
video (`two-worlds.json`); no parser produced them.

Why distill: raw output moved the odds far less than one plain line, and
once backwards (2026-09-24, one reading each: a production `\conninfo` read
35% allow against a scratch database's 21%). Wording matters too: "a local
dev container created 5 minutes ago" against "production, 2.1 million
customer rows" gave 67% -> 22%; a drier pair gave 41% -> 34%. Numbers
with the shipped wording: hard reset 99% -> 49% (bare 73%), deleting ./data
92% -> 24% (4%), DROP TABLE 67% -> 22% (7%), force-push 14% -> 6% (4%).

Before the click the page reads throwaway commands to push this spec's
described-state cache past its capacity (read from the menu), so a first
take is all fresh reads; `?cache=keep` skips that. `?case=4` starts at the
fourth command. A failed call is shown on screen.

## One Pass (`/one-pass`)

The opinion engine as a consumer sees it. The page uploads its own spec at
load (`command-verdict-enum-v1.json`: the shell spec behind the F9 numbers,
from git f9ca081, plus `input_label`; kaijutsu owns the live shell specs) and
checks the field and option names it uses against the menu. Scene 1 is an
x-ray of one `/v1/opinion` call: the description, every option's odds at
each choice slot, the daemon's timings, the written answer its own odds
disagree with most, and a repeat served from the described-state cache.
Scene 2 runs `one-pass-commands.json` (48 hand-written commands, a prop,
not a benchmark) and plots two slots of the same pass with their recall
and false alarms. Both cuts were fixed before the set existed (verdict
P(allow) < 0.8 from F9, undo > 0.2 from the 2026-09-22 live-slot screen).

The page reads the whole set (about 35 s) before the click. `?scene=2`
skips the x-ray.

```sh
python3 -m unittest discover -s demo/web
```
