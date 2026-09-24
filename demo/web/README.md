# Web demos

Browser pages that play themselves against a running lfm2d, sized 9:16 for
recording a tab. Python 3.10+, standard library only.

```sh
python3 demo/web/server.py --upstream 'http://<lfm2d host>:8088'
# demos on http://<tailnet ip>:8765/
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
