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

```sh
python3 -m unittest discover -s demo/web
```
