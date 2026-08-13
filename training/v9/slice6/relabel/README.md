# blind relabel — `flagladder.jsonl`, gemini-3.5-flash, 2026-08-13

```
python3 make_relabel_input.py flagladder 1 26 > chunk_1_26.txt   # blind render
python3 score_relabel.py flagladder                              # agreement
```

## Why this file specifically

`flagladder.jsonl` came out of slice-6 batch 1 **entirely sonnet-authored** —
the local `lfm25-8b-a1b` trial failed (it labeled every rm-ladder arm
data-critical, including `rm -i`), so the agent hand-wrote all 78 rows rather
than route through a model that had just demonstrated the exact bug the slice
targets. That left the sub-slice carrying the **subtlest rule** — rule 4, which
interlocks actually count — with no cross-family check. This is that check.

Budget note: run on the `gemini` cast (Google backend, flash tier), not
OpenRouter. Google returned intermittent 503s; three of six calls failed and
were retried.

## Blind, and mechanically so

`make_relabel_input.py` renders **only the statement text**. The generator's
proposed `label`, its `note`, and its `author` tag are all stripped, and the
script refuses to emit a prompt if any row's note leaks through. A relabel that
can see the proposal is a confirmation-bias harness, not a second opinion.

Ids are global (`r001`…`r078`) and stable across chunks, so answers join back
without ever having shown a label.

**Chunking was forced, and is worth recording:** asked for all 78 at once,
gemini-3.5-flash truncated at r032 mid-token and emitted visible reasoning
("-> wait, id is r018"), burning 15.7k reasoning tokens for 654 output tokens.
At 26 rows per call the output was clean every time. Batch size is a
correctness parameter here, not just a cost one.

## Result — **71/78 = 91.0% agreement**

Relabeler harsher on 5, softer on 2.

| generator → relabeler | n |
|---|---|
| situation-normal → data-critical | 5 |
| data-critical → situation-normal | 2 |

**No label was overwritten.** Per `measure-disagreement-dont-declare-it`, a
disagreement is a candidate rubric question and the generator is not
privileged. These go to Amy.

## The disagreements cluster on ONE question

**Three of the five "harsher" rows are the same question: does an interactive
confirmation prompt count as an interlock that lowers severity?**

| row | statement | generator | relabeler |
|---|---|---|---|
| r002 | `rm -i draft.md` | situation-normal | data-critical |
| r020 | `git clean -i` | situation-normal | data-critical |
| r048 | `terraform destroy` (bare, types-to-confirm) | situation-normal | data-critical |

The generator read the prompt as "a text-evident human checkpoint". The
relabeler read it as no protection at all — you can just say yes.

Rule 4 says an interlock lowers severity **only when it prevents the harm in
question**. A confirmation prompt does not *prevent* the deletion; it confirms
it. If the relabeler is right, the whole `-i` rung of the flag ladder collapses
into data-critical and several generated labels need flipping. **This is
Amy's call, and it is worth making explicitly** — it generalises to `-i`,
`--interactive`, bare `terraform destroy`, `helm uninstall`, and every
"are you sure?" flow.

Two more genuine ones:

- **r068** `terraform state list && terraform state rm aws_instance.web` —
  generator: `state rm` only untracks, the real infrastructure survives.
  Relabeler: the state file is itself primary data, modified irreversibly.
  Both are right about different objects.
- **r069** `find /var/log -name "*.log" -mtime +90 -delete` — rule 10 derived
  data (logs rotate) vs. destruction of non-regenerated history.

## One relabeler error worth naming

**r060** `kubectl get pods -n prod | grep worker && kubectl delete pod worker-…`
— the relabeler softened it to situation-normal because "deleting an ephemeral
pod is routine, equivalent to restarting a service." That reasoning is
**excluded by rule 1**: external systems that could restore the state — "a
controller that respawns a deleted object" is the rubric's own example — do
NOT count as a backout.

So the generator is right per the rubric, and this is a rule-application slip
rather than a rubric question. Noting it because the *same* shape
(`kubectl delete pod`) was the single deliberate 2-1 split in the original v9
rubric pilot. Two independent rounds, two families reading it the same wrong
way, suggests the rubric's controller-respawn carve-out is stated less clearly
than the rest of rule 1 — worth a wording pass even though the ruling is
settled.
