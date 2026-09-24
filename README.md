# Over-refusal across post-training

Where in the OLMo post-training pipeline (base → SFT → DPO → final) does a model
start refusing safe prompts that merely *look* harmful, and what happens inside
the model when it does?

Two model families, OLMo 2 and OLMo 3, chosen with `--family` on every script.

## Questions and the script that answers each

| | question | script | output |
|---|---|---|---|
| Q1 | How does refusal change across stages? | `behavior.py` | `behavior_by_group.csv`, `behavior_by_source.csv` |
| Q2 | Where do pseudo-harmful prompts sit on the refusal axis, token by token? | `axis.py` | `geometry/axis.csv` |
| Q3 | What separates the pseudo-harmful prompts the model refuses from the ones it answers? | `axis.py` (the `cos_vbeh_*` and `t_refused/t_answered` columns) | same |
| Q4 | Which stage moves the representations? | `drift.py` | `geometry/drift.csv` |
| Q5 | What does a linear read-out recover, beyond dataset identity? | `probe.py` | `geometry/probes.csv` |

Checks prompt by prompt: `inspect_cell.py` (permutation test, length
confound, the texts at the extremes, SFT→DPO flips).

Figures: `fig_axis.py` (measures across layers and positions, from `axis.csv`),
`fig_scatter.py` (every prompt as a point, refusal plane or PCA).

## Pipeline

```
generate.py   prompts → responses                       results/<family>/raw_results.csv
judge.py      responses → coherence, GA, PD             same file, new columns
extract.py    responses → activations on the Hub        <repo>/data/<family>/<checkpoint>/
then the experiments above
```

```bash
pip install -e ".[dev]"
python scripts/behavior.py --family olmo2
python scripts/axis.py --family olmo2 --per-source --bootstrap 300
python scripts/fig_axis.py --family olmo2
pytest
```

## Definitions (each lives in exactly one module)

**Groups** (`groups.py`). *harmful*: label 1, plus XSTest's `contrast_*`
prompts, which XSTest ships as safe but are not. *pseudo_harm*: safe prompts
built to look harmful (OR-Bench, FalseReject, the rest of XSTest).
*harmless*: every other safe prompt (Alpaca, ToxicChat and WildGuard label 0).
BeaverTails is excluded everywhere (label noise, see `config.py`).

**Refusal** (`refusal.py`). From the judge (gpt-oss-120b; a second judge
can be run with `judge.py --tag` and compared with `judge_agreement.py`): answered = GA ≥ 2 and PD = 0,
refused otherwise. Incoherent responses are neither and are left out of rates;
their share is always printed, because it is about 40% for the base model.
For the base model "refused" mostly means "did not really answer": read the
PD rate next to it for explicit refusals. `response_type` splits "refused"
into hard refusal, answer with distancing and partial answer: DPO mostly
turns hard refusals into answers with distancing. The keyword detector is
kept only as a cross-check column.

**Base model prompt** (`prompts.py`). The base model has no chat template; it
gets the minimal frame `User: {prompt}\nAssistant:` (`config.BASE_FRAME`) in
both generation and extraction, and its answer is cut at the first
`\nUser:` it invents. Without a frame it continues the text instead of
answering, and refusal is not defined.

**Positions** (`config.py`, `activations.py`). `last_prompt` is the last token
of the user's text; `post_instr_0..k` are the template tokens after it (for
the base model, the three tokens of `\nAssistant:`, which are not the chat
ones: compare base and chat only at last_prompt, pre_gen, first_gen); `pre_gen` is an alias for the
last of them, where the model decides; `first_gen` is the first generated
token, whose state already encodes the chosen word, so it is read *after* the
decision.

**Geometry** (`geometry.py`). All measures are functions of five centroids,
so confidence intervals come from a stratified bootstrap of those centroids.
The module docstring maps each measure to its old name.

## Known limits

- OLMo 3's chat template inserts a default system prompt when none is given;
  OLMo 2's does not. OLMo 2 also splits `<|assistant|>` into five tokens
  (see `config.TEMPLATE_TOKENS`); the two `|` positions carry little signal.

- The base model sees a minimal dialogue frame, every other stage its chat
  template. Base vs SFT therefore mixes training with formatting; SFT vs DPO
  vs final does not.
- Probe accuracy on groups can come from dataset identity; read
  `loso_accuracy` before `cv_accuracy`.
- Raw centroid cosines across checkpoints are close to 1 by construction;
  `drift.py` also reports the cosines of the directions `v_ref`, `v_over`.
