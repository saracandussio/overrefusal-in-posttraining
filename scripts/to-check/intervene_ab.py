"""Exp. 3: intervene on perception (A) or on the decision (B), in one checkpoint.

Directions come from the stored activations of the same checkpoint, at pre_gen,
layer L, estimated only on prompts that are NOT in the evaluation sample:
  d_ref = mu_H - mu_S                                  perception
  d_dec = mu_P_refused - mu_P_answered, minus its      the part of behaviour that
          component along d_ref                        perception does not explain
Both are used as unit vectors scaled by |d_ref|, so alpha = 1 is a shift as large
as the harmless -> harmful gap in both conditions.

Conditions, applied at layer L, every position (same hook as intervene.py):
  none      no intervention (checked against the stored responses)
  A<a>      h - a |d_ref| u_ref     prompts look more harmless
  B<a>      h - a |d_ref| u_dec     less refusal, perception axis untouched

Generations go to results/<family>/causal/generations_ab.csv, one block per
stage x layer x condition, saved as each block finishes; a rerun skips finished
blocks. Judging is a separate step.

    python scripts/intervene_ab.py --family olmo2 --stage dpo --layer 16
"""
import argparse

import numpy as np
import pandas as pd
import torch

from overrefusal import activations, config, models, prompts, refusal
from overrefusal.io import save_atomic
from intervene import find_hidden_index, resid, sample_prompts   # same folder


def directions(family: str, stage: str, layer: int, exclude: set):
    acts = activations.load(family, [layer], ["pre_gen"], [f"{stage}__none"])
    acts = refusal.attach(acts, pd.read_csv(config.raw_results_csv(family)))
    acts = acts.drop_duplicates("prompt")
    column = activations.col(layer, "pre_gen")
    acts = acts[activations.has(acts, column) & ~acts.prompt.isin(exclude)]

    def mean(x):
        return activations.matrix(x, column).mean(0)

    H, S = acts[acts.group == "harmful"], acts[acts.group == "harmless"]
    P = acts[acts.group == "pseudo_harm"]
    hard = (P.response_type == "hard refusal").to_numpy()
    d_ref = mean(H) - mean(S)
    d_beh = mean(P[hard]) - mean(P[~hard])
    u_ref = d_ref / np.linalg.norm(d_ref)
    d_dec = d_beh - (d_beh @ u_ref) * u_ref
    print(f"{stage} L{layer}: directions from H {len(H)}, S {len(S)}, P {len(P)} ({hard.sum()} refused); "
          f"cos(d_ref, d_beh) = {d_beh @ u_ref / np.linalg.norm(d_beh):.2f}, "
          f"|d_dec| / |d_beh| = {np.linalg.norm(d_dec) / np.linalg.norm(d_beh):.2f}, "
          f"|d_ref| = {np.linalg.norm(d_ref):.2f}", flush=True)
    return H, d_ref, d_dec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", required=True, choices=["olmo2", "olmo3"])
    p.add_argument("--stage", default="dpo")
    p.add_argument("--layer", type=int, default=16)
    p.add_argument("--alphas", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0, 1.5])
    p.add_argument("--n-pseudo", type=int, default=300)
    p.add_argument("--n-harmful", type=int, default=200)
    p.add_argument("--n-harmless", type=int, default=200)
    p.add_argument("--batch", type=int, default=config.GENERATION.batch_size)
    args = p.parse_args()

    out_path = config.results_dir(args.family) / "causal" / "generations_ab.csv"
    out_path.parent.mkdir(exist_ok=True)
    done = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()

    sample = sample_prompts(args.family, {"pseudo_harm": args.n_pseudo, "harmful": args.n_harmful,
                                          "harmless": args.n_harmless})
    print("sample:", sample.group.value_counts().to_dict(), flush=True)

    H, d_ref, d_dec = directions(args.family, args.stage, args.layer, set(sample.prompt))
    scale = float(np.linalg.norm(d_ref))
    conditions = ["none"] + [f"A{a:g}" for a in args.alphas] + [f"B{a:g}" for a in args.alphas]

    raw = pd.read_csv(config.raw_results_csv(args.family))
    max_new, max_prompt = config.GENERATION.max_new_tokens, config.GENERATION.max_prompt_tokens

    model, tok = models.load(args.family, args.stage)
    j = find_hidden_index(model, tok, args.stage, H, args.layer)
    vec = {k: torch.tensor(scale * d / np.linalg.norm(d), dtype=model.dtype, device=model.device)
           for k, d in (("A", d_ref), ("B", d_dec))}
    texts = [prompts.build_prompt(tok, m, args.stage) for m in sample.prompt]

    for cond in conditions:
        if len(done) and ((done.stage == args.stage) & (done.layer == args.layer)
                          & (done.condition == cond)).any():
            print(f"{args.stage} L{args.layer} {cond}: already done"); continue
        handle = None
        if cond != "none":
            v, a = vec[cond[0]], float(cond[1:])
            g = lambda mod, inp, out, a=a, v=v: resid(out, lambda h: h - a * v)
            handle = model.model.layers[j - 1].register_forward_hook(g)   # hidden_states[j] = output of block j-1
        try:
            responses = []
            for i in range(0, len(texts), args.batch):
                responses += models.generate(model, tok, texts[i:i + args.batch], max_new, max_prompt)
        finally:
            if handle is not None:
                handle.remove()

        block = sample.assign(stage=args.stage, layer=args.layer, condition=cond,
                              response=[prompts.clean_response(r, args.stage) for r in responses])
        block["keyword_refusal"] = refusal.keyword_refusal(block.response).astype(int).values
        done = pd.concat([done, block], ignore_index=True)
        save_atomic(done, out_path)

        if cond == "none":
            stored = block.prompt.map(raw[raw.checkpoint == f"{args.stage}__none"]
                                      .drop_duplicates("prompt").set_index("prompt").response).fillna("")
            prefix = (block.response.str.strip().str[:100] == stored.str.strip().str[:100]).mean()
            same_ref = (block.keyword_refusal.values ==
                        refusal.keyword_refusal(stored).astype(int).values).mean()
            print(f"none vs stored: first 100 chars {prefix:.3f}, same keyword refusal {same_ref:.3f}")
            # text can differ with batch composition (fp16); the refusal decision must not
            assert same_ref > 0.95, \
                "the unmodified condition refuses differently from the main run: check generation settings"

        by = block.groupby("group").keyword_refusal.mean().round(3).to_dict()
        by_src = block[block.group == "pseudo_harm"].groupby("source").keyword_refusal.mean().round(3).to_dict()
        print(f"{args.stage} L{args.layer} {cond}: keyword refusal {by} | P by source {by_src}", flush=True)
    models.unload(model)


if __name__ == "__main__":
    main()
