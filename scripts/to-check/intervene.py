"""C3b: does the model use the refusal axis? Ablation and activation addition.

The direction is SFT's v_ref = mu_harmful - mu_harmless at one layer, at the
last template token (pre_gen), used for every model so that the push is the
same in all of them. alpha = 1 adds the full harmless->harmful gap.

Conditions, on a fixed stratified sample of prompts:
  none      no intervention (must reproduce the stored greedy responses)
  ablate    v_ref projected out of the residual stream at every layer and
            position (after the embeddings and after each decoder block)
  add<a>    alpha * v_ref added at the chosen layer, every position

Generations go to results/<family>/causal/generations.csv, one block per
stage x condition, saved as each block finishes; a rerun skips finished blocks.
Judging is a separate step.

    python scripts/intervene.py --family olmo2 --layer 16
"""
import argparse

import numpy as np
import pandas as pd
import torch

from overrefusal import activations, config, models, prompts, refusal
from overrefusal.groups import assign_group
from overrefusal.io import save_atomic


def sample_prompts(family: str, n: dict, seed: int = 0) -> pd.DataFrame:
    raw = pd.read_csv(config.raw_results_csv(family))
    d = raw[(raw.checkpoint == "sft__none") & ~raw.source.isin(config.EXCLUDED_SOURCES)].copy()
    d = d.drop_duplicates("prompt")
    d["group"] = assign_group(d)
    parts = []
    for g, k in n.items():
        x = d[d.group == g]
        frac = min(1.0, k / len(x))
        parts.append(x.groupby("source").sample(frac=frac, random_state=seed))
    return pd.concat(parts)[["prompt", "label", "category", "source", "group"]].reset_index(drop=True)


def resid(out, fn):
    return (fn(out[0]),) + tuple(out[1:]) if isinstance(out, tuple) else fn(out)


def find_hidden_index(model, tok, stage, ref: pd.DataFrame, layer: int) -> int:
    """Index j of output_hidden_states matching the stored activations (cos > 0.99)."""
    column = activations.col(layer, "pre_gen")
    best = {}
    for _, r in ref.head(4).iterrows():
        text = prompts.build_prompt(tok, r.prompt, stage)
        ids = tok(text, return_tensors="pt", add_special_tokens=config.ADD_SPECIAL_TOKENS).to(model.device)
        ids.pop("token_type_ids", None)
        with torch.inference_mode():
            hs = model(**ids, output_hidden_states=True).hidden_states
        stored = np.asarray(r[column], dtype=np.float32)
        for j in (layer - 1, layer, layer + 1):
            h = hs[j][0, -1].float().cpu().numpy()
            best.setdefault(j, []).append(float(h @ stored / (np.linalg.norm(h) * np.linalg.norm(stored))))
    means = {j: float(np.mean(v)) for j, v in best.items()}
    j = max(means, key=means.get)
    print("cosine with stored activations by hidden_states index:", {k: round(v, 4) for k, v in means.items()})
    assert means[j] > 0.99, "no hidden_states index matches the stored activations: check the extraction"
    return j


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", required=True, choices=["olmo2", "olmo3"])
    p.add_argument("--layer", type=int, default=16)
    p.add_argument("--stages", nargs="+", default=["sft", "dpo"])
    p.add_argument("--alphas", nargs="+", type=float, default=[-0.5, 0.25, 0.5, 1.0])
    p.add_argument("--n-pseudo", type=int, default=300)
    p.add_argument("--n-harmful", type=int, default=150)
    p.add_argument("--n-harmless", type=int, default=150)
    p.add_argument("--batch", type=int, default=config.GENERATION.batch_size)
    args = p.parse_args()

    out_path = config.results_dir(args.family) / "causal" / "generations.csv"
    out_path.parent.mkdir(exist_ok=True)
    done = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()

    sample = sample_prompts(args.family, {"pseudo_harm": args.n_pseudo, "harmful": args.n_harmful,
                                          "harmless": args.n_harmless})
    print("sample:", sample.group.value_counts().to_dict())

    # SFT's direction, from the stored activations
    acts = activations.load(args.family, [args.layer], ["pre_gen"], ["sft__none"])
    acts = acts.drop_duplicates("prompt")
    column = activations.col(args.layer, "pre_gen")
    mu = {g: activations.matrix(acts[acts.group == g], column).mean(0) for g in ["harmless", "harmful"]}
    vref = mu["harmful"] - mu["harmless"]
    print(f"|v_ref| = {np.linalg.norm(vref):.2f}")

    raw = pd.read_csv(config.raw_results_csv(args.family))
    max_new, max_prompt = config.GENERATION.max_new_tokens, config.GENERATION.max_prompt_tokens
    conditions = ["none", "ablate"] + [f"add{a:g}" for a in args.alphas]

    for stage in args.stages:
        model, tok = models.load(args.family, stage)
        j = find_hidden_index(model, tok, stage, acts[acts.group == "harmful"], args.layer)
        layers, emb = model.model.layers, model.model.embed_tokens
        v = torch.tensor(vref, dtype=model.dtype, device=model.device)
        u = v / v.norm()
        texts = [prompts.build_prompt(tok, m, stage) for m in sample.prompt]

        for cond in conditions:
            if len(done) and ((done.stage == stage) & (done.condition == cond)).any():
                print(f"{stage} {cond}: already done"); continue
            handles = []
            if cond == "ablate":
                f = lambda mod, inp, out: resid(out, lambda h: h - (h @ u).unsqueeze(-1) * u)
                handles = [emb.register_forward_hook(f)] + [l.register_forward_hook(f) for l in layers]
            elif cond.startswith("add"):
                a = float(cond[3:])
                g = lambda mod, inp, out, a=a: resid(out, lambda h: h + a * v)
                handles = [layers[j - 1].register_forward_hook(g)]   # hidden_states[j] = output of block j-1
            try:
                responses = []
                for i in range(0, len(texts), args.batch):
                    responses += models.generate(model, tok, texts[i:i + args.batch], max_new, max_prompt)
            finally:
                for h in handles:
                    h.remove()
            block = sample.assign(stage=stage, condition=cond,
                                  response=[prompts.clean_response(r, stage) for r in responses])
            block["keyword_refusal"] = refusal.keyword_refusal(block.response).astype(int).values
            done = pd.concat([done, block], ignore_index=True)
            save_atomic(done, out_path)
            if cond == "none":
                stored = block.prompt.map(raw[raw.checkpoint == f"{stage}__none"]
                                          .drop_duplicates("prompt").set_index("prompt").response).fillna("")
                new_r, old_r = block.response.str.strip(), stored.str.strip()
                exact = (new_r == old_r).mean()
                prefix = (new_r.str[:100] == old_r.str[:100]).mean()
                same_ref = (block.keyword_refusal.values ==
                            refusal.keyword_refusal(stored).astype(int).values).mean()
                print(f"{stage}: vs stored responses: exact {exact:.3f}, first 100 chars {prefix:.3f}, "
                      f"same keyword refusal {same_ref:.3f}")
                assert prefix > 0.9 and same_ref > 0.95, \
                    "the unmodified condition diverges early from the main run: check generation settings"
            print(f"{stage} {cond}: keyword refusal by group",
                  block.groupby("group").keyword_refusal.mean().round(3).to_dict())
        models.unload(model)


if __name__ == "__main__":
    main()
