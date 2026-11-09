"""Check one cell prompt by prompt, instead of trusting centroid summaries.

For one family, checkpoint, layer and position, every pseudo-harmful prompt
gets its own t (projection on the refusal axis: 0 harmless centroid,
1 harmful centroid). Then:

  1. t by response type: do hard refusals, answers with distancing, partial
     and full answers sit at different places, or is "refused" one blob?
  2. Permutation test: shuffle refused/answered within each source 1000
     times. Are the real gap t_refused - t_answered and cos(v_beh, v_ref)
     far outside what random labels give? (In 4096 dimensions a difference
     of two random means is not zero, so this is the honest baseline.)
  3. Confound: does t just track prompt length? Gap within length quartiles.
  4. The prompts themselves at the extremes: answered but far toward
     harmful, refused but close to harmless. Read them.
  5. With --compare (e.g. dpo__none): t in this checkpoint of the prompts
     the other one flips. Does DPO recover the prompts near the boundary?

Writes results/<family>/inspect/<checkpoint>_L<layer>_<position>.csv with
one row per pseudo-harmful prompt.

    python scripts/inspect_cell.py --family olmo2 --checkpoint sft__none --layer 16 \\
        --position pre_gen --compare dpo__none
"""
import argparse
import logging

import numpy as np
import pandas as pd

from overrefusal import activations, config, geometry, refusal
from overrefusal.io import save_atomic

rng = np.random.default_rng(0)


def per_prompt_t(X, mu_harmless, mu_harmful):
    axis = mu_harmful - mu_harmless
    return (X - mu_harmless) @ axis / (axis @ axis)


def permutation(Xp, refused, sources, v_ref, n=1000):
    """Null distribution of (gap, cos) with labels shuffled within source."""
    def stats(lab):
        v_beh = Xp[lab == 1].mean(0) - Xp[lab == 0].mean(0)
        t = Xp @ v_ref / (v_ref @ v_ref)
        return t[lab == 1].mean() - t[lab == 0].mean(), geometry.cos(v_beh, v_ref)

    null = []
    for _ in range(n):
        lab = refused.copy()
        for s in np.unique(sources):
            idx = np.flatnonzero(sources == s)
            lab[idx] = rng.permutation(lab[idx])
        null.append(stats(lab))
    return stats(refused), np.array(null)


def show(rows, title):
    print(f"\n{title}")
    for _, r in rows.iterrows():
        print(f"  t={r.t:+.2f} [{r.source}] {r.response_type}")
        print(f"     P: {r.prompt[:120]!r}")
        print(f"     R: {str(r.response)[:120]!r}")


def main():
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", required=True, choices=list(config.MODELS))
    p.add_argument("--checkpoint", default="sft__none")
    p.add_argument("--layer", type=int, default=16)
    p.add_argument("--position", default="pre_gen")
    p.add_argument("--compare", default=None, help="second checkpoint for per-prompt flips")
    p.add_argument("--permutations", type=int, default=1000)
    args = p.parse_args()

    ckpts = [args.checkpoint] + ([args.compare] if args.compare else [])
    raw = pd.read_csv(config.raw_results_csv(args.family))
    df = refusal.attach(activations.load(args.family, [args.layer], [args.position], ckpts), raw)
    column = activations.col(args.layer, args.position)
    d = df[(df.checkpoint == args.checkpoint) & activations.has(df, column)].reset_index(drop=True)
    X = activations.matrix(d, column)

    mu_l, mu_h = X[(d.group == "harmless").to_numpy()].mean(0), X[(d.group == "harmful").to_numpy()].mean(0)
    d["t"] = per_prompt_t(X, mu_l, mu_h)
    pseudo = (d.group == "pseudo_harm").to_numpy()
    p_ = d[pseudo].copy()
    print(f"{args.family} {args.checkpoint} layer {args.layer} {args.position}: "
          f"{len(p_)} pseudo prompts, {int(p_.refused.sum())} refused")

    # 1. by response type
    print("\n1. t per tipo di risposta (quantili 10/50/90)")
    q = p_.groupby("response_type").t.describe(percentiles=[.1, .5, .9])
    print(q.reindex(refusal.RESPONSE_TYPES)[["count", "10%", "50%", "90%"]].round(2).to_string())
    for g in ["harmless", "harmful"]:
        print(f"   riferimento {g:9s} mediana t = {d[d.group == g].t.median():.2f}")

    # 2. permutation
    (gap, c), null = permutation(X[pseudo], p_.refused.to_numpy(), p_.source.to_numpy(), mu_h - mu_l,
                                 args.permutations)
    print(f"\n2. permutazione ({args.permutations} rimescolamenti dentro ogni fonte)")
    print(f"   gap t_rifiutate - t_accettate = {gap:.3f}   null 99° percentile = "
          f"{np.percentile(null[:, 0], 99):.3f}   p = {(null[:, 0] >= gap).mean():.3f}")
    print(f"   cos(v_beh, v_ref)            = {c:.3f}   null 99° percentile = "
          f"{np.percentile(null[:, 1], 99):.3f}   p = {(null[:, 1] >= c).mean():.3f}")

    # 3. length
    p_["length"] = p_.prompt.str.len()
    rho = p_[["t", "length"]].corr(method="spearman").iloc[0, 1]
    p_["length_q"] = pd.qcut(p_.length, 4, labels=["corti", "medio-corti", "medio-lunghi", "lunghi"])
    by_len = p_.groupby("length_q", observed=True).apply(
        lambda g: g[g.refused == 1].t.mean() - g[g.refused == 0].t.mean())
    print(f"\n3. lunghezza del prompt: Spearman(t, lunghezza) = {rho:.2f}")
    print("   gap dentro ogni quartile di lunghezza:", by_len.astype(float).round(2).to_dict())

    # 4. extremes
    show(p_[p_.refused == 0].nlargest(5, "t"), "4a. ACCETTATE ma più vicine alle harmful")
    show(p_[p_.refused == 1].nsmallest(5, "t"), "4b. RIFIUTATE ma più vicine alle harmless")

    # 5. flips
    if args.compare:
        other = df[(df.checkpoint == args.compare) & (df.group == "pseudo_harm")][["source", "prompt", "refused"]]
        f = p_.merge(other, on=["source", "prompt"], suffixes=("", "_other"))
        f["transition"] = f.refused.map({1: "rifiuta", 0: "risponde"}) + " -> " + \
            f.refused_other.map({1: "rifiuta", 0: "risponde"})
        print(f"\n5. t in {args.checkpoint}, secondo cosa fa {args.compare} con lo stesso prompt")
        print(f.groupby("transition").t.describe()[["count", "25%", "50%", "75%"]].round(2).to_string())
        p_ = p_.merge(f[["source", "prompt", "transition"]], on=["source", "prompt"], how="left")

    out = config.results_dir(args.family) / "inspect" / f"{args.checkpoint}_L{args.layer}_{args.position}.csv"
    save_atomic(p_.drop(columns=[c for c in p_.columns if c.startswith("layer_")]), out)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
