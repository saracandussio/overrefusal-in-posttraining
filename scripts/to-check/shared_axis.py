"""C2: does SFT move pseudo-harmful prompts towards harmful ones, or does the axis move?

Base and SFT measured on one shared axis, at last_prompt (the only token both
see), every layer. Each axis (base's, SFT's) is applied to both checkpoints:
  t_<ckpt>_on_<axis>   t of the pseudo-harmful centroid, with 95% bootstrap CI
  cos_axes             cosine between the base and SFT axes. Comparing the two
                       models in one space assumes SFT stays close to the base's
                       coordinates; a low value means that assumption fails.
If t rises from base to SFT under both axes, the shift is in the prompts.

    python scripts/shared_axis.py --family olmo2
"""
import numpy as np
import pandas as pd

from overrefusal import activations, cli, config
from overrefusal.io import save_atomic

PAIR = ("base__none", "sft__none")
N_BOOT = 300


def main():
    p = cli.parser(__doc__, activations=True)
    p.set_defaults(positions=["last_prompt"], checkpoints=list(PAIR))
    args = p.parse_args()

    df = activations.load(args.family, args.layers, ["last_prompt"], list(PAIR))
    df = df.drop_duplicates(["checkpoint", "prompt"])
    rng = np.random.default_rng(0)

    rows = []
    for layer in args.layers:
        column = activations.col(layer, "last_prompt")
        X = {(c, g): activations.matrix(df[(df.checkpoint == c) & (df.group == g)], column)
             for c in PAIR for g in ["harmless", "harmful", "pseudo_harm"]}
        axes = {c: (X[(c, "harmless")].mean(0), X[(c, "harmful")].mean(0)) for c in PAIR}
        v = {c: mu1 - mu0 for c, (mu0, mu1) in axes.items()}
        row = {"layer": layer, "cos_axes": float(v[PAIR[0]] @ v[PAIR[1]] /
               (np.linalg.norm(v[PAIR[0]]) * np.linalg.norm(v[PAIR[1]])))}
        for a in PAIR:                       # whose axis
            mu0, mu1 = axes[a]
            u = mu1 - mu0
            for c in PAIR:                   # whose activations
                P = X[(c, "pseudo_harm")]
                t = lambda M: float((M.mean(0) - mu0) @ u / (u @ u))
                boot = [t(P[rng.integers(0, len(P), len(P))]) for _ in range(N_BOOT)]
                key = f"t_{c.split('__')[0]}_on_{a.split('__')[0]}"
                row[key] = t(P)
                row[key + "_lo"], row[key + "_hi"] = np.percentile(boot, [2.5, 97.5])
        rows.append(row)
        print(f"L{layer} done")

    out = pd.DataFrame(rows)
    save_atomic(out, config.results_dir(args.family) / "geometry" / "shared_axis.csv")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
