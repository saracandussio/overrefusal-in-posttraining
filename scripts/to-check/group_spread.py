"""Where do harmless, pseudo-harmful and harmful prompts sit on the refusal axis?

The centroids of harmless and harmful prompts are at 0 and 1 by construction, so
this reports the spread of each group (median and quartiles of t) and how well the
axis separates harmful from harmless: d' = 1 / pooled std of t within the two groups.

    python scripts/group_spread.py --family olmo2 --layers 16
"""
import numpy as np
import pandas as pd

from overrefusal import activations, cli, config
from overrefusal.io import save_atomic


def main():
    p = cli.parser(__doc__, activations=True)
    p.set_defaults(positions=["last_prompt", "post_instr", "first_gen"])
    args = p.parse_args()

    df = activations.load(args.family, args.layers, args.positions, args.checkpoints)
    df = df.drop_duplicates(["checkpoint", "prompt"])

    rows = []
    for ckpt in args.checkpoints:
        for layer in args.layers:
            for pos in activations.expand(df, layer, args.positions):
                if pos == "pre_gen":
                    continue
                column = activations.col(layer, pos)
                d = df[(df.checkpoint == ckpt) & activations.has(df, column)]
                if d.empty:
                    continue
                X = {g: activations.matrix(d[d.group == g], column)
                     for g in ["harmless", "pseudo_harm", "harmful"]}
                mu0, mu1 = X["harmless"].mean(0), X["harmful"].mean(0)
                v = mu1 - mu0
                t = {g: (x - mu0) @ v / (v @ v) for g, x in X.items()}
                pooled = np.sqrt((t["harmless"].var() + t["harmful"].var()) / 2)
                for g, tg in t.items():
                    q1, med, q3 = np.percentile(tg, [25, 50, 75])
                    rows.append({"checkpoint": ckpt, "layer": layer, "position": pos,
                                 "token": config.position_label(args.family, pos,
                                                                config.stage_of(ckpt)),
                                 "group": g, "n": len(tg), "median": med, "q1": q1, "q3": q3,
                                 "d_prime": 1 / pooled,
                                 "share_above_half": float((tg > 0.5).mean())})
            print(f"{ckpt} L{layer} done")

    out = pd.DataFrame(rows)
    save_atomic(out, config.results_dir(args.family) / "geometry" / "group_spread.csv")
    view = out[out.position == "last_prompt"].pivot_table(
        index=["layer", "group"], columns="checkpoint", values="median", sort=False).round(2)
    print(view.to_string())
    print(out[out.position == "last_prompt"].drop_duplicates(["checkpoint", "layer"])
          .pivot_table(index="layer", columns="checkpoint", values="d_prime").round(2).to_string())


if __name__ == "__main__":
    main()
