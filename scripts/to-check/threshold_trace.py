"""Does DPO move the prompts, or the threshold? Traced along the positions.

For each layer and position (last_prompt, every template token, first_gen):
the refusal axis is built from SFT's harmless and harmful centroids, and every
pseudo-harmful prompt gets its position t on that one fixed axis. Each
checkpoint's hard refusals are then fitted on that same t, so SFT, DPO and
final are compared in one space:

  spearman   t on the checkpoint's own axis vs t on SFT's: do prompts keep
             their place? (1 for SFT by construction)
  auroc      how well SFT's t predicts this checkpoint's hard refusals
  t50, slope logistic fit P(hard refusal | t): where the model switches from
             answering to refusing, and how sharply

A higher t50 with the same slope means: same geometry, stricter rule.
first_gen already encodes the chosen token, so its AUROC is expected near 1.

    python scripts/threshold_trace.py --family olmo2 --layers 16
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from overrefusal import activations, cli, config, refusal
from overrefusal.io import save_atomic

REFERENCE = "sft__none"


def axis_t(d: pd.DataFrame, column: str, ref: pd.DataFrame) -> pd.Series:
    """Position of each pseudo-harmful prompt of d on ref's harmless->harmful axis."""
    X = lambda frame, g: activations.matrix(frame[frame.group == g], column)
    mu0, mu1 = X(ref, "harmless").mean(0), X(ref, "harmful").mean(0)
    v = mu1 - mu0
    p = d[d.group == "pseudo_harm"]
    return pd.Series((activations.matrix(p, column) - mu0) @ v / (v @ v), index=p.prompt.values)


def logistic(t: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(t[:, None], y)
    b0, b1 = lr.intercept_[0], lr.coef_[0, 0]
    return -b0 / b1, b1


def main():
    p = cli.parser(__doc__, activations=True)
    p.add_argument("--bootstrap", type=int, default=300)
    p.set_defaults(checkpoints=["sft__none", "dpo__none", "final__none"],
                   positions=["last_prompt", "post_instr", "first_gen"])
    args = p.parse_args()
    assert REFERENCE in args.checkpoints, f"{REFERENCE} is the reference space"

    df = activations.load(args.family, args.layers, args.positions, args.checkpoints)
    df = refusal.attach(df, pd.read_csv(config.raw_results_csv(args.family)))
    df = df.drop_duplicates(["checkpoint", "prompt"])
    df["hard"] = (df.response_type == "hard refusal").astype(int)
    rng = np.random.default_rng(0)

    rows = []
    for layer in args.layers:
        for pos in activations.expand(df, layer, args.positions):
            if pos == "pre_gen":  # alias of the last template token, already traced
                continue
            column = activations.col(layer, pos)
            d = df[activations.has(df, column)]
            ref = d[d.checkpoint == REFERENCE]
            t_ref = axis_t(ref, column, ref)
            for ckpt in args.checkpoints:
                c = d[d.checkpoint == ckpt]
                own = axis_t(c, column, c)
                y = c[c.group == "pseudo_harm"].set_index("prompt").hard
                common = t_ref.index.intersection(y.index).intersection(own.index)
                t, yy = t_ref[common].to_numpy(), y[common].to_numpy()
                t50, slope = logistic(t, yy)
                boot = np.array([logistic(t[i], yy[i]) for i in
                                 (rng.integers(0, len(t), len(t)) for _ in range(args.bootstrap))])
                (t50_lo, s_lo), (t50_hi, s_hi) = np.percentile(boot, [2.5, 97.5], axis=0)
                rows.append({"layer": layer, "position": pos,
                             "token": config.position_label(args.family, pos),
                             "checkpoint": ckpt, "n": len(t), "hard_rate": yy.mean(),
                             "spearman": spearmanr(t, own[common])[0],
                             "auroc": roc_auc_score(yy, t),
                             "t50": t50, "t50_lo": t50_lo, "t50_hi": t50_hi,
                             "slope": slope, "slope_lo": s_lo, "slope_hi": s_hi})
            print(f"L{layer} {pos} done")

    out = pd.DataFrame(rows)
    save_atomic(out, config.results_dir(args.family) / "geometry" / "threshold_trace.csv")
    view = out.pivot_table(index=["layer", "position"], columns="checkpoint",
                           values=["t50", "auroc", "spearman"], sort=False).round(2)
    print(view.to_string())


if __name__ == "__main__":
    main()
