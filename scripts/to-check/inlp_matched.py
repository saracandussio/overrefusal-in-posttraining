"""C4 with matched samples: does correct refusal lose its linear information
faster than over-refusal under iterative nullspace projection?

For each target (pseudo-harmful, harmful prompts; label = hard refusal) and
stage, the same number of refusals and non-refusals is drawn: the smallest
count over all targets and stages, so sample size and class balance cannot
explain differences between curves. Each draw is split 70/30; directions are
fitted on the training part only and AUROC is measured on the held-out part.
Repeated over seeds; mean and std reported per number of removed directions.

    python scripts/inlp_matched.py --family olmo2 --layers 16
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from overrefusal import activations, cli, config, probes, refusal
from overrefusal.io import save_atomic

TARGETS = ["pseudo_harm", "harmful"]
N_ITER = 20
N_SEEDS = 5


def direction(clf) -> np.ndarray:
    lr = clf[-1] if hasattr(clf, "__getitem__") else clf
    w = lr.coef_.ravel()
    scaler = clf[0] if hasattr(clf, "__getitem__") and hasattr(clf[0], "scale_") else None
    return w / scaler.scale_ if scaler is not None else w


def remove(X, v):
    u = v / np.linalg.norm(v)
    return X - np.outer(X @ u, u)


def curve(X, y, seed):
    tr, te = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=seed)
    out = []
    for _ in range(N_ITER + 1):
        clf = probes.fit(X[tr], y[tr])
        out.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
        X = remove(X, direction(clf))
    return out


def main():
    p = cli.parser(__doc__, activations=True)
    p.set_defaults(positions=["last_prompt", "pre_gen"],
                   checkpoints=["sft__none", "dpo__none", "final__none"])
    args = p.parse_args()

    df = activations.load(args.family, args.layers, args.positions, args.checkpoints)
    df = refusal.attach(df, pd.read_csv(config.raw_results_csv(args.family)))
    df = df.drop_duplicates(["checkpoint", "prompt"])
    df["hard"] = (df.response_type == "hard refusal").astype(int)

    cells = df[df.group.isin(TARGETS)].groupby(["checkpoint", "group"]).hard
    n_pos, n_neg = int(cells.sum().min()), int((cells.count() - cells.sum()).min())
    print(f"matched sample per target and stage: {n_pos} refusals + {n_neg} non-refusals")

    rows = []
    for layer in args.layers:
        for pos in args.positions:
            column = activations.col(layer, pos)
            for ckpt in args.checkpoints:
                for tg in TARGETS:
                    g = df[(df.checkpoint == ckpt) & (df.group == tg) & activations.has(df, column)]
                    for seed in range(N_SEEDS):
                        s = pd.concat([g[g.hard == 1].sample(n_pos, random_state=seed),
                                       g[g.hard == 0].sample(n_neg, random_state=seed)])
                        auc = curve(activations.matrix(s, column), s.hard.to_numpy(), seed)
                        rows += [{"layer": layer, "position": pos, "checkpoint": ckpt, "target": tg,
                                  "seed": seed, "removed": k, "auroc_test": a} for k, a in enumerate(auc)]
                    print(f"L{layer} {pos} {ckpt} {tg} done")

    out = pd.DataFrame(rows)
    save_atomic(out, config.results_dir(args.family) / "geometry" / "inlp_matched.csv")
    s = out[out.removed.isin([0, 1, 2, 5, 10, 20])].groupby(
        ["position", "checkpoint", "target", "removed"]).auroc_test.agg(["mean", "std"])
    print(s.unstack("removed").round(2).to_string())


if __name__ == "__main__":
    main()
