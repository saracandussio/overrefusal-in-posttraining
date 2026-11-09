"""C4: how spread out is the information that predicts hard refusal?

Iterative nullspace projection (Ravfogel et al., 2020): fit a linear probe for
hard refusal on a training split, record its AUROC on a held-out split and its
cosine with v_ref, project its direction out of all activations, repeat.

Read the curves comparatively (target group, stage), not as a dimension: for a
binary target all linear information can in principle be removed with one
well-chosen direction (LEACE), which greedy probes do not find.

    python scripts/inlp_curve.py --family olmo2 --layers 16 --target-group pseudo_harm
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from overrefusal import activations, cli, config, probes, refusal
from overrefusal.io import save_atomic

N_ITER = 20
MIN_PER_CLASS = 20


def direction(clf) -> np.ndarray:
    lr = clf[-1] if hasattr(clf, "__getitem__") else clf
    w = lr.coef_.ravel()
    scaler = clf[0] if hasattr(clf, "__getitem__") and hasattr(clf[0], "scale_") else None
    return w / scaler.scale_ if scaler is not None else w


def remove(X, v):
    u = v / np.linalg.norm(v)
    return X - np.outer(X @ u, u)


def main():
    p = cli.parser(__doc__, activations=True)
    p.add_argument("--target-group", default="pseudo_harm", choices=["pseudo_harm", "harmful"])
    p.set_defaults(positions=["last_prompt", "post_instr", "first_gen"])
    args = p.parse_args()

    df = activations.load(args.family, args.layers, args.positions, args.checkpoints)
    df = refusal.attach(df, pd.read_csv(config.raw_results_csv(args.family)))
    df = df.drop_duplicates(["checkpoint", "prompt"])
    df["hard"] = (df.response_type == "hard refusal").astype(int)

    rows = []
    for ckpt in args.checkpoints:
        for layer in args.layers:
            for pos in activations.expand(df, layer, args.positions):
                if pos == "pre_gen":
                    continue
                column = activations.col(layer, pos)
                d = df[(df.checkpoint == ckpt) & activations.has(df, column)]
                g = d[d.group == args.target_group]
                y = g.hard.to_numpy()
                if d.empty or min(y.sum(), len(y) - y.sum()) < MIN_PER_CLASS:
                    continue
                X = activations.matrix(g, column)
                vref = (activations.matrix(d[d.group == "harmful"], column).mean(0)
                        - activations.matrix(d[d.group == "harmless"], column).mean(0))
                tr, te = train_test_split(np.arange(len(y)), test_size=0.3,
                                          stratify=y, random_state=0)
                for i in range(N_ITER + 1):
                    clf = probes.fit(X[tr], y[tr])
                    auroc = roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1])
                    w = direction(clf)
                    rows.append({"checkpoint": ckpt, "layer": layer, "position": pos,
                                 "token": config.position_label(args.family, pos, config.stage_of(ckpt)),
                                 "target": args.target_group, "removed": i, "auroc_test": auroc,
                                 "cos_w_vref": float(w @ vref / (np.linalg.norm(w) * np.linalg.norm(vref)))})
                    X = remove(X, w)
                print(f"{ckpt} L{layer} {pos} done")

    out = pd.DataFrame(rows)
    save_atomic(out, config.results_dir(args.family) / "geometry" / f"inlp_{args.target_group}.csv")
    print(out[out.removed.isin([0, 1, 2, 5, 10, 20])]
          .pivot_table(index=["checkpoint", "position"], columns="removed", values="auroc_test", sort=False)
          .round(3).to_string())


if __name__ == "__main__":
    main()
