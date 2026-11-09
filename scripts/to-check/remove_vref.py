"""Is the information that predicts hard refusal concentrated on the refusal axis?

For each checkpoint, layer and position, on pseudo-harmful prompts, the
cross-validated AUROC of a linear probe for hard refusal:
  auroc_full       on the full hidden state
  auroc_no_vref    after projecting out v_ref = mu_harmful - mu_harmless
  auroc_no_random  after projecting out a random direction (control: should
                   barely change; mean over 5 directions)
  auroc_no_w       after projecting out the probe's own direction (first INLP step)
  auroc_t_full     AUROC of t, the position on the refusal axis
  auroc_t_no_w     the same, with the axis recomputed after removing w: does the
                   axis's predictive power go through the probe's direction?
  vref_kept        share of v_ref's length left after removing w

auroc_no_vref close to auroc_full: the information is also present off the
axis (redundant). Close to 0.5: it lies essentially on the axis.

    python scripts/remove_vref.py --family olmo2 --layers 16
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from overrefusal import activations, cli, config, probes, refusal
from overrefusal.io import save_atomic

MIN_PER_CLASS = 20


def remove(X: np.ndarray, v: np.ndarray) -> np.ndarray:
    u = v / np.linalg.norm(v)
    return X - np.outer(X @ u, u)


def direction(clf) -> np.ndarray:
    lr = clf[-1] if hasattr(clf, "__getitem__") else clf
    w = lr.coef_.ravel()
    scaler = clf[0] if hasattr(clf, "__getitem__") and hasattr(clf[0], "scale_") else None
    return w / scaler.scale_ if scaler is not None else w


def main():
    p = cli.parser(__doc__, activations=True)
    p.set_defaults(positions=["last_prompt", "post_instr", "first_gen"])
    p.add_argument("--target-group", default="pseudo_harm", choices=["pseudo_harm", "harmful"])
    args = p.parse_args()

    df = activations.load(args.family, args.layers, args.positions, args.checkpoints)
    df = refusal.attach(df, pd.read_csv(config.raw_results_csv(args.family)))
    df = df.drop_duplicates(["checkpoint", "prompt"])
    df["hard"] = (df.response_type == "hard refusal").astype(int)
    rng = np.random.default_rng(0)

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
                pseudo = d[d.group == args.target_group]
                y = pseudo.hard.to_numpy()
                if min(y.sum(), len(y) - y.sum()) < MIN_PER_CLASS:
                    print(f"  skip {ckpt} L{layer} {pos}: {y.sum()} hard refusals")
                    continue
                X = activations.matrix(pseudo, column)
                H1 = activations.matrix(d[d.group == "harmful"], column)
                H0 = activations.matrix(d[d.group == "harmless"], column)
                vref = H1.mean(0) - H0.mean(0)
                w = direction(probes.fit(X, y))
                t_full = (X - H0.mean(0)) @ vref / (vref @ vref)
                Xw, H1w, H0w = remove(X, w), remove(H1, w), remove(H0, w)
                vref_w = H1w.mean(0) - H0w.mean(0)
                t_no_w = (Xw - H0w.mean(0)) @ vref_w / (vref_w @ vref_w)
                rand = [probes.cv_auroc(remove(X, rng.standard_normal(X.shape[1])), y)[0]
                        for _ in range(5)]
                rows.append({"checkpoint": ckpt, "layer": layer, "position": pos,
                             "token": config.position_label(args.family, pos, config.stage_of(ckpt)),
                             "n": len(y), "n_hard": int(y.sum()),
                             "auroc_full": probes.cv_auroc(X, y)[0],
                             "auroc_no_vref": probes.cv_auroc(remove(X, vref), y)[0],
                             "auroc_no_random": float(np.mean(rand)),
                             "auroc_no_w": probes.cv_auroc(Xw, y)[0],
                             "auroc_t_full": roc_auc_score(y, t_full),
                             "auroc_t_no_w": roc_auc_score(y, t_no_w),
                             "vref_kept": float(np.linalg.norm(vref_w) / np.linalg.norm(vref))})
                print(f"{ckpt} L{layer} {pos} done")

    out = pd.DataFrame(rows)
    name = "remove_vref.csv" if args.target_group == "pseudo_harm" else f"remove_vref_{args.target_group}.csv"
    save_atomic(out, config.results_dir(args.family) / "geometry" / name)
    print(out.drop(columns=["n", "token"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
