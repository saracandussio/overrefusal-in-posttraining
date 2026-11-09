"""Do hidden states predict hard refusals of pseudo-harmful prompts, and along
which direction?

For each checkpoint, layer and position, on pseudo-harmful prompts:
  auroc_probe   5-fold cross-validated AUROC of a linear probe on the full
                hidden state (hard refusal vs any other response)
  auroc_loso    AUROC on each held-out pseudo-harmful source, training on the
                others (mean, min, and one column per source)
  auroc_t       AUROC of t, the position on the refusal axis: if close to
                auroc_probe, the axis already captures what the probe finds
  cos_vref      cosine between the probe's weight vector (in activation space)
                and v_ref = mu_harmful - mu_harmless
  cos_null_*    the same cosine for probes trained on shuffled labels: the
                baseline any direction reaches in an anisotropic space

    python scripts/probe_direction.py --family olmo2 --layers 16
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from overrefusal import activations, cli, config, probes, refusal
from overrefusal.io import save_atomic

MIN_PER_CLASS = 20
N_NULL = 20


def weights(clf, d: int) -> np.ndarray:
    """Probe direction in raw activation space, whether or not make() standardizes."""
    lr = clf[-1] if hasattr(clf, "__getitem__") else clf
    w = lr.coef_.ravel()
    scaler = clf[0] if hasattr(clf, "__getitem__") and hasattr(clf[0], "scale_") else None
    w = w / scaler.scale_ if scaler is not None else w
    assert w.shape == (d,)
    return w


def loso_auroc(X: np.ndarray, y: np.ndarray, sources: np.ndarray) -> dict[str, float]:
    """AUROC on each held-out source, training on the others. Sources where
    either the training or the test side has fewer than 5 prompts of a class
    are skipped: AUROC is undefined or meaningless there."""
    out = {}
    for s in np.unique(sources):
        test = sources == s
        if min(y[test].sum(), (1 - y[test]).sum(), y[~test].sum(), (1 - y[~test]).sum()) < 5:
            continue
        clf = probes.fit(X[~test], y[~test])
        out[str(s)] = roc_auc_score(y[test], clf.predict_proba(X[test])[:, 1])
    return out


def cos(a, b) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    p = cli.parser(__doc__, activations=True)
    p.set_defaults(positions=["last_prompt", "post_instr", "first_gen"])
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
                pseudo = d[d.group == "pseudo_harm"]
                y = pseudo.hard.to_numpy()
                if min(y.sum(), len(y) - y.sum()) < MIN_PER_CLASS:
                    print(f"  skip {ckpt} L{layer} {pos}: {y.sum()} hard refusals of {len(y)}")
                    continue
                X = activations.matrix(pseudo, column)
                mu0 = activations.matrix(d[d.group == "harmless"], column).mean(0)
                mu1 = activations.matrix(d[d.group == "harmful"], column).mean(0)
                vref = mu1 - mu0
                t = (X - mu0) @ vref / (vref @ vref)

                auroc_probe = probes.cv_auroc(X, y)[0]
                loso = loso_auroc(X, y, pseudo.source.to_numpy())
                w = weights(probes.fit(X, y), X.shape[1])
                null = [cos(weights(probes.fit(X, rng.permutation(y)), X.shape[1]), vref)
                        for _ in range(N_NULL)]
                rows.append({"checkpoint": ckpt, "layer": layer, "position": pos,
                             "token": config.position_label(args.family, pos, config.stage_of(ckpt)),
                             "n": len(y), "n_hard": int(y.sum()),
                             "auroc_probe": auroc_probe,
                             "auroc_loso_mean": float(np.mean(list(loso.values()))) if loso else np.nan,
                             "auroc_loso_min": float(min(loso.values())) if loso else np.nan,
                             **{f"auroc_loso_{k}": v for k, v in loso.items()},
                             "auroc_t": roc_auc_score(y, t),
                             "cos_vref": cos(w, vref),
                             "cos_null_mean": float(np.mean(null)),
                             "cos_null_p95": float(np.percentile(np.abs(null), 95))})
                print(f"{ckpt} L{layer} {pos} done")

    out = pd.DataFrame(rows)
    save_atomic(out, config.results_dir(args.family) / "geometry" / "probe_direction.csv")
    print(out.drop(columns=["n", "token"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
