"""Exp. 1: is the perception measure robust to the choice of probe?

For every checkpoint, layer and position, a probe harmful (H) vs harmless (S)
is trained on all sources, and then once more with each (group, source) left
out. Each probe gets its own threshold, calibrated so that a fixed share of
harmless prompts it did not train on (out of fold) scores above it: 5% and 10%.
"Perceived harmful" means logit > threshold, which is comparable across stages,
unlike the probe's own zero.

Per probe and threshold (loso_perception.csv):
  held_correct      share of the held-out source put on its own side
  seen, seen_<src>  share of pseudo-harmful prompts (P) perceived harmful, overall and per P source
  refused_if_seen   share of perceived-harmful P with a hard refusal
  xstest_auroc      only for held_out = harmful:xstest (probe that never saw XSTest):
                    AUROC of XSTest contrast vs XSTest safe

The full probe also writes per-prompt scores for P (loso_perception_scores.csv),
with z = (logit - mean S) / (mean H - mean S), means taken on out-of-fold H and S
logits: 0 = typical harmless, 1 = typical harmful. This is the score for Exp. 2.

    python scripts/loso_perception.py --family olmo2 --layers 16 19 24 26 31
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from overrefusal import activations, cli, config, probes, refusal
from overrefusal.io import save_atomic

FOLDS = 3
FPRS = (0.05, 0.10)


def logit(clf, X) -> np.ndarray:
    p = np.clip(clf.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_with_oof(X, y):
    """Probe on all rows, plus out-of-fold logits for every row (for calibration)."""
    oof = np.empty(len(y))
    for tr, te in StratifiedKFold(FOLDS, shuffle=True, random_state=0).split(X, y):
        oof[te] = logit(probes.fit(X[tr], y[tr]), X[te])
    return probes.fit(X, y), oof


def main():
    p = cli.parser(__doc__, activations=True)
    p.set_defaults(positions=["last_prompt", "pre_gen"])
    args = p.parse_args()

    df = activations.load(args.family, args.layers, args.positions, args.checkpoints)
    df = refusal.attach(df, pd.read_csv(config.raw_results_csv(args.family)))
    df = df.drop_duplicates(["checkpoint", "prompt"])
    df["hard"] = (df.response_type == "hard refusal").astype(int)

    rows, scores = [], []
    for ckpt in args.checkpoints:
        for layer in args.layers:
            for pos in args.positions:
                column = activations.col(layer, pos)
                if column not in df.columns:
                    continue
                d = df[(df.checkpoint == ckpt) & activations.has(df, column)]
                if d.empty:
                    continue
                G = d[d.group.isin(["harmful", "harmless"])]
                XG = activations.matrix(G, column)
                yG = (G.group == "harmful").to_numpy().astype(int)
                key = (G.group + ":" + G.source).to_numpy()
                P = d[d.group == "pseudo_harm"]
                XP, hard, psrc = activations.matrix(P, column), P.hard.to_numpy(), P.source.to_numpy()

                for held in ["none"] + sorted(set(key)):
                    keep = key != held
                    clf, oof = fit_with_oof(XG[keep], yG[keep])
                    lp = logit(clf, XP)
                    lh = logit(clf, XG[~keep]) if held != "none" else None

                    xs_auc = np.nan
                    if held == "harmful:xstest":
                        safe = psrc == "xstest"
                        xs_auc = roc_auc_score(np.r_[np.ones(len(lh)), np.zeros(safe.sum())],
                                               np.r_[lh, lp[safe]])

                    for fpr in FPRS:
                        tau = float(np.quantile(oof[yG[keep] == 0], 1 - fpr))
                        seen = lp > tau
                        row = {"checkpoint": ckpt, "layer": layer, "position": pos,
                               "held_out": held, "fpr": fpr, "tau": tau,
                               "n_pseudo": len(lp), "seen": float(seen.mean()),
                               "refused_if_seen": float(hard[seen].mean()) if seen.any() else np.nan,
                               "xstest_auroc": xs_auc}
                        if held != "none":
                            row["n_held"] = int((~keep).sum())
                            row["held_correct"] = float(((lh > tau) == held.startswith("harmful")).mean())
                        for s in sorted(set(psrc)):
                            row[f"seen_{s}"] = float(seen[psrc == s].mean())
                        rows.append(row)

                    if held == "none":
                        mS, mH = oof[yG == 0].mean(), oof[yG == 1].mean()
                        taus = {f"tau{int(f * 100):02d}": float(np.quantile(oof[yG == 0], 1 - f)) for f in FPRS}
                        scores.append(pd.DataFrame({"checkpoint": ckpt, "layer": layer, "position": pos,
                                                    "prompt": P.prompt.values, "source": psrc, "hard": hard,
                                                    "logit": lp, "z": (lp - mS) / (mH - mS), **taus}))
                print(f"{ckpt} L{layer} {pos} done", flush=True)

    geo = config.results_dir(args.family) / "geometry"
    out = pd.DataFrame(rows)
    save_atomic(out, geo / "loso_perception.csv")
    save_atomic(pd.concat(scores, ignore_index=True), geo / "loso_perception_scores.csv")

    # quick look: share of P perceived harmful at 5% FPR, full probe and range over left-out probes
    o = out[out.fpr == 0.05]
    full = o[o.held_out == "none"].set_index(["position", "layer", "checkpoint"]).seen
    loso = o[o.held_out != "none"].groupby(["position", "layer", "checkpoint"]).seen.agg(["min", "max"])
    print(pd.concat([full.rename("full"), loso], axis=1).round(2).to_string())


if __name__ == "__main__":
    main()
