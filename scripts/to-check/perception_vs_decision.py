"""Perception versus decision, each measured two ways, compared like with like.

  perception (ground truth: harmful vs harmless prompts)
      d_ref = mu_H - mu_S                 -> t_ref, projection of pseudo-harmful prompts
      probe trained on H vs S             -> p_ref, its logit on pseudo-harmful prompts
  decision (behaviour: hard refusal of pseudo-harmful prompts)
      d_beh = mu_refused - mu_answered    -> t_beh, out of fold
      probe trained on refused vs not     -> p_beh, out of fold

The perception side never sees pseudo-harmful prompts or behaviour. The decision
side is scored out of fold (5 folds), so no prompt is scored by a direction or
probe that saw it.

summary  (perception_vs_decision.csv): per checkpoint, layer, position
  auroc_<score>          AUROC of each score for hard refusal
  rho_<a>_<b>            Spearman between scores on pseudo-harmful prompts
  cos_<a>_<b>            cosine between directions; probe cosines come with a
                         95th-percentile null from shuffled-label probes
scores   (perception_vs_decision_scores.csv): the four scores for every prompt

    python scripts/perception_vs_decision.py --family olmo2 --layers 16
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from overrefusal import activations, cli, config, probes, refusal
from overrefusal.io import save_atomic

MIN_PER_CLASS = 20
N_NULL = 5
FOLDS = 5


def direction(clf) -> np.ndarray:
    lr = clf[-1] if hasattr(clf, "__getitem__") else clf
    w = lr.coef_.ravel()
    scaler = clf[0] if hasattr(clf, "__getitem__") and hasattr(clf[0], "scale_") else None
    return w / scaler.scale_ if scaler is not None else w


def logit(clf, X) -> np.ndarray:
    p = np.clip(clf.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


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

    summary, scores = [], []
    for ckpt in args.checkpoints:
        for layer in args.layers:
            for pos in activations.expand(df, layer, args.positions):
                if pos == "pre_gen":
                    continue
                column = activations.col(layer, pos)
                d = df[(df.checkpoint == ckpt) & activations.has(df, column)]
                if d.empty:
                    continue
                XH = activations.matrix(d[d.group == "harmful"], column)
                XS = activations.matrix(d[d.group == "harmless"], column)
                P = d[d.group == "pseudo_harm"]
                XP, y = activations.matrix(P, column), P.hard.to_numpy()

                # perception: fitted on ground truth only
                d_ref = XH.mean(0) - XS.mean(0)
                t_ref = (XP - XS.mean(0)) @ d_ref / (d_ref @ d_ref)
                XG, yG = np.vstack([XH, XS]), np.r_[np.ones(len(XH)), np.zeros(len(XS))].astype(int)
                probe_ref = probes.fit(XG, yG)
                p_ref, w_ref = logit(probe_ref, XP), direction(probe_ref)
                null_ref = [abs(cos(direction(probes.fit(XG, rng.permutation(yG))), d_ref))
                            for _ in range(N_NULL)]

                row = {"checkpoint": ckpt, "layer": layer, "position": pos,
                       "token": config.position_label(args.family, pos, config.stage_of(ckpt)),
                       "n_pseudo": len(y), "n_hard": int(y.sum()),
                       "mean_t_ref": float(t_ref.mean()), "mean_p_ref": float(p_ref.mean()),
                       "cos_dref_wref": cos(d_ref, w_ref), "null95_dref_wref": float(np.percentile(null_ref, 95))}
                sc = {"t_ref": t_ref, "p_ref": p_ref}

                if min(y.sum(), len(y) - y.sum()) >= MIN_PER_CLASS:
                    # decision: out of fold
                    t_beh, p_beh = np.empty(len(y)), np.empty(len(y))
                    for tr, te in StratifiedKFold(FOLDS, shuffle=True, random_state=0).split(XP, y):
                        db = XP[tr][y[tr] == 1].mean(0) - XP[tr][y[tr] == 0].mean(0)
                        mid = (XP[tr][y[tr] == 1].mean(0) + XP[tr][y[tr] == 0].mean(0)) / 2
                        t_beh[te] = (XP[te] - mid) @ db / (db @ db)
                        p_beh[te] = logit(probes.fit(XP[tr], y[tr]), XP[te])
                    d_beh = XP[y == 1].mean(0) - XP[y == 0].mean(0)
                    w_beh = direction(probes.fit(XP, y))
                    null_beh = [direction(probes.fit(XP, rng.permutation(y))) for _ in range(N_NULL)]
                    sc |= {"t_beh": t_beh, "p_beh": p_beh}
                    row |= {f"auroc_{k}": roc_auc_score(y, v) for k, v in sc.items()}
                    names = list(sc)
                    row |= {f"rho_{a}_{b}": spearmanr(sc[a], sc[b])[0]
                            for i, a in enumerate(names) for b in names[i + 1:]}
                    row |= {"cos_dref_dbeh": cos(d_ref, d_beh),
                            "cos_wref_wbeh": cos(w_ref, w_beh),
                            "null95_wref_wbeh": float(np.percentile([abs(cos(w_ref, n)) for n in null_beh], 95)),
                            "cos_dbeh_wbeh": cos(d_beh, w_beh),
                            "null95_dbeh_wbeh": float(np.percentile([abs(cos(d_beh, n)) for n in null_beh], 95))}
                else:
                    row |= {"auroc_t_ref": roc_auc_score(y, t_ref) if 0 < y.sum() < len(y) else np.nan,
                            "auroc_p_ref": roc_auc_score(y, p_ref) if 0 < y.sum() < len(y) else np.nan}

                summary.append(row)
                scores.append(pd.DataFrame({"checkpoint": ckpt, "layer": layer, "position": pos,
                                            "prompt": P.prompt.values, "source": P.source.values,
                                            "hard": y, **sc}))
                print(f"{ckpt} L{layer} {pos} done")

    geo = config.results_dir(args.family) / "geometry"
    s = pd.DataFrame(summary)
    save_atomic(s, geo / "perception_vs_decision.csv")
    save_atomic(pd.concat(scores, ignore_index=True), geo / "perception_vs_decision_scores.csv")
    cols = [c for c in s.columns if c.startswith(("auroc_", "rho_", "cos_", "null95_"))]
    print(s[["checkpoint", "position"] + cols].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
