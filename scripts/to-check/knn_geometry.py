"""Beyond the refusal axis: neighbourhoods instead of a linear projection.

1. kNN score (knn_score.csv): for each pseudo-harmful prompt, the fraction of
   harmful prompts among its k nearest harmful-or-harmless neighbours. No axis,
   no projection. Compared with t, the position on the refusal axis:
     auroc_knn, auroc_t   how well each predicts hard refusal
     spearman_knn_t       how much the two scores agree prompt by prompt

2. Neighbourhood preservation (knn_preservation.csv): for each pseudo-harmful
   prompt, its k nearest neighbours among all prompts, in two checkpoints; the
   Jaccard overlap of the two sets. Near 1: the local geometry is unchanged.
   Chance level is about k / n_prompts. Broken down by outcome (hard refusal
   in the first vs the second checkpoint), to see whether the prompts whose
   behaviour changes are those whose neighbourhood changes.

Cosine on activations centred on the mean of all prompts at that checkpoint
and position: raw residual-stream states share a large common component.

    python scripts/knn_geometry.py --family olmo2 --layers 16
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from overrefusal import activations, cli, config, refusal
from overrefusal.io import save_atomic

PAIRS = [("sft__none", "dpo__none"), ("dpo__none", "final__none"), ("sft__none", "final__none")]


def unit_centred(X: np.ndarray) -> np.ndarray:
    X = X - X.mean(0)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def knn_idx(Q: np.ndarray, P: np.ndarray, k: int, exclude_self: bool) -> np.ndarray:
    """Indices of the k most cosine-similar rows of P for each row of Q (both unit)."""
    S = Q @ P.T
    if exclude_self:
        np.fill_diagonal(S, -np.inf)
    return np.argpartition(-S, k, axis=1)[:, :k]


def main():
    p = cli.parser(__doc__, activations=True)
    p.add_argument("--k", type=int, default=25)
    p.set_defaults(positions=["last_prompt", "post_instr", "first_gen"])
    args = p.parse_args()
    k = args.k

    df = activations.load(args.family, args.layers, args.positions, args.checkpoints)
    df = refusal.attach(df, pd.read_csv(config.raw_results_csv(args.family)))
    df = df.drop_duplicates(["checkpoint", "prompt"])
    df["hard"] = (df.response_type == "hard refusal").astype(int)

    scores, preserv = [], []
    for layer in args.layers:
        for pos in activations.expand(df, layer, args.positions):
            if pos == "pre_gen":
                continue
            column = activations.col(layer, pos)
            token = config.position_label(args.family, pos)
            frames = {}
            for ckpt in args.checkpoints:
                d = df[(df.checkpoint == ckpt) & activations.has(df, column)].reset_index(drop=True)
                if d.empty:
                    continue
                frames[ckpt] = d
                U = unit_centred(activations.matrix(d, column))
                ref = (d.group.isin(["harmless", "harmful"])).to_numpy()
                pseudo = (d.group == "pseudo_harm").to_numpy()
                nn = knn_idx(U[pseudo], U[ref], k, exclude_self=False)
                knn = (d.group[ref].to_numpy()[nn] == "harmful").mean(1)
                mu0 = U[(d.group == "harmless").to_numpy()].mean(0)
                v = U[(d.group == "harmful").to_numpy()].mean(0) - mu0
                t = (U[pseudo] - mu0) @ v / (v @ v)
                y = d.hard[pseudo].to_numpy()
                ok = min(y.sum(), len(y) - y.sum()) >= 20
                scores.append({"checkpoint": ckpt, "layer": layer, "position": pos, "token": token,
                               "n": len(y), "n_hard": int(y.sum()),
                               "auroc_knn": roc_auc_score(y, knn) if ok else np.nan,
                               "auroc_t": roc_auc_score(y, t) if ok else np.nan,
                               "spearman_knn_t": spearmanr(knn, t)[0],
                               "mean_knn_score": knn.mean()})

            for a, b in PAIRS:
                if a not in frames or b not in frames:
                    continue
                common = sorted(set(frames[a].prompt) & set(frames[b].prompt))
                A = frames[a].set_index("prompt").loc[common]
                B = frames[b].set_index("prompt").loc[common]
                Ua = unit_centred(activations.matrix(A.reset_index(), column))
                Ub = unit_centred(activations.matrix(B.reset_index(), column))
                nna, nnb = knn_idx(Ua, Ua, k, True), knn_idx(Ub, Ub, k, True)
                is_pseudo = (A.group == "pseudo_harm").to_numpy()
                jac = np.array([len(set(x) & set(z)) / len(set(x) | set(z))
                                for x, z in zip(nna[is_pseudo], nnb[is_pseudo])])
                ya, yb = A.hard.to_numpy()[is_pseudo], B.hard.to_numpy()[is_pseudo]
                outcome = np.where(ya == 1, "ref", "ans").astype(object) + " -> " + \
                          np.where(yb == 1, "ref", "ans").astype(object)
                base = {"pair": f"{a} vs {b}", "layer": layer, "position": pos, "token": token,
                        "chance": k / (len(common) - 1)}
                preserv.append({**base, "outcome": "all", "n": len(jac),
                                "jaccard_mean": jac.mean(), "jaccard_median": np.median(jac)})
                for o in sorted(set(outcome)):
                    m = outcome == o
                    preserv.append({**base, "outcome": o, "n": int(m.sum()),
                                    "jaccard_mean": jac[m].mean(), "jaccard_median": np.median(jac[m])})
            print(f"L{layer} {pos} done")

    geo = config.results_dir(args.family) / "geometry"
    s, pr = pd.DataFrame(scores), pd.DataFrame(preserv)
    save_atomic(s, geo / "knn_score.csv")
    save_atomic(pr, geo / "knn_preservation.csv")
    print(s.drop(columns=["token"]).round(3).to_string(index=False))
    print(pr[pr.pair == "sft__none vs dpo__none"].drop(columns=["pair", "token"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
