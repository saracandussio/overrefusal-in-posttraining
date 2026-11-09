"""Where and how do the activations move from SFT to DPO?

Same prompts, same layer, same token, two checkpoints (DPO starts from SFT,
so the two spaces are comparable). For each prompt: delta = a_DPO - a_SFT.

Per layer and token (geometry_shift.csv):
  rel_shift     median |delta| / |a_SFT|: how much the activations move
  common_share  share of the movement that is the same for every prompt (mean delta)
  pc1, pc2, pc3 share of the remaining movement on the top 3 directions of delta
  bacc_group    can a probe tell H / S / P apart from delta alone? (chance 0.33)
  bacc_flip     among P refused by SFT: can a probe tell from delta which ones
                DPO stops refusing? (chance 0.5)

    python scripts/geometry_shift.py --family olmo2 --layers 16 19 24 26 31
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from overrefusal import activations, cli, config, probes, refusal
from overrefusal.io import save_atomic

CKPTS = ["sft__none", "dpo__none"]


def oof_bacc(X, y, folds=5):
    y = pd.factorize(y)[0]
    if len(np.unique(y)) < 2 or np.bincount(y).min() < 10:
        return np.nan
    pred = np.empty(len(y), dtype=int)
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=0).split(X, y):
        pred[te] = probes.fit(X[tr], y[tr]).predict(X[te])
    return float(balanced_accuracy_score(y, pred))


def main():
    p = cli.parser(__doc__, activations=True)
    p.set_defaults(positions=["last_prompt", "post_instr"])
    args = p.parse_args()

    df = activations.load(args.family, args.layers, args.positions, CKPTS)
    df = refusal.attach(df, pd.read_csv(config.raw_results_csv(args.family)))
    df = df.drop_duplicates(["checkpoint", "prompt"])
    df["hard"] = (df.response_type == "hard refusal").astype(int)
    s = df[df.checkpoint == CKPTS[0]].set_index("prompt")
    d = df[df.checkpoint == CKPTS[1]].set_index("prompt")
    both = s.index.intersection(d.index)
    s, d = s.loc[both], d.loc[both]

    rows = []
    for layer in args.layers:
        for pos in activations.expand(df, layer, args.positions):
            if pos == "pre_gen":
                continue
            col = activations.col(layer, pos)
            if col not in s.columns:
                continue
            ok = (activations.has(s, col) & activations.has(d, col)).to_numpy()
            if ok.sum() < 100:
                continue
            S, D = s[ok], d[ok]
            XS, XD = activations.matrix(S, col), activations.matrix(D, col)
            delta = XD - XS
            mean = delta.mean(0)
            pcs = PCA(3).fit(delta - mean).explained_variance_ratio_

            P = (S.group == "pseudo_harm").to_numpy()
            sft_refused = P & (S.hard.to_numpy() == 1)
            flip = D.hard.to_numpy()[sft_refused] == 0

            rows.append({"layer": layer, "position": pos, "n": int(ok.sum()),
                         "rel_shift": float(np.median(np.linalg.norm(delta, axis=1)
                                                      / np.linalg.norm(XS, axis=1))),
                         "common_share": float(len(delta) * (mean @ mean) / (delta ** 2).sum()),
                         "pc1": pcs[0], "pc2": pcs[1], "pc3": pcs[2],
                         "bacc_group": oof_bacc(delta, S.group.to_numpy()),
                         "n_sft_refused_P": int(sft_refused.sum()), "n_flip": int(flip.sum()),
                         "bacc_flip": oof_bacc(delta[sft_refused], flip)})
            print(f"L{layer} {pos} done", flush=True)

    out = pd.DataFrame(rows)
    save_atomic(out, config.results_dir(args.family) / "geometry" / "geometry_shift.csv")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
