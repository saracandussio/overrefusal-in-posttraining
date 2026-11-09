"""Three-way probe (harmful / harmless / pseudo-harmful) and k-means with k = 3.

Per checkpoint, layer and position (last_prompt, pre_gen):
  probe   out-of-fold multinomial probe (5 folds): balanced accuracy, confusion
          matrix normalised by true class (<true>_as_<pred>), and pseudo-harmful
          predictions split by hard refusal (p_refused_as_*, p_answered_as_*)
  loso    each (source, group) held out in turn -> share predicted as each group.
  kmeans  k = 3 on centred, unit-norm states: AMI with group, hard refusal and
          source; cluster x group counts (+ refused pseudo-harmful per cluster)

WildGuardMix and ToxicChat are included (harmful + harmless, split verified).

    python scripts/three_way.py --family olmo2 --layers 16
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from overrefusal import activations, cli, config, probes, refusal
from overrefusal.io import save_atomic

GROUPS = ["harmful", "harmless", "pseudo_harm"]
MIN_TRAIN = 20
PRE = {"olmo2": "post_instr_6", "olmo3": "post_instr_4"}
BASE_PRE = "post_instr_2"


def main():
    p = cli.parser(__doc__, activations=True)
    p.add_argument("--no-loso", action="store_true")
    p.set_defaults(checkpoints=["base__none", "sft__none", "dpo__none", "final__none"],
                   positions=["last_prompt", "post_instr"])
    args = p.parse_args()

    df = activations.load(args.family, args.layers, args.positions, args.checkpoints)
    df = refusal.attach(df, pd.read_csv(config.raw_results_csv(args.family)))
    df = df.drop_duplicates(["checkpoint", "prompt"])
    df = df[df.group.isin(GROUPS)]
    df["hard"] = (df.response_type == "hard refusal").astype(int)
    print(df.groupby(["checkpoint", "group"]).size().unstack())
    print(sorted(df.source.unique()))

    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    rows, loso_rows, cluster_rows = [], [], []
    for layer in args.layers:
        for pos in activations.expand(df, layer, args.positions):
            column = activations.col(layer, pos)
            d = df[activations.has(df, column)]
            for ckpt in args.checkpoints:
                pre = BASE_PRE if ckpt == "base__none" else PRE[args.family]
                if pos not in ("last_prompt", pre):
                    continue
                c = d[d.checkpoint == ckpt]
                if c.empty:
                    continue
                X, y = activations.matrix(c, column), c.group.to_numpy()
                hard, source = c.hard.to_numpy(), c.source.to_numpy()
                key = {"checkpoint": ckpt, "layer": layer, "position": pos if pos == "last_prompt" else "pre_gen",
                       "token": config.position_label(args.family, pos, config.stage_of(ckpt))}

                # 3-way probe, out of fold
                pred = cross_val_predict(probes.make(), X, y, cv=cv)
                cm = confusion_matrix(y, pred, labels=GROUPS, normalize="true")
                row = {**key, "n": len(c), "bal_acc": balanced_accuracy_score(y, pred)}
                row.update({f"{a}_as_{b}": cm[i, j]
                            for i, a in enumerate(GROUPS) for j, b in enumerate(GROUPS)})
                is_p = y == "pseudo_harm"
                for h, name in [(1, "refused"), (0, "answered")]:
                    m = is_p & (hard == h)
                    row[f"n_p_{name}"] = int(m.sum())
                    for b in GROUPS:
                        row[f"p_{name}_as_{b}"] = (pred[m] == b).mean() if m.any() else np.nan

                # k-means, k = 3
                Z = X - X.mean(0)
                Z /= np.linalg.norm(Z, axis=1, keepdims=True)
                km = KMeans(3, n_init=10, random_state=0).fit_predict(Z)
                row.update({"ami_group": ami(y, km), "ami_hard": ami(hard, km),
                            "ami_source": ami(source, km)})
                rows.append(row)
                ct = pd.crosstab(km, y).reindex(columns=GROUPS, fill_value=0)
                ref_p = pd.Series(km[is_p & (hard == 1)]).value_counts()
                for k, counts in ct.iterrows():
                    cluster_rows.append({**key, "cluster": k, **counts.to_dict(),
                                         "pseudo_refused": int(ref_p.get(k, 0))})

                # leave one (source, group) out
                if args.no_loso:
                    continue
                for s, g in sorted(set(zip(source, y))):
                    test = (source == s) & (y == g)
                    if (y[~test] == g).sum() < MIN_TRAIN:
                        continue
                    pr = probes.make().fit(X[~test], y[~test]).predict(X[test])
                    loso_rows.append({**key, "source": s, "group": g, "n": int(test.sum()),
                                      **{f"as_{b}": (pr == b).mean() for b in GROUPS}})
            print(f"L{layer} {pos} done", flush=True)

    out = config.results_dir(args.family) / "geometry"
    save_atomic(pd.DataFrame(rows), out / "three_way.csv")
    save_atomic(pd.DataFrame(cluster_rows), out / "three_way_clusters.csv")
    if loso_rows:
        save_atomic(pd.DataFrame(loso_rows), out / "three_way_loso.csv")


if __name__ == "__main__":
    main()
