"""Q5 — What can a linear read-out recover, and does it survive training?

Two probes per checkpoint x layer x position:

  group     harmful / pseudo_harm / harmless. Cross-validated accuracy, plus
            leave-one-source-out accuracy: the control for the probe having
            learnt the dataset instead of the concept (see probes.py).
  behavior  refused vs answered, on pseudo-harmful prompts only. Across all
            prompts this target is mostly the group itself; within pseudo it
            asks the real question. AUROC, overall and per source.

Transfer: each probe trained on one checkpoint is tested on every later one.

Output results/<family>/geometry/probes.csv, long format:
probe, metric, train_on, test_on, layer, position, source, value.

    python scripts/probe.py --family olmo2 --positions pre_gen first_gen
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from overrefusal import activations, cli, config, probes, refusal
from overrefusal.io import save_atomic


def main():
    p = cli.parser(__doc__, activations=True)
    p.add_argument("--no-loso", action="store_true", help="skip leave-one-source-out (slow)")
    args = p.parse_args()

    df = activations.load(args.family, args.layers, args.positions, args.checkpoints)
    df = refusal.attach(df, pd.read_csv(config.raw_results_csv(args.family)))

    rows = []
    add = lambda **r: rows.append({"source": "all", **r})
    for layer in args.layers:
        for pos in activations.expand(df, layer, args.positions):
            column = activations.col(layer, pos)
            fitted = {}
            for ckpt in args.checkpoints:
                d = df[(df.checkpoint == ckpt) & activations.has(df, column)]
                if d.empty:
                    continue
                X, y = activations.matrix(d, column), d.group.to_numpy()
                cell = dict(train_on=ckpt, test_on=ckpt, layer=layer, position=pos)
                add(probe="group", metric="cv_accuracy", value=probes.cv_accuracy(X, y), **cell)
                add(probe="group", metric="majority", value=probes.majority_baseline(
                    pd.factorize(y)[0]), **cell)
                if not args.no_loso:
                    for src, acc in probes.leave_one_source_out(X, y, d.source.to_numpy()).items():
                        rows.append(dict(probe="group", metric="loso_accuracy", source=src,
                                         value=acc, **cell))

                pseudo = (d.group == "pseudo_harm").to_numpy()
                Xp, yp, srcp = X[pseudo], d.refused.to_numpy()[pseudo], d.source.to_numpy()[pseudo]
                auroc, scores = probes.cv_auroc(Xp, yp)
                add(probe="behavior", metric="cv_auroc", value=auroc, **cell)
                add(probe="behavior", metric="refusal_rate", value=float(yp.mean()), **cell)
                for src in np.unique(srcp):
                    m = srcp == src
                    if len(set(yp[m])) == 2:
                        rows.append(dict(probe="behavior", metric="cv_auroc", source=src,
                                         value=roc_auc_score(yp[m], scores[m]), **cell))
                fitted[ckpt] = (probes.fit(X, y), probes.fit(Xp, yp))

            # transfer to later checkpoints (training order)
            order = [c for c in args.checkpoints if c in fitted]
            for i, train in enumerate(order):
                for test in order[i + 1:]:
                    d = df[(df.checkpoint == test) & activations.has(df, column)]
                    X = activations.matrix(d, column)
                    pseudo = (d.group == "pseudo_harm").to_numpy()
                    group_clf, beh_clf = fitted[train]
                    cell = dict(train_on=train, test_on=test, layer=layer, position=pos)
                    add(probe="group", metric="transfer_accuracy",
                        value=probes.transfer_accuracy(group_clf, X, d.group.to_numpy()), **cell)
                    add(probe="behavior", metric="transfer_auroc", value=roc_auc_score(
                        d.refused.to_numpy()[pseudo], beh_clf.predict_proba(X[pseudo])[:, 1]), **cell)

    out = pd.DataFrame(rows)
    save_atomic(out, config.results_dir(args.family) / "geometry" / "probes.csv")
    within = out[(out.train_on == out.test_on) & (out.source == "all")]
    print(within.pivot_table(index=["probe", "metric", "position", "train_on"], columns="layer",
                             values="value", sort=False).round(3).to_string())


if __name__ == "__main__":
    main()
