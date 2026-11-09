"""Q4 — Which training stage moves the representations?

For each group, layer and position: cosine between the group's centroid in
two checkpoints. Near 1 means that stage left the group where it was.
Behaviour changes between SFT and DPO while these stay near 1 is the
"two levels" argument: DPO moves the read-out, not the representation.

Caveat built in: raw residual-stream centroids share a large common
component, so their cosines sit near 1 almost by construction. The same
table is therefore also computed for the *directions* v_ref and v_over
(differences of centroids), where that component cancels. Trust the
direction rows more.

    python scripts/drift.py --family olmo2
"""
from itertools import combinations

import pandas as pd

from overrefusal import activations, cli, config, geometry
from overrefusal.groups import GROUPS
from overrefusal.io import save_atomic


def main():
    args = cli.parser(__doc__, activations=True).parse_args()
    df = activations.load(args.family, args.layers, args.positions, args.checkpoints)

    rows = []
    for layer in args.layers:
        for pos in activations.expand(df, layer, args.positions):
            column = activations.col(layer, pos)
            # only checkpoints that have this position (base has no template tokens)
            ckpts = [c for c in args.checkpoints
                     if activations.has(df[df.checkpoint == c], column).any()]
            pairs = list(combinations(ckpts, 2))
            mu = {(c, g): activations.matrix(df[(df.checkpoint == c) & (df.group == g)],
                                             column).mean(0)
                  for c in ckpts for g in GROUPS}
            vectors = {g: {c: mu[c, g] for c in ckpts} for g in GROUPS}
            vectors["v_ref"] = {c: mu[c, "harmful"] - mu[c, "harmless"] for c in ckpts}
            vectors["v_over"] = {c: mu[c, "pseudo_harm"] - mu[c, "harmless"] for c in ckpts}
            for name, per_ckpt in vectors.items():
                for pair, value in geometry.centroid_drift(per_ckpt, pairs).items():
                    rows.append({"layer": layer, "position": pos, "vector": name,
                                 "pair": pair, "cosine": value})

    out = pd.DataFrame(rows)
    save_atomic(out, config.results_dir(args.family) / "geometry" / "drift.csv")
    print(out.pivot_table(index=["position", "vector", "layer"], columns="pair",
                          values="cosine", sort=False).round(3).to_string())


if __name__ == "__main__":
    main()
