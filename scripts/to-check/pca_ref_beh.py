"""Quick PCA of hidden states, coloured by reality (group) and by behaviour (hard refusal).

One figure per layer: rows = positions, columns = checkpoint x {ref, beh}.
PCA is fitted once per checkpoint and position on harmful, harmless and
pseudo-harmful prompts; the two panels of a pair show the same points.
Axis labels give the variance explained by PC1 and PC2.
Base template tokens differ from the post-trained ones: only its last token
(':') is shown, on the pre_gen row. OLMo 2's two '|' tokens are skipped.

    python scripts/pca_ref_beh.py --family olmo2 --layers 16
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

from overrefusal import activations, cli, config, refusal
from overrefusal.io import save_atomic

GROUPS = {"harmless": "tab:blue", "harmful": "tab:red", "pseudo_harm": "tab:orange"}
BEH = {0: ("answered", "lightgray"), 1: ("hard refusal", "black")}
MIXED = "wildguard|toxic"
PRE = {"olmo2": "post_instr_6", "olmo3": "post_instr_4"}
SKIP = {"olmo2": {"post_instr_2", "post_instr_4", "pre_gen"}, "olmo3": {"pre_gen"}}
BASE_PRE = "post_instr_2"


def source_pos(family, ckpt, row):
    """Position to draw for this checkpoint on this row (None = empty panel)."""
    if ckpt != "base__none" or row in ("last_prompt", "first_gen"):
        return row
    return BASE_PRE if row == PRE[family] else None


def main():
    p = cli.parser(__doc__, activations=True)
    p.set_defaults(checkpoints=["base__none", "sft__none", "dpo__none", "final__none"],
                   positions=["last_prompt", "post_instr", "first_gen"])
    args = p.parse_args()

    df = activations.load(args.family, args.layers, args.positions, args.checkpoints)
    df = refusal.attach(df, pd.read_csv(config.raw_results_csv(args.family)))
    df = df.drop_duplicates(["checkpoint", "prompt"])
    df = df[df.group.isin(list(GROUPS)) & ~df.source.str.contains(MIXED, case=False)]
    df["hard"] = (df.response_type == "hard refusal").astype(int)

    fig_dir = config.results_dir(args.family) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    var_rows = []
    for layer in args.layers:
        rows = [q for q in activations.expand(df, layer, args.positions) if q not in SKIP[args.family]]
        ckpts = args.checkpoints
        fig, axes = plt.subplots(len(rows), 2 * len(ckpts), squeeze=False,
                                 figsize=(2.4 * 2 * len(ckpts), 2.4 * len(rows)))
        for r, row in enumerate(rows):
            for j, ckpt in enumerate(ckpts):
                a_ref, a_beh = axes[r, 2 * j], axes[r, 2 * j + 1]
                pos = source_pos(args.family, ckpt, row)
                column = activations.col(layer, pos) if pos else None
                c = df[df.checkpoint == ckpt]
                if column is None or column not in c.columns or not activations.has(c, column).any():
                    a_ref.axis("off"), a_beh.axis("off")
                    continue
                c = c[activations.has(c, column)]
                X = activations.matrix(c, column)
                pca = PCA(2).fit(X)
                Z = pca.transform(X)
                v1, v2 = 100 * pca.explained_variance_ratio_

                for g, colour in GROUPS.items():
                    m = (c.group == g).to_numpy()
                    a_ref.scatter(Z[m, 0], Z[m, 1], s=2, alpha=0.4, c=colour, label=g, rasterized=True)
                for h, (name, colour) in BEH.items():
                    m = (c.hard == h).to_numpy()
                    a_beh.scatter(Z[m, 0], Z[m, 1], s=2, alpha=0.4, c=colour, label=name, rasterized=True)

                token = config.position_label(args.family, pos, config.stage_of(ckpt))
                for a, kind in ((a_ref, "ref"), (a_beh, "beh")):
                    a.set_title(f"{config.stage_of(ckpt)} | {token} | {kind}", fontsize=7)
                    a.set_xlabel(f"PC1 {v1:.1f}%", fontsize=6)
                    a.set_ylabel(f"PC2 {v2:.1f}%", fontsize=6)
                    a.tick_params(labelsize=5)
                var_rows.append({"layer": layer, "row": row, "position": pos, "token": token,
                                 "checkpoint": ckpt, "n": len(c), "pc1": v1, "pc2": v2})
            print(f"L{layer} {row} done", flush=True)

        handles = [h for a in (axes[0, 0], axes[0, 1]) for h in a.get_legend_handles_labels()[0]]
        labels = [l for a in (axes[0, 0], axes[0, 1]) for l in a.get_legend_handles_labels()[1]]
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), markerscale=4, fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        fig.savefig(fig_dir / f"pca_ref_beh_L{layer}.png", dpi=150)
        plt.close(fig)

    save_atomic(pd.DataFrame(var_rows), fig_dir / "pca_ref_beh_variance.csv")


if __name__ == "__main__":
    main()
