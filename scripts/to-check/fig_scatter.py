"""Figure — every prompt as a point, checkpoints in rows, layers in columns.

  --basis refusal   x = position on the refusal axis (0 harmless centroid,
                    1 harmful centroid), y = along v_over with its v_ref part
                    removed, in the same units. Replaces plot_2d_refusal_space*.py.
  --basis pca       first two principal components of that cell.
                    Replaces plot_pca_umap.py (each panel has its own basis,
                    so compare shapes, not coordinates, across panels).

Colour = group x judge behaviour (see plotting.py).

    python scripts/fig_scatter.py --family olmo2 --position pre_gen --layers 8 19 26 31
"""
import numpy as np
import pandas as pd

from overrefusal import activations, cli, config, geometry, plotting, refusal


def coordinates(X, groups, basis):
    if basis == "pca":
        Xc = X - X.mean(0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        return Xc @ Vt[:2].T
    mu = {g: X[groups == g].mean(0) for g in ("harmless", "harmful", "pseudo_harm")}
    axis = mu["harmful"] - mu["harmless"]
    side = geometry.unit(geometry.remove_component(mu["pseudo_harm"] - mu["harmless"], axis))
    rel = X - mu["harmless"]
    return np.column_stack([rel @ axis / (axis @ axis), rel @ side / np.linalg.norm(axis)])


def main():
    p = cli.parser(__doc__, activations=True)
    p.add_argument("--position", default="pre_gen")
    p.add_argument("--basis", choices=["refusal", "pca"], default="refusal")
    p.add_argument("--sample", type=int, default=1500, help="points per panel")
    args = p.parse_args()

    df = activations.load(args.family, args.layers, [args.position], args.checkpoints)
    df = refusal.attach(df, pd.read_csv(config.raw_results_csv(args.family)))
    rng = np.random.default_rng(0)

    fig, axes = plotting.grid(len(args.checkpoints), len(args.layers),
                              share=args.basis == "refusal")
    for i, ckpt in enumerate(args.checkpoints):
        for j, layer in enumerate(args.layers):
            ax, column = axes[i, j], activations.col(layer, args.position)
            d = df[(df.checkpoint == ckpt) & activations.has(df, column)]
            if d.empty:
                ax.set_axis_off()
                continue
            xy = coordinates(activations.matrix(d, column), d.group.to_numpy(), args.basis)
            show = rng.permutation(len(d))[:args.sample]
            for (g, r), color in plotting.COLOR.items():
                m = show[((d.group == g) & (d.refused == r)).to_numpy()[show]]
                ax.scatter(*xy[m].T, s=5, c=color, marker=plotting.MARKER[g],
                           alpha=0.5, linewidths=0.6)
            if args.basis == "refusal":
                for x in (0, 1):
                    ax.axvline(x, color="#999", lw=0.6, ls=":")
            if i == 0:
                ax.set_title(f"layer {layer}")
            if j == 0:
                ax.set_ylabel(ckpt.removesuffix("__none"))
    plotting.legend(fig)
    fig.suptitle(f"{args.family}, {config.position_label(args.family, args.position)}, "
                 f"{args.basis} basis")

    out = config.results_dir(args.family) / "figures" / f"scatter_{args.basis}_{args.position}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()
