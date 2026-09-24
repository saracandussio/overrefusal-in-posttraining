"""Figure — axis measures across layers, one line per checkpoint, with 95% CIs.

Reads results/<family>/geometry/axis.csv (run scripts/axis.py first), so it
needs no activations. Positions go in columns, in reading order, so the
jump from the last template token to first_gen is visible at a glance.

    python scripts/fig_axis.py --family olmo2
    python scripts/fig_axis.py --family olmo2 --measures t t_refused t_answered
    python scripts/fig_axis.py --family olmo2 --positions last_prompt post_instr_3 post_instr_6 first_gen

Each row shares its y scale, so positions can be compared by eye.
"""
import pandas as pd

from overrefusal import cli, config, plotting
from overrefusal.activations import reading_order

DEFAULT = ["entanglement", "t", "off_axis", "cos_vbeh_vref", "cos_vbeh_vover_orth"]


def main():
    p = cli.parser(__doc__)
    p.add_argument("--measures", nargs="+", default=DEFAULT)
    p.add_argument("--positions", nargs="+", default=None, help="default: all in the file")
    args = p.parse_args()

    axis = pd.read_csv(config.results_dir(args.family) / "geometry" / "axis.csv")
    axis = axis[(axis.pseudo_source == "all") & axis.checkpoint.isin(args.checkpoints)]
    # base template tokens are not the chat ones: same index, different token
    axis = axis[~(axis.checkpoint.str.startswith("base") & axis.position.str.startswith("post_instr_"))]
    positions = reading_order(args.positions or axis.position)

    fig, axes = plotting.grid(len(args.measures), len(positions), size=2.4, share="row")
    for i, measure in enumerate(args.measures):
        for j, pos in enumerate(positions):
            ax = axes[i, j]
            here = axis[axis.position == pos]
            for ckpt in [c for c in args.checkpoints if c in set(here.checkpoint)]:
                d = here[here.checkpoint == ckpt].sort_values("layer")
                color = plotting.STAGE_COLOR.get(ckpt, "#333")
                ax.plot(d.layer, d[measure], marker="o", ms=3, color=color,
                        label=ckpt.removesuffix("__none"))
                if f"{measure}_lo" in d:
                    ax.fill_between(d.layer, d[f"{measure}_lo"], d[f"{measure}_hi"],
                                    color=color, alpha=0.15, lw=0)
            if i == 0:
                ax.set_title(config.position_label(args.family, pos), fontsize=8)
            if j == 0:
                ax.set_ylabel(measure)
    axes[0, -1].legend(frameon=False, fontsize=7)

    out = config.results_dir(args.family) / "figures" / "axis_by_layer.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()
