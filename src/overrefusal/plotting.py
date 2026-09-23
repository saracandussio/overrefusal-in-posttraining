"""One look for every figure: colours carry (group, behaviour)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# (group, refused) -> colour. Warm = refused, cool = answered.
COLOR = {
    ("harmful", 1):     "#c1272d",   # refused, correctly
    ("harmful", 0):     "#f2a7a0",   # answered: safety miss
    ("pseudo_harm", 1): "#e8871e",   # refused: over-refusal
    ("pseudo_harm", 0): "#2a9d8f",   # answered, correctly
    ("harmless", 1):    "#7b2d8e",   # refused: severe over-refusal
    ("harmless", 0):    "#3d6fa8",   # answered, correctly
}
MARKER = {"harmful": "x", "pseudo_harm": "^", "harmless": "o"}
LABEL = {
    ("harmful", 1): "harmful, refused", ("harmful", 0): "harmful, answered",
    ("pseudo_harm", 1): "pseudo, refused", ("pseudo_harm", 0): "pseudo, answered",
    ("harmless", 1): "harmless, refused", ("harmless", 0): "harmless, answered",
}
STAGE_COLOR = {"base__none": "#8c8c8c", "sft__none": "#c1272d",
               "dpo__none": "#3d6fa8", "final__none": "#2a9d8f"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})


def grid(n_rows: int, n_cols: int, size: float = 2.6, share: bool = True):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(size * n_cols, size * n_rows),
                             squeeze=False, sharex=share, sharey=share)
    return fig, axes


def legend(fig) -> None:
    handles = [Line2D([], [], marker=MARKER[g], color=c, linestyle="", label=LABEL[g, r])
               for (g, r), c in COLOR.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
