"""Figures — flows of prompts between categories across checkpoints (alluvial plots).

Three kinds, one file each, panels = groups (pseudo, harmful by default):

  behaviour.{png,pdf}
      base -> sft -> dpo -> final, categories = refusal.response_type (judge).
  geometry_<position>_L<layer>.{png,pdf}
      the same checkpoints, categories = zone of knn_20 (harmless-like < 0.3,
      mixed 0.3-0.7, harmful-like > 0.7). No judge involved. Each checkpoint
      is measured against its own reference prompts, so a ribbon that stays
      horizontal means "same place relative to the references".
      The base is left out at pre_gen (its template tokens are different) and
      marked as indicative at last_prompt (different frame before the prompt).
  combined_<position>_L<layer>.{png,pdf}
      behaviour sft -> dpo, each ribbon coloured by the prompt's zone in SFT
      at that position: both signals in one picture.

Only prompts with a value in every column of a panel are drawn; n is in the
panel title. Behaviour comes from raw_results.csv, knn_20 from
results/<family>/centroids/distances_<ckpt>.parquet.

    python scripts/fig_flows.py --family olmo2
    python scripts/fig_flows.py --family olmo3 --layer 19 --groups pseudo_harm harmful harmless
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from overrefusal import config, groups, refusal

BEHAVIOUR = ["refusal", "weak refusal", "weak answer", "answer"]
BEHAVIOUR_COLOURS = {"refusal": "#b2182b", "weak refusal": "#ef8a62",
                     "weak answer": "#67a9cf", "answer": "#2166ac"}
ZONES = ["harmful-like", "mixed", "harmless-like"]
ZONE_COLOURS = {"harmful-like": "#762a83", "mixed": "#bababa", "harmless-like": "#1b7837"}
STAGE_NAMES = {f"{s}__{v}": n for s, n in [("base", "base"), ("sft", "SFT"), ("dpo", "DPO"), ("final", "final")]
               for v in ("none", "nosys")}
GROUP_NAMES = {"pseudo_harm": "pseudo", "harmful": "harmful", "harmless": "harmless"}
KEY = ["source", "prompt"]


def zone(knn: pd.Series) -> pd.Series:
    out = pd.Series("mixed", index=knn.index, dtype=object)
    out[knn < 0.3] = "harmless-like"
    out[knn > 0.7] = "harmful-like"
    out[knn.isna()] = np.nan
    return out


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def behaviour_table(family: str) -> pd.DataFrame:
    """One row per (source, prompt): group and response type per checkpoint."""
    raw = pd.read_csv(config.raw_results_csv(family))
    raw = raw[~raw.source.isin(config.EXCLUDED_SOURCES)].drop_duplicates(refusal.KEY).copy()
    raw["group"] = groups.assign_group(raw)
    raw["response"] = refusal.response_type(raw)
    wide = raw.pivot_table(index=KEY + ["group"], columns="checkpoint", values="response",
                           aggfunc="first")
    return wide.reset_index()


def zone_table(family: str, layer: int, position: str, checkpoints) -> pd.DataFrame:
    """One row per (source, prompt): group and knn_20 zone per checkpoint."""
    parts = []
    for ckpt in checkpoints:
        path = config.results_dir(family) / "centroids" / f"distances_{ckpt}.parquet"
        if not path.exists():
            continue
        d = pd.read_parquet(path, columns=KEY + ["group", "layer", "position", "knn_20"])
        d = d[(d.layer == layer) & (d.position == position)].drop_duplicates(KEY)
        parts.append(d.assign(checkpoint=ckpt, zone=zone(d.knn_20))[KEY + ["group", "checkpoint", "zone"]])
    long = pd.concat(parts, ignore_index=True)
    return long.pivot_table(index=KEY + ["group"], columns="checkpoint", values="zone",
                            aggfunc="first").reset_index()


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------
def _band(ax, x0, x1, y0a, y0b, y1a, y1b, colour, alpha):
    """A ribbon from [y0a, y0b] at x0 to [y1a, y1b] at x1, smooth in between."""
    t = np.linspace(0, 1, 60)
    s = 3 * t ** 2 - 2 * t ** 3
    xs = x0 + (x1 - x0) * t
    lo = y0a + (y1a - y0a) * s
    hi = y0b + (y1b - y0b) * s
    ax.fill_between(xs, lo, hi, color=colour, alpha=alpha, linewidth=0)


def alluvial(ax, df: pd.DataFrame, stages: list[str], categories: list[str], colours: dict,
             ribbon_by: str | None = None, ribbon_colours: dict | None = None,
             ribbon_order: list[str] | None = None, gap: float = 0.02, width: float = 0.08):
    """Columns = stages (column names of df), stacked bars = categories (top to bottom),
    ribbons between consecutive stages. Heights are shares of len(df)."""
    n = len(df)
    xs = np.arange(len(stages), dtype=float)
    # node extents per stage: top-down stacking, y in [0, 1 + gaps]
    nodes = {}
    for i, st in enumerate(stages):
        counts = df[st].value_counts().reindex(categories, fill_value=0)
        y = 1 + gap * (len(categories) - 1)
        for c in categories:
            h = counts[c] / n
            nodes[(i, c)] = [y - h, y, counts[c]]
            y -= h + gap
    # ribbons
    for i in range(len(stages) - 1):
        a, b = stages[i], stages[i + 1]
        out_cursor = {c: nodes[(i, c)][1] for c in categories}
        in_cursor = {c: nodes[(i + 1, c)][1] for c in categories}
        keys = [a, b] + ([ribbon_by] if ribbon_by else [])
        flows = df.groupby(keys).size()
        for ca in categories:
            for cb in categories:
                subs = ribbon_order if ribbon_by else [None]
                for sub in subs:
                    k = (ca, cb, sub) if ribbon_by else (ca, cb)
                    m = flows.get(k, 0)
                    if m == 0:
                        continue
                    h = m / n
                    y0b = out_cursor[ca]; y0a = y0b - h; out_cursor[ca] = y0a
                    y1b = in_cursor[cb]; y1a = y1b - h; in_cursor[cb] = y1a
                    colour = ribbon_colours[sub] if ribbon_by else colours[ca]
                    _band(ax, xs[i] + width / 2, xs[i + 1] - width / 2, y0a, y0b, y1a, y1b,
                          colour, 0.55 if ribbon_by else 0.35)
    # bars and labels
    for (i, c), (lo, hi, cnt) in nodes.items():
        if cnt == 0:
            continue
        ax.add_patch(plt.Rectangle((xs[i] - width / 2, lo), width, hi - lo,
                                   color=colours[c], linewidth=0))
        if hi - lo > 0.025:
            ax.text(xs[i] + width / 2 + 0.03, (lo + hi) / 2, f"{cnt}", va="center", fontsize=7)
    ax.set_xlim(-0.3, len(stages) - 0.5)
    ax.set_ylim(-0.02, 1 + gap * (len(categories) - 1) + 0.02)
    ax.set_xticks(xs)
    ax.set_xticklabels([STAGE_NAMES.get(s, s) for s in stages])
    ax.set_yticks([])
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)


def legend(fig, colours: dict, order: list[str], title: str, y: float):
    handles = [plt.Rectangle((0, 0), 1, 1, color=colours[c]) for c in order]
    fig.legend(handles, order, title=title, loc="lower center", ncol=len(order),
               bbox_to_anchor=(0.5, y), frameon=False, fontsize=8, title_fontsize=8)


def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", path.with_suffix(".png"))


def panels(df_all, group_list, stages, categories, colours, title, out, **kw):
    fig, axes = plt.subplots(1, len(group_list), figsize=(4.2 * len(group_list), 4.2), squeeze=False)
    for ax, g in zip(axes[0], group_list):
        df = df_all[df_all.group == g].dropna(subset=stages + ([kw["ribbon_by"]] if kw.get("ribbon_by") else []))
        alluvial(ax, df, stages, categories, colours, **kw)
        ax.set_title(f"{GROUP_NAMES.get(g, g)} (n = {len(df)})", fontsize=10)
    fig.suptitle(title, fontsize=10, y=1.0 + 0.04 * title.count("\n") + 0.02)
    return fig


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", required=True, choices=["olmo2", "olmo3"])
    p.add_argument("--layer", type=int, default=16)
    p.add_argument("--positions", nargs="+", default=["last_prompt", "pre_gen"])
    p.add_argument("--groups", nargs="+", default=["pseudo_harm", "harmful"])
    args = p.parse_args()
    out_dir = config.figures_dir(args.family) / "flows"
    ckpts = list(config.CHECKPOINTS)

    beh = behaviour_table(args.family)
    stages = [c for c in ckpts if c in beh.columns]
    fig = panels(beh, args.groups, stages, BEHAVIOUR, BEHAVIOUR_COLOURS,
                 f"{args.family}: response type per checkpoint (judge; refusal = GA 0)", None)
    legend(fig, BEHAVIOUR_COLOURS, BEHAVIOUR, "response type", -0.06)
    save(fig, out_dir / "behaviour")

    for pos in args.positions:
        geo_ckpts = [c for c in ckpts if not (pos != "last_prompt" and c.startswith("base"))]
        zt = zone_table(args.family, args.layer, pos, geo_ckpts)
        g_stages = [c for c in geo_ckpts if c in zt.columns]
        note = " (base: different frame, indicative)" if "base__none" in g_stages else ""
        fig = panels(zt, args.groups, g_stages, ZONES, ZONE_COLOURS,
                     f"{args.family}: knn_20 zone at {pos}, layer {args.layer}{note}\n"
                     "each checkpoint measured against its own reference prompts", None)
        legend(fig, ZONE_COLOURS, ZONES, "zone (knn_20: < 0.3, 0.3-0.7, > 0.7)", -0.08)
        save(fig, out_dir / f"geometry_{pos}_L{args.layer}")

        sft_col = [c for c in ckpts if config.stage_of(c) == "sft"][0]
        both = beh.merge(zt[KEY + [sft_col]].rename(columns={sft_col: "zone_sft"}), on=KEY)
        sft_tag, dpo_tag = [c for c in ckpts if config.stage_of(c) in ("sft", "dpo")]
        fig = panels(both, args.groups, [sft_tag, dpo_tag], BEHAVIOUR, BEHAVIOUR_COLOURS,
                     f"{args.family}: response type SFT -> DPO, ribbons coloured by zone in SFT "
                     f"at {pos}, layer {args.layer}", None,
                     ribbon_by="zone_sft", ribbon_colours=ZONE_COLOURS, ribbon_order=ZONES)
        legend(fig, BEHAVIOUR_COLOURS, BEHAVIOUR, "response type (bars)", -0.06)
        legend(fig, ZONE_COLOURS, ZONES, "zone in SFT (ribbons)", -0.16)
        save(fig, out_dir / f"combined_{pos}_L{args.layer}")


if __name__ == "__main__":
    main()
