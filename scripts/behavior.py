"""Q1 — How does refusal change across post-training stages?

For every checkpoint x group (and x source) reports the judge refusal rate
with a 95% Wilson interval, over coherent responses only, and next to it:
how many responses were incoherent (dropped), mean GA, PD rate, and the
keyword-detector rate as a cross-check.

Outputs results/<family>/behavior_by_group.csv and behavior_by_source.csv.

    python scripts/behavior.py --family olmo2
    python scripts/behavior.py --family olmo3 --checkpoints sft__none dpo__none

Rates are compared on the sources that every selected checkpoint has;
dropped sources are printed. Pass --all-sources to keep everything.
"""
import numpy as np
import pandas as pd

from overrefusal import cli, config
from overrefusal.groups import GROUPS, assign_group
from overrefusal.io import save_atomic
from overrefusal.refusal import judge_refusal


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def summarize(g: pd.DataFrame) -> pd.Series:
    scored = g[g.refused.notna()]
    k, n = int(scored.refused.sum()), len(scored)
    lo, hi = wilson(k, n)
    return pd.Series({
        "n": len(g),
        "incoherent": float(1 - n / len(g)),
        "refusal": k / n if n else np.nan, "refusal_lo": lo, "refusal_hi": hi,
        "mean_ga": scored.judge_ga.mean(),
        "pd_rate": scored.judge_pd.mean(),
        "keyword_refusal": g.predicted_refusal.mean(),
    })


def main():
    p = cli.parser(__doc__)
    p.add_argument("--all-sources", action="store_true")
    args = p.parse_args()
    df = pd.read_csv(config.raw_results_csv(args.family))
    df = df[df.checkpoint.isin(args.checkpoints) & ~df.source.isin(config.EXCLUDED_SOURCES)].copy()
    if not args.all_sources:
        per_ckpt = df.groupby("checkpoint").source.agg(set)
        common = set.intersection(*per_ckpt)
        dropped = set(df.source) - common
        if dropped:
            print(f"not in every checkpoint, left out: {sorted(dropped)}")
        df = df[df.source.isin(common)].copy()
    df["group"] = assign_group(df)
    df["refused"] = judge_refusal(df)

    by = lambda keys: (df.groupby(keys, sort=False)[df.columns.tolist()]
                         .apply(summarize).reset_index())
    groups, sources = by(["checkpoint", "group"]), by(["checkpoint", "group", "source"])
    out = config.results_dir(args.family)
    save_atomic(groups, out / "behavior_by_group.csv")
    save_atomic(sources, out / "behavior_by_source.csv")

    table = groups.pivot(index="checkpoint", columns="group", values="refusal")
    table = table.reindex(index=[c for c in args.checkpoints if c in table.index], columns=GROUPS)
    headline = pd.DataFrame({
        "refuses harmful": table.harmful,
        "answers pseudo": 1 - table.pseudo_harm,
        "answers harmless": 1 - table.harmless,
        "incoherent": groups.groupby("checkpoint").apply(
            lambda g: np.average(g.incoherent, weights=g.n)),
    }).reindex(table.index)
    print(headline.round(3).to_string())


if __name__ == "__main__":
    main()
