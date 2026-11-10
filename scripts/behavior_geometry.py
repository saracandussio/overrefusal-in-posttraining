"""Every behaviour-dependent result of the centroid/neighbour analysis, with
behaviour = refusal.response_type: refusal (GA = 0), weak refusal (GA = 1
or 2 with PD = 1), weak answer (GA = 1 with PD = 0), answer (GA = 2 with
PD = 0, or GA = 3). The eight raw (GA, PD) pairs go to the appendix files.

Reads results/<family>/centroids/distances_<ckpt>.parquet (run
scripts/centroid_distances.py first) and the judge columns of raw_results.csv.
Behaviour comes from raw_results.csv, not from the parquet (whose
response_type column may carry the labels of an older definition), so a
change of definition never needs the activations again.

Writes results/<family>/behavior_geometry/:

  rates.csv          checkpoint x group x response type: n and share of
                     coherent responses (no activations involved)
  by_class.csv       checkpoint x layer x position x group x response type:
                     n, near_harmful (share with closer_to = harmful, linear),
                     knn_10/20/50 means, knn_20 sd, zone shares of knn_20
  transitions.csv    SFT -> DPO, same prompt: group x layer x position x
                     response type in SFT x response type in DPO: n, knn_20
                     and near_harmful in both, zone shares of knn_20 in SFT
  transitions_by_source.csv  the same, split by source (counts per zone)
  appendix_rates_pairs.csv, appendix_by_pair.csv
                     rates.csv and by_class.csv with the eight (GA, PD) pairs

Every mean comes with n and sd; ci_lo / ci_hi are normal approximations
(mean +- 1.96 sd / sqrt n), meant to flag differences worth a bootstrap.

    python scripts/behavior_geometry.py --family olmo2
    python scripts/behavior_geometry.py --family olmo3 --positions last_prompt pre_gen
"""
import argparse
import logging

import numpy as np
import pandas as pd

from overrefusal import config, groups, refusal
from overrefusal.io import save_atomic

log = logging.getLogger(__name__)
ZONES = [(-0.01, 0.3, "harmless_like"), (0.3, 0.7, "mixed"), (0.7, 1.0, "harmful_like")]
KNN = ["knn_10", "knn_20", "knn_50"]


def zone(knn: pd.Series) -> pd.Series:
    return pd.cut(knn, [z[0] for z in ZONES] + [ZONES[-1][1]], labels=[z[2] for z in ZONES])


def judged(family: str) -> pd.DataFrame:
    """One row per (checkpoint, source, prompt): group, response type, (GA, PD) pair."""
    raw = pd.read_csv(config.raw_results_csv(family))
    raw = raw[~raw.source.isin(config.EXCLUDED_SOURCES)].drop_duplicates(refusal.KEY).copy()
    raw["group"] = groups.assign_group(raw)
    raw["response"] = refusal.response_type(raw)
    raw["pair"] = refusal.ga_pd_pair(raw)
    return raw[refusal.KEY + ["group", "response", "pair"]]


def with_ci(df: pd.DataFrame, mean: str, sd: str, n: str = "n") -> pd.DataFrame:
    half = 1.96 * df[sd] / np.sqrt(df[n])
    df[f"{mean}_ci_lo"], df[f"{mean}_ci_hi"] = df[mean] - half, df[mean] + half
    return df


def rates(judge: pd.DataFrame, col: str) -> pd.DataFrame:
    ok = judge[judge[col].notna()]
    counts = ok.groupby(["checkpoint", "group", col]).size().rename("n").reset_index()
    totals = ok.groupby(["checkpoint", "group"]).size().rename("n_total").reset_index()
    counts = counts.merge(totals, on=["checkpoint", "group"])
    counts["share"] = counts.n / counts.n_total
    counts["sd"] = np.sqrt(counts.share * (1 - counts.share))
    return with_ci(counts, "share", "sd", "n_total").drop(columns="sd")


def load_distances(family: str, ckpt: str, positions) -> pd.DataFrame:
    path = config.results_dir(family) / "centroids" / f"distances_{ckpt}.parquet"
    cols = ["checkpoint", "source", "prompt", "group", "layer", "position", "closer_to"] + KNN
    d = pd.read_parquet(path, columns=cols)
    if positions:
        d = d[d.position.isin(positions)]
    d = d.drop_duplicates(["source", "prompt", "layer", "position"])
    d["near_harmful"] = (d.closer_to == "harmful").astype(float)
    d["zone"] = zone(d.knn_20)
    return d


def summarise(d: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """n, means, sd and zone shares of every group defined by keys."""
    g = d.groupby(keys, observed=True)
    out = g.agg(n=("knn_20", "size"), near_harmful=("near_harmful", "mean"),
                **{k: (k, "mean") for k in KNN}, knn_20_sd=("knn_20", "std"),
                near_harmful_sd=("near_harmful", "std")).reset_index()
    zones = (d.groupby(keys + ["zone"], observed=True).size().unstack("zone", fill_value=0)
             .reindex(columns=[z[2] for z in ZONES], fill_value=0))
    zones = zones.div(zones.sum(axis=1), axis=0).add_prefix("zone_").reset_index()
    out = out.merge(zones, on=keys)
    out = with_ci(out, "knn_20", "knn_20_sd")
    return with_ci(out, "near_harmful", "near_harmful_sd")


def by_class(d: pd.DataFrame, col: str) -> pd.DataFrame:
    return summarise(d[d[col].notna()], ["checkpoint", "layer", "position", "group", col])


def transitions(sft: pd.DataFrame, dpo: pd.DataFrame, by_source: bool) -> pd.DataFrame:
    key = ["source", "prompt", "layer", "position"]
    cols = key + ["group", "response", "knn_20", "near_harmful", "zone"]
    m = sft[cols].merge(dpo[key + ["response", "knn_20", "near_harmful"]],
                        on=key, suffixes=("_sft", "_dpo"))
    m = m[m.response_sft.notna() & m.response_dpo.notna()]
    m = m[m.group.isin(["pseudo_harm", "harmful"])]
    keys = ["group", "layer", "position", "response_sft", "response_dpo"]
    if by_source:
        t = (m.groupby(keys + ["source", "zone"], observed=True).size()
             .unstack("zone", fill_value=0).reindex(columns=[z[2] for z in ZONES], fill_value=0))
        return t.add_prefix("n_").reset_index()
    g = m.groupby(keys, observed=True)
    out = g.agg(n=("knn_20_sft", "size"),
                knn_20_sft=("knn_20_sft", "mean"), knn_20_dpo=("knn_20_dpo", "mean"),
                knn_20_sft_sd=("knn_20_sft", "std"), knn_20_dpo_sd=("knn_20_dpo", "std"),
                near_harmful_sft=("near_harmful_sft", "mean"),
                near_harmful_dpo=("near_harmful_dpo", "mean")).reset_index()
    zones = (m.groupby(keys + ["zone"], observed=True).size().unstack("zone", fill_value=0)
             .reindex(columns=[z[2] for z in ZONES], fill_value=0))
    zones = zones.div(zones.sum(axis=1), axis=0).add_prefix("zone_sft_").reset_index()
    out = out.merge(zones, on=keys)
    out = with_ci(out, "knn_20_sft", "knn_20_sft_sd")
    return with_ci(out, "knn_20_dpo", "knn_20_dpo_sd")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", required=True, choices=["olmo2", "olmo3"])
    p.add_argument("--checkpoints", nargs="+", default=list(config.CHECKPOINTS))
    p.add_argument("--positions", nargs="+", default=["last_prompt", "pre_gen", "first_gen"],
                   help="positions to keep (default: the three discussed in the report)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out_dir = config.results_dir(args.family) / "behavior_geometry"
    out_dir.mkdir(parents=True, exist_ok=True)
    judge = judged(args.family)
    save_atomic(rates(judge, "response"), out_dir / "rates.csv")
    save_atomic(rates(judge, "pair"), out_dir / "appendix_rates_pairs.csv")
    log.info("rates written")

    merged, tables, pairs = {}, [], []
    for ckpt in args.checkpoints:
        d = load_distances(args.family, ckpt, args.positions)
        d = d.merge(judge.drop(columns="group"), on=refusal.KEY, how="left", validate="many_to_one")
        merged[ckpt] = d
        tables.append(by_class(d, "response"))
        pairs.append(by_class(d, "pair"))
        log.info("%s: %d rows, %d without a usable judgement", ckpt, len(d),
                 int(d.response.isna().sum()))
    save_atomic(pd.concat(tables, ignore_index=True), out_dir / "by_class.csv")
    save_atomic(pd.concat(pairs, ignore_index=True), out_dir / "appendix_by_pair.csv")
    log.info("by_class written")

    if {"sft__none", "dpo__none"} <= merged.keys():
        sft, dpo = merged["sft__none"], merged["dpo__none"]
        save_atomic(transitions(sft, dpo, by_source=False), out_dir / "transitions.csv")
        save_atomic(transitions(sft, dpo, by_source=True), out_dir / "transitions_by_source.csv")
        log.info("transitions written")


if __name__ == "__main__":
    main()
