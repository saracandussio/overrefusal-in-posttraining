"""Distances of every prompt to the harmless and harmful centroids, per cell.

For each checkpoint, layer and position, runs centroids.measure_frame (see
src/overrefusal/centroids.py for the definitions) and writes, under
results/<family>/centroids/:

  distances_<checkpoint>.parquet  one row per prompt x layer x position:
      metadata (prompt, source, category, label, group), the judge's
      behaviour (refused, response_type; NaN if incoherent or unjudged),
      and d_harmless, d_harmful, z_harmless, z_harmful, closer_to
      (linear: closer_to is one hyperplane), plus knn_10, knn_20, knn_50
      (non-linear: balanced harmful share among the k nearest reference prompts)
  summary_<checkpoint>.csv        one row per layer x position: separation of
      the two centroids, the spread of each reference group, and the mean
      knn_20 of each reference group (sanity check: harmful well above 0.5)
  summary.csv                     all checkpoints' summaries together

Reference groups are defined by labels, so prompts are kept whether or not
the judge scored them; behaviour is attached, never used to select rows.

A checkpoint whose parquet already exists is skipped (rerun resumes);
--overwrite recomputes it. Activations are loaded one layer at a time.

    python scripts/centroid_distances.py --family olmo2
    python scripts/centroid_distances.py --family olmo3 --checkpoints sft__none dpo__none --layers 16
    python scripts/centroid_distances.py --family olmo2 --pca 128 --tag pca128

--pca N computes the neighbours after a PCA to N dimensions (fitted on
reference prompts only): a robustness check for distance concentration.
Use it with --tag, so the files get a suffix and do not overwrite the main run.

--add-knn adds the neighbour columns to files that already exist, without
touching anything already in them (distances, z, closer_to stay as they
were). Activations still have to be loaded, but nothing is recomputed.
With --tag the new columns get it as a suffix, so a robustness variant
lives next to the main one in the same file:

    python scripts/centroid_distances.py --family olmo2 --add-knn
    python scripts/centroid_distances.py --family olmo2 --add-knn --pca 128 --tag pca128
      -> knn_10_pca128, knn_20_pca128, knn_50_pca128
"""
import logging
import os

import pandas as pd

from overrefusal import activations, centroids, cli, config, refusal
from overrefusal.io import save_atomic

log = logging.getLogger(__name__)
META = ["prompt", "source", "category", "label", "group"]


def behaviour(family: str) -> pd.DataFrame:
    """Judge columns per (checkpoint, source, prompt), with refused and response_type."""
    raw = pd.read_csv(config.raw_results_csv(family))
    cols = refusal.KEY + [c for c in ["judge_ga", "judge_pd", "is_coherent"] if c in raw]
    raw = raw[cols].drop_duplicates(refusal.KEY)
    raw["refused"] = refusal.judge_refusal(raw)
    raw["response_type"] = refusal.response_type(raw)
    return raw[refusal.KEY + ["refused", "response_type"]]


def save_parquet(df: pd.DataFrame, path) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def run_checkpoint(family, ckpt, layers, positions, judged, pca):
    stage = config.stage_of(ckpt)
    rows, summaries = [], []
    for layer in layers:
        df = activations.load(family, [layer], positions, [ckpt])
        df = df.merge(judged, on=refusal.KEY, how="left", validate="many_to_one")
        for pos in activations.expand(df, layer, positions):
            column = activations.col(layer, pos)
            per_row, summary = centroids.measure_frame(df, column)
            per_row = per_row.join(centroids.neighbours_frame(df, column, pca=pca))
            for g in centroids.REFERENCE:
                summary[f"knn20_{g}_mean"] = float(per_row.knn_20[per_row.group == g].mean())
            cell = {"checkpoint": ckpt, "layer": layer, "position": pos,
                    "position_label": config.position_label(family, pos, stage)}
            out = df.loc[per_row.index, [c for c in META if c != "group"]
                         + ["refused", "response_type"]].join(per_row)
            rows.append(out.assign(**cell))
            summaries.append({**cell, **summary})
            log.info("%s L%d %-14s separation %.3f  harmless sd %.3f  harmful sd %.3f"
                     "  knn20 harmless %.2f harmful %.2f",
                     ckpt, layer, pos, summary["separation"], summary["harmless_sd"],
                     summary["harmful_sd"], summary["knn20_harmless_mean"],
                     summary["knn20_harmful_mean"])
        del df
    return pd.concat(rows, ignore_index=True), pd.DataFrame(summaries)


def _occurrence(df: pd.DataFrame, keys: list[str]) -> pd.Series:
    """Running count within duplicate keys: a prompt listed twice in a source
    gets 0 and 1, in file order, so rows pair up exactly."""
    return df.groupby(keys, sort=False).cumcount()


def add_knn(family, ckpt, pca, suffix, dist_path, summary_path):
    """Neighbour columns appended to an existing distances file (and summary)."""
    dist = pd.read_parquet(dist_path)
    names = [f"knn_{k}{suffix}" for k in centroids.KNN_KS]
    if set(names) <= set(dist.columns):
        log.info("%s: %s already there, skipping", ckpt, ", ".join(names))
        return
    key = ["source", "prompt", "layer", "position"]
    pieces = []
    for layer in sorted(dist.layer.unique()):
        positions = dist.loc[dist.layer == layer, "position"].unique().tolist()
        df = activations.load(family, [int(layer)], positions, [ckpt])
        for pos in positions:
            knn = centroids.neighbours_frame(df, activations.col(layer, pos), pca=pca)
            knn.columns = [f"{c}{suffix}" for c in knn.columns]
            knn = df.loc[knn.index, ["source", "prompt"]].join(knn).assign(layer=layer, position=pos)
            pieces.append(knn)
            log.info("%s L%d %-14s knn added", ckpt, layer, pos)
        del df
    new = pd.concat(pieces, ignore_index=True)
    new["occ"] = _occurrence(new, key)
    dist["occ"] = _occurrence(dist, key)
    merged = dist.drop(columns=names, errors="ignore").merge(
        new, on=key + ["occ"], how="left", validate="one_to_one").drop(columns="occ")
    if len(merged) != len(dist) or merged[names].isna().all(axis=1).any():
        raise RuntimeError(f"{ckpt}: rows do not pair up with the stored file; nothing written")
    save_parquet(merged, dist_path)

    if summary_path.exists():
        means = (merged[merged.group.isin(centroids.REFERENCE)]
                 .groupby(["layer", "position", "group"])[f"knn_20{suffix}"].mean()
                 .unstack("group").add_prefix(f"knn20{suffix}_").add_suffix("_mean").reset_index())
        summary = pd.read_csv(summary_path)
        summary = summary.drop(columns=[c for c in means if c not in ("layer", "position")],
                               errors="ignore").merge(means, on=["layer", "position"], how="left")
        save_atomic(summary, summary_path)


def main():
    p = cli.parser(__doc__, activations=True)
    p.set_defaults(positions=["last_prompt", "post_instr", "pre_gen", "first_gen"])
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--pca", type=int, default=None, help="neighbours after PCA to N dims")
    p.add_argument("--tag", default="", help="suffix for output files, e.g. pca128 "
                   "(with --add-knn: suffix for the new columns)")
    p.add_argument("--add-knn", action="store_true",
                   help="add neighbour columns to existing files, recompute nothing else")
    args = p.parse_args()
    for noisy in ("httpx", "huggingface_hub", "datasets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    suffix = f"_{args.tag}" if args.tag else ""

    out_dir = config.results_dir(args.family) / "centroids"
    out_dir.mkdir(parents=True, exist_ok=True)
    judged = behaviour(args.family)

    if args.add_knn:
        for ckpt in args.checkpoints:
            dist_path = out_dir / f"distances_{ckpt}.parquet"
            if not dist_path.exists():
                log.info("%s: no distances file, run without --add-knn first", ckpt)
                continue
            add_knn(args.family, ckpt, args.pca, suffix,
                    dist_path, out_dir / f"summary_{ckpt}.csv")
        parts = [pd.read_csv(out_dir / f"summary_{c}.csv") for c in args.checkpoints
                 if (out_dir / f"summary_{c}.csv").exists()]
        if parts:
            save_atomic(pd.concat(parts, ignore_index=True), out_dir / "summary.csv")
        return

    for ckpt in args.checkpoints:
        dist_path = out_dir / f"distances_{ckpt}{suffix}.parquet"
        if dist_path.exists() and not args.overwrite:
            log.info("%s: already done, skipping (--overwrite to redo)", ckpt)
            continue
        dist, summary = run_checkpoint(args.family, ckpt, args.layers, args.positions,
                                       judged, args.pca)
        n_unjudged = int(dist.drop_duplicates(["prompt", "source"]).refused.isna().sum())
        log.info("%s: %d prompts without a usable judgement (kept: groups come from labels)",
                 ckpt, n_unjudged)
        save_parquet(dist, dist_path)
        save_atomic(summary, out_dir / f"summary_{ckpt}{suffix}.csv")

    parts = [pd.read_csv(out_dir / f"summary_{c}{suffix}.csv") for c in args.checkpoints
             if (out_dir / f"summary_{c}{suffix}.csv").exists()]
    if parts:
        save_atomic(pd.concat(parts, ignore_index=True), out_dir / f"summary{suffix}.csv")


if __name__ == "__main__":
    main()
