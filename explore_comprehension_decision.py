"""
explore_comprehension_decision.py

Esplorazione PURA, senza fit di alcun probe. Da girare prima di
compute_comprehension_decision_dissociation.py per capire se ha senso
fittare qualcosa, e su quali (checkpoint, layer, position).

*** AGGIORNAMENTO IMPORTANTE (path/repo) ***
La repo delle attivazioni ha DUE strutture di path coesistenti per motivi
storici:
  - FLAT:   data/{checkpoint}/*.parquet          (5 dataset originali)
  - NESTED: data/{model_family}/{checkpoint}/*.parquet   (5 dataset aggiunti
            dopo, scritti da extract_and_push.py v3)
Nessuno dei due path da solo ha il quadro completo per checkpoint. Inoltre
OLMo2 e OLMo3 vivono su DUE REPO HF DIVERSE:
  - saracandu/olmo-activations   (OLMo2)
  - saracandu/olmo3-activations  (OLMo3)
Il path FLAT su saracandu/olmo-activations puo' contenere dati non
correlati a un dato modello (verificato: quello su OLMo3 conteneva un
mix estraneo, non le attivazioni OLMo3 reali) — usare --nested-only per
leggere ESCLUSIVAMENTE il path nested quando si lavora su OLMo3, o quando
si vuole isolare con certezza cosa contiene la v3.

Uso
---
    # OLMo2, quadro completo (merge flat+nested)
    python explore_comprehension_decision.py --model-family olmo2

    # OLMo3, SOLO il path nested (consigliato, il flat e' inaffidabile per OLMo3)
    python explore_comprehension_decision.py \
        --hf-repo saracandu/olmo3-activations --model-family olmo3 --nested-only

    python explore_comprehension_decision.py --checkpoints sft__none dpo__none --model-family olmo2
    python explore_comprehension_decision.py --pca-layers 16 24 --pca-positions last_prompt first_gen --model-family olmo2
    python explore_comprehension_decision.py --exclude-sources beavertails --model-family olmo2
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _has_valid_vector(v) -> bool:
    """True if v is an actual activation vector (ndarray/list), not NaN/None.

    Needed because flat+nested merges (pd.concat across checkpoints with
    different available columns, e.g. post_instr_0 missing on nested-only
    rows) leave the missing cells as NaN scalars instead of (4096,) vectors.
    """
    return isinstance(v, (np.ndarray, list))


def filter_valid_activations(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Drop rows where df[col] isn't a real vector (see _has_valid_vector)."""
    if col not in df.columns:
        return df.iloc[0:0]
    mask = df[col].apply(_has_valid_vector)
    return df[mask]

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

HF_REPO = "saracandu/olmo-activations"
PSEUDO_HARM_SOURCES = {"or_bench", "false_reject", "xstest"}
CHECKPOINT_ORDER = ["base__none", "sft__none", "dpo__none", "final__none"]
DEFAULT_LAYERS = [8, 16, 19, 24, 26, 31]
DEFAULT_POSITIONS = ["last_prompt", "post_instr_0", "first_gen"]


def assign_group(df: pd.DataFrame) -> pd.Series:
    g = pd.Series("harmless", index=df.index)
    g[df["label"] == 1] = "harmful"
    g[(df["label"] == 0) & (df["source"].isin(PSEUDO_HARM_SOURCES))] = "pseudo_harm"
    return g


def _data_paths(ckpt: str, model_family: str | None, nested_only: bool) -> list[str]:
    """
    Mirrors the same flat/nested logic already validated in check_sources.py.
    Returns the list of glob patterns to read and merge for this checkpoint.
    """
    flat = f"data/{ckpt}/*.parquet"
    nested = f"data/{model_family}/{ckpt}/*.parquet" if model_family else None
    if nested_only:
        if not nested:
            raise SystemExit("--nested-only requires --model-family to be set.")
        return [nested]
    if nested:
        return [flat, nested]
    return [flat]


def load_checkpoint_df(hf_repo, ckpt, layers, positions, token, max_rows=None,
                        max_rows_per_group=None, model_family=None, nested_only=False):
    from datasets import load_dataset

    base_cols = ["label", "source", "checkpoint", "predicted_refusal", "prompt"]
    wanted = [f"layer_{l}_{p}" for l in layers for p in positions]
    keep = set(base_cols) | set(wanted)
    paths = _data_paths(ckpt, model_family, nested_only)

    def load_one_full(path: str):
        # columns= prunes at the Parquet-read level (network + materialization),
        # same fix already applied in check_sources.py after the dpo__none OOM.
        try:
            ds = load_dataset(
                hf_repo, data_files={"train": path},
                split="train", token=token,
                columns=[c for c in base_cols] + wanted,  # may partially not exist; handled below
            )
        except TypeError:
            ds = load_dataset(hf_repo, data_files={"train": path}, split="train", token=token)
            present = [c for c in wanted if c in ds.column_names]
            available = [c for c in base_cols + present if c in ds.column_names]
            ds = ds.select_columns(available)
        except Exception:
            # columns= might reference activation columns that don't exist
            # for this path (e.g. different layer set) — retry without columns=
            # pruning and post-filter instead.
            ds = load_dataset(hf_repo, data_files={"train": path}, split="train", token=token)
            present = [c for c in wanted if c in ds.column_names]
            available = [c for c in base_cols + present if c in ds.column_names]
            ds = ds.select_columns(available)
        return ds

    def load_one_streaming(path: str):
        return load_dataset(
            hf_repo, data_files={"train": path},
            split="train", token=token, streaming=True,
        )

    # --- per-group quota streaming (unchanged behavior, now merged across paths) ---
    if max_rows_per_group is not None:
        quota = {"harmful": max_rows_per_group, "harmless": max_rows_per_group,
                  "pseudo_harm": max_rows_per_group}
        rows = []
        seen_cols = set()
        for path in paths:
            try:
                ds = load_one_streaming(path)
            except Exception as e:
                logger.warning("[%s] %s: skipping (%s)", ckpt, path, e)
                continue
            for ex in ds:
                if not seen_cols:
                    seen_cols = set(ex.keys())
                grp = (
                    "harmful" if ex.get("label") == 1
                    else "pseudo_harm" if ex.get("source") in PSEUDO_HARM_SOURCES
                    else "harmless"
                )
                if quota[grp] <= 0:
                    if all(v <= 0 for v in quota.values()):
                        break
                    continue
                rows.append({k: ex[k] for k in keep if k in ex})
                quota[grp] -= 1
            if all(v <= 0 for v in quota.values()):
                break

        present = [c for c in wanted if c in seen_cols]
        missing = [c for c in wanted if c not in seen_cols]
        if missing:
            logger.warning("[%s] missing %d/%d requested activation columns (e.g. %s)",
                            ckpt, len(missing), len(wanted), missing[:3])
        unfilled = {k: max_rows_per_group - v for k, v in quota.items()}
        for grp, got in unfilled.items():
            if got < max_rows_per_group:
                logger.warning(
                    "[%s] only found %d/%d rows for group '%s' (scanned across %d path(s))",
                    ckpt, got, max_rows_per_group, grp, len(paths),
                )
        if not rows:
            return None, []
        return pd.DataFrame(rows), present

    # --- flat row cutoff (legacy) ---
    if max_rows is not None:
        rows = []
        seen_cols = set()
        for path in paths:
            try:
                ds = load_one_streaming(path)
            except Exception as e:
                logger.warning("[%s] %s: skipping (%s)", ckpt, path, e)
                continue
            for i, ex in enumerate(ds):
                if not seen_cols:
                    seen_cols = set(ex.keys())
                if len(rows) >= max_rows:
                    break
                rows.append({k: ex[k] for k in keep if k in ex})
        present = [c for c in wanted if c in seen_cols]
        if not rows:
            return None, []
        return pd.DataFrame(rows), present

    # --- full load (default): merge all paths for this checkpoint ---
    dfs = []
    present_union = set()
    for path in paths:
        try:
            ds = load_one_full(path)
        except Exception as e:
            logger.warning("[%s] %s: skipping (%s)", ckpt, path, e)
            continue
        present_union |= (set(ds.column_names) & set(wanted))
        dfs.append(ds.to_pandas())

    if not dfs:
        return None, []

    df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    present = [c for c in wanted if c in present_union]
    missing = [c for c in wanted if c not in present_union]
    if missing:
        logger.warning("[%s] missing %d/%d requested activation columns across all paths (e.g. %s)",
                        ckpt, len(missing), len(wanted), missing[:3])
    return df, present


# ---------------------------------------------------------------------------
# 1-3: structural + distributional summaries
# ---------------------------------------------------------------------------

def print_structure(ckpt: str, df: pd.DataFrame, present_cols: list[str]) -> None:
    print(f"\n--- {ckpt} ---")
    print(f"rows: {len(df)}")
    print(f"activation columns present: {len(present_cols)}")
    df["group"] = assign_group(df)
    print("\ngroup x source counts:")
    print(df.groupby(["group", "source"]).size().to_string())


def print_behavior_rates(all_df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== predicted_refusal rate by group x checkpoint (no probe, raw behavior) ===")
    tab = all_df.pivot_table(
        index="group", columns="checkpoint", values="predicted_refusal", aggfunc="mean"
    ).reindex(index=["harmful", "pseudo_harm", "harmless"])
    print(tab.round(3).to_string())
    return tab


# ---------------------------------------------------------------------------
# 4: centroid distances, no classifier
# ---------------------------------------------------------------------------

def centroid_distance_table(all_df: pd.DataFrame, layers: list[int], positions: list[str]) -> pd.DataFrame:
    rows = []
    for ckpt, sub in all_df.groupby("checkpoint"):
        for position in positions:
            for layer in layers:
                col = f"layer_{layer}_{position}"
                if col not in sub.columns:
                    continue
                g_harm = sub[sub["group"] == "harmful"]
                g_safe = sub[sub["group"] == "harmless"]
                g_pseu = sub[sub["group"] == "pseudo_harm"]

                g_harm = filter_valid_activations(g_harm, col)
                g_safe = filter_valid_activations(g_safe, col)
                g_pseu = filter_valid_activations(g_pseu, col)

                if len(g_harm) == 0 or len(g_safe) == 0 or len(g_pseu) == 0:
                    logger.warning(
                        "centroid_distance_table: skipping %s/%s/layer%d — "
                        "empty group after dropping missing activations "
                        "(harm=%d safe=%d pseu=%d)",
                        ckpt, position, layer, len(g_harm), len(g_safe), len(g_pseu),
                    )
                    continue

                mu_harm = np.stack(g_harm[col].values).astype(np.float32).mean(0)
                mu_safe = np.stack(g_safe[col].values).astype(np.float32).mean(0)
                mu_pseu = np.stack(g_pseu[col].values).astype(np.float32).mean(0)

                def cos(a, b):
                    na, nb = np.linalg.norm(a), np.linalg.norm(b)
                    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else np.nan

                d_pseu_harm = float(np.linalg.norm(mu_pseu - mu_harm))
                d_pseu_safe = float(np.linalg.norm(mu_pseu - mu_safe))
                d_harm_safe = float(np.linalg.norm(mu_harm - mu_safe))

                rows.append({
                    "checkpoint": ckpt,
                    "position": position,
                    "layer": layer,
                    "dist_pseudo_to_harmful": d_pseu_harm,
                    "dist_pseudo_to_harmless": d_pseu_safe,
                    "dist_harmful_to_harmless": d_harm_safe,
                    "pseudo_closer_to_harmless_ratio": (
                        d_pseu_safe / d_pseu_harm if d_pseu_harm > 0 else np.nan
                    ),
                    "cos_pseudo_harmless": cos(mu_pseu - mu_safe, mu_harm - mu_safe),
                })
    return pd.DataFrame(rows)


def nearest_centroid_read(all_df: pd.DataFrame, layers: list[int], positions: list[str]) -> pd.DataFrame:
    rows = []
    for ckpt, sub in all_df.groupby("checkpoint"):
        for position in positions:
            for layer in layers:
                col = f"layer_{layer}_{position}"
                if col not in sub.columns:
                    continue

                g_harm = filter_valid_activations(sub[sub["group"] == "harmful"], col)
                g_safe = filter_valid_activations(sub[sub["group"] == "harmless"], col)
                if len(g_harm) == 0 or len(g_safe) == 0:
                    logger.warning(
                        "nearest_centroid_read: skipping %s/%s/layer%d — "
                        "empty centroid group after dropping missing activations",
                        ckpt, position, layer,
                    )
                    continue

                mu_harm = np.stack(g_harm[col].values).astype(np.float32).mean(0)
                mu_safe = np.stack(g_safe[col].values).astype(np.float32).mean(0)

                for group_name in ["harmful", "harmless", "pseudo_harm"]:
                    g = filter_valid_activations(sub[sub["group"] == group_name], col)
                    if len(g) == 0:
                        continue
                    X = np.stack(g[col].values).astype(np.float32)
                    d_harm = np.linalg.norm(X - mu_harm, axis=1)
                    d_safe = np.linalg.norm(X - mu_safe, axis=1)
                    model_read = (d_harm < d_safe).astype(int)

                    for i, (_, row) in enumerate(g.iterrows()):
                        rows.append({
                            "checkpoint": ckpt,
                            "position": position,
                            "layer": layer,
                            "group": group_name,
                            "source": row["source"],
                            "label": int(row["label"]),
                            "predicted_refusal": int(row["predicted_refusal"]),
                            "model_read": int(model_read[i]),
                        })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    df["gold_for_agreement"] = np.where(df["group"] == "pseudo_harm", 0, df["label"])
    df["agrees_with_gold"] = (df["model_read"] == df["gold_for_agreement"]).astype(int)
    return df


def divergence_summary(read_df: pd.DataFrame) -> pd.DataFrame:
    tab = (
        read_df.groupby(["checkpoint", "position", "layer", "group"])
        .agg(
            n=("agrees_with_gold", "size"),
            divergence_rate=("agrees_with_gold", lambda s: 1 - s.mean()),
        )
        .reset_index()
    )
    return tab


def behavior_vs_geometry_divergence(read_df: pd.DataFrame) -> pd.DataFrame:
    pseudo = read_df[read_df["group"] == "pseudo_harm"].copy()
    if len(pseudo) == 0:
        return pd.DataFrame()

    pseudo["behavior_diverges_from_geometry"] = (
        (pseudo["model_read"] == 0) & (pseudo["predicted_refusal"] == 1)
    ).astype(int)

    tab = (
        pseudo.groupby(["checkpoint", "position", "layer"])
        .agg(
            n=("behavior_diverges_from_geometry", "size"),
            rate=("behavior_diverges_from_geometry", "mean"),
            cluster_reads_harmful_rate=("model_read", "mean"),
            refusal_rate=("predicted_refusal", "mean"),
        )
        .reset_index()
        .sort_values(["position", "layer", "checkpoint"])
    )
    return tab


def plot_pca(all_df: pd.DataFrame, checkpoint: str, layer: int, position: str, out_dir: Path,
             sample_per_group: int = 150, seed: int = 42, read_df: pd.DataFrame | None = None) -> None:
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    col = f"layer_{layer}_{position}"
    sub = all_df[all_df["checkpoint"] == checkpoint]
    if col not in sub.columns:
        logger.warning("[%s] %s not present, skipping PCA plot", checkpoint, col)
        return

    parts = []
    for group in ["harmful", "harmless", "pseudo_harm"]:
        g = filter_valid_activations(sub[sub["group"] == group], col)
        if len(g) == 0:
            continue
        parts.append(g.sample(min(sample_per_group, len(g)), random_state=seed))
    if not parts:
        return
    plot_df = pd.concat(parts, ignore_index=True)

    X = np.stack(plot_df[col].values).astype(np.float32)
    xy = PCA(n_components=2, random_state=seed).fit_transform(X)
    plot_df["pc1"], plot_df["pc2"] = xy[:, 0], xy[:, 1]

    has_read = read_df is not None and len(read_df) > 0
    n_panels = 3 if has_read else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 2:
        axes = list(axes)

    group_colors = {"harmful": "tab:red", "harmless": "tab:green", "pseudo_harm": "tab:orange"}
    for group, color in group_colors.items():
        g = plot_df[plot_df["group"] == group]
        axes[0].scatter(g["pc1"], g["pc2"], s=10, alpha=0.6, label=group, color=color)
    axes[0].set_title(f"GOLD group\n{checkpoint} | {position} | layer {layer}")
    axes[0].legend()

    panel_idx = 1
    if has_read:
        mu_harm = np.stack(filter_valid_activations(sub[sub["group"] == "harmful"], col)[col].values).astype(np.float32).mean(0)
        mu_safe = np.stack(filter_valid_activations(sub[sub["group"] == "harmless"], col)[col].values).astype(np.float32).mean(0)
        d_harm = np.linalg.norm(X - mu_harm, axis=1)
        d_safe = np.linalg.norm(X - mu_safe, axis=1)
        plot_df["model_read"] = (d_harm < d_safe).astype(int)

        read_colors = {0: "tab:green", 1: "tab:red"}
        read_labels = {0: "geometric read: safe (closer to harmless centroid)",
                       1: "geometric read: harmful (closer to harmful centroid)"}
        for r, color in read_colors.items():
            g = plot_df[plot_df["model_read"] == r]
            axes[panel_idx].scatter(g["pc1"], g["pc2"], s=10, alpha=0.6, label=read_labels[r], color=color)
        axes[panel_idx].set_title(f"MODEL'S OWN READ (nearest-centroid)\n{checkpoint} | {position} | layer {layer}")
        axes[panel_idx].legend(fontsize=8)
        panel_idx += 1

    beh_colors = {0: "tab:blue", 1: "tab:purple"}
    beh_labels = {0: "answered (predicted_refusal=0)", 1: "refused (predicted_refusal=1)"}
    for beh, color in beh_colors.items():
        g = plot_df[plot_df["predicted_refusal"] == beh]
        axes[panel_idx].scatter(g["pc1"], g["pc2"], s=10, alpha=0.6, label=beh_labels[beh], color=color)
    axes[panel_idx].set_title(f"BEHAVIOR\n{checkpoint} | {position} | layer {layer}")
    axes[panel_idx].legend()

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pca_{checkpoint}_{position}_layer{layer}.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-repo", default=HF_REPO,
                        help="saracandu/olmo-activations (OLMo2) or "
                             "saracandu/olmo3-activations (OLMo3) — these are "
                             "TWO SEPARATE REPOS, not just different paths.")
    parser.add_argument("--model-family", default=None, choices=[None, "olmo2", "olmo3"],
                        help="If set, ALSO reads the nested data/{model_family}/{ckpt}/ "
                             "path in addition to the flat data/{ckpt}/ path (merged), "
                             "unless --nested-only is given.")
    parser.add_argument("--nested-only", action="store_true",
                        help="With --model-family set, read ONLY the nested path. "
                             "Recommended for OLMo3 — the flat path on that repo has "
                             "been found to contain unrelated/stale data.")
    parser.add_argument("--checkpoints", nargs="+", default=CHECKPOINT_ORDER)
    parser.add_argument("--layers", nargs="+", type=int, default=DEFAULT_LAYERS)
    parser.add_argument("--positions", nargs="+", default=DEFAULT_POSITIONS)
    parser.add_argument("--exclude-sources", nargs="*", default=None)
    parser.add_argument("--pca-layers", nargs="+", type=int, default=[16, 24])
    parser.add_argument("--pca-positions", nargs="+", default=["last_prompt", "first_gen"])
    parser.add_argument("--skip-pca", action="store_true")
    parser.add_argument("--max-rows-per-checkpoint", type=int, default=None,
                        help="[LEGACY] flat streaming cutoff — prefer --max-rows-per-group.")
    parser.add_argument("--max-rows-per-group", type=int, default=None,
                        help="Stream and stop once N rows are found for EACH of "
                             "harmful/harmless/pseudo_harm. If unset, loads the FULL "
                             "checkpoint (now safe: ~5800 rows/checkpoint after the "
                             "columns= pruning fix, no longer OOM-prone like before).")
    parser.add_argument("--out-dir", default="results/olmo2/geometry/explore")
    parser.add_argument("--raw-results-csv", default="results/olmo2/raw_results.csv",
                        help="CSV con judge_ga/judge_pd, per usare il giudizio del "
                             "giudice al posto del keyword detector predicted_refusal.")
    parser.add_argument("--refusal-source", default="judge", choices=["judge", "keyword"],
                        help="'judge' (default): predicted_refusal viene sovrascritto "
                             "con judge_refusal via merge. 'keyword': comportamento legacy.")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or open(
        os.path.expanduser("~/.hf_token")
    ).read().strip()

    all_frames = []
    for ckpt in args.checkpoints:
        df, present_cols = load_checkpoint_df(
            args.hf_repo, ckpt, args.layers, args.positions, token,
            max_rows=args.max_rows_per_checkpoint,
            max_rows_per_group=args.max_rows_per_group,
            model_family=args.model_family,
            nested_only=args.nested_only,
        )
        if df is None or len(df) == 0:
            continue
        if args.exclude_sources:
            df = df[~df["source"].isin(args.exclude_sources)].reset_index(drop=True)
        print_structure(ckpt, df, present_cols)
        all_frames.append(df)

    if not all_frames:
        logger.error("No data loaded.")
        return

    all_df = pd.concat(all_frames, ignore_index=True)

    if args.refusal_source == "judge":
        from analysis.judge_utils import attach_judge_refusal
        n_before = len(all_df)
        all_df = attach_judge_refusal(all_df, args.raw_results_csv, drop_missing=True)
        all_df["predicted_refusal"] = all_df["judge_refusal"]
        logger.info(
            "Refusal source: JUDGE. %d -> %d righe dopo merge con %s.",
            n_before, len(all_df), args.raw_results_csv,
        )
    else:
        logger.info("Refusal source: KEYWORD (predicted_refusal, comportamento legacy).")

    all_df["group"] = assign_group(all_df)

    behavior_tab = print_behavior_rates(all_df)

    print("\n=== centroid distances (no classifier) ===")
    dist_tab = centroid_distance_table(all_df, args.layers, args.positions)
    print(dist_tab.round(3).to_string(index=False))

    logger.info("Computing nearest-centroid reads (unsupervised, no CV)...")
    read_df = nearest_centroid_read(all_df, args.layers, args.positions)

    if read_df is None or len(read_df) == 0 or "checkpoint" not in read_df.columns:
        logger.warning(
            "No nearest-centroid reads computed — check group x source counts above."
        )
        return

    print("\n=== divergence_rate: model_read (nearest-centroid) vs GOLD label ===")
    div_gold = divergence_summary(read_df)
    print(
        div_gold[div_gold["group"] == "pseudo_harm"]
        .pivot_table(index=["position", "layer"], columns="checkpoint", values="divergence_rate")
        .round(3).to_string()
    )
    print("\n--- for reference, same on genuinely harmful/harmless ---")
    print(
        div_gold[div_gold["group"] != "pseudo_harm"]
        .pivot_table(index=["group", "position", "layer"], columns="checkpoint", values="divergence_rate")
        .round(3).to_string()
    )

    print("\n=== WHERE does BEHAVIOR diverge from the model's OWN geometric read? (pseudo_harm only) ===")
    div_beh = behavior_vs_geometry_divergence(read_df)
    print(
        div_beh.pivot_table(index=["position", "layer"], columns="checkpoint", values="rate")
        .round(3).to_string()
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    behavior_tab.to_csv(out_dir / "behavior_rates.csv")
    dist_tab.to_csv(out_dir / "centroid_distances.csv", index=False)
    read_df.to_csv(out_dir / "nearest_centroid_reads_per_example.csv", index=False)
    div_gold.to_csv(out_dir / "divergence_vs_gold.csv", index=False)
    div_beh.to_csv(out_dir / "behavior_vs_geometry_divergence.csv", index=False)
    logger.info("Saved all summary CSVs to %s", out_dir)

    if not args.skip_pca:
        for ckpt in args.checkpoints:
            for position in args.pca_positions:
                for layer in args.pca_layers:
                    plot_pca(all_df, ckpt, layer, position, out_dir / "pca", read_df=read_df)


if __name__ == "__main__":
    main()
