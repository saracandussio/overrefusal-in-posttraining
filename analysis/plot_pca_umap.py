"""
plot_pca_umap.py
Produce una griglia di scatter plot PCA e UMAP per le attivazioni di OLMo2.
Righe = layer, colonne = checkpoint (base / sft / final per default).
Due figure separate: una per PCA, una per UMAP.

Colori = 3 categorie (harmful / pseudo_harm / harmless).
Con --behavioural: marker pieno = compliant, marker vuoto = rifiutato.

Usage:
    python analysis/plot_pca_umap.py \
        --exclude-sources beavertails \
        --position first_gen last_prompt \
        --output-dir figures/pca_umap/

    # con distinzione behavioural:
    python analysis/plot_pca_umap.py \
        --behavioural \
        --exclude-sources beavertails \
        --position first_gen last_prompt \
        --output-dir figures/pca_umap/
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA

PSEUDO_SOURCES = {"or_bench", "false_reject"}

GROUP_MARKER = {
    "harmful":     "x",
    "pseudo_harm": "^",
    "harmless":    "o",
}

# Colori per (group, predicted_refusal) — stesso schema di plot_2d_refusal_space.py
POINT_COLOR = {
    ("harmful",     True):  "#e63946",   # rosso vivo   — harmful rifiutato (corretto)
    ("harmful",     False): "#ffb3b3",   # rosa chiaro  — harmful accettato (miss)
    ("pseudo_harm", True):  "#f4a261",   # arancione    — pseudo-harmful rifiutato (over-refusal)
    ("pseudo_harm", False): "#2a9d8f",   # verde        — pseudo-harmful accettato (corretto)
    ("harmless",    True):  "#9b2226",   # bordeaux     — harmless rifiutato (over-refusal grave)
    ("harmless",    False): "#457b9d",   # blu          — harmless accettato (corretto)
}

# Colore gruppo senza distinzione behavioural
GROUP_COLOR = {g: POINT_COLOR[(g, False)] for g in ["harmful", "pseudo_harm", "harmless"]}

GROUP_LABEL_BEH = {
    ("harmful",     True):  "harmful — rifiutato v",
    ("harmful",     False): "harmful — accettato (miss)",
    ("pseudo_harm", True):  "pseudo-harmful — rifiutato (over-r)",
    ("pseudo_harm", False): "pseudo-harmful — accettato v",
    ("harmless",    True):  "harmless — rifiutato (over-r grave)",
    ("harmless",    False): "harmless — accettato v",
}
GROUP_LABEL = {
    "harmful":     "harmful",
    "pseudo_harm": "pseudo-harmful",
    "harmless":    "harmless",
}

CENTROID_STYLE = dict(s=180, edgecolors="black", linewidths=0.6, zorder=5)


def assign_group(df):
    groups = pd.Series("harmless", index=df.index)
    groups[df["label"] == 1] = "harmful"
    groups[(df["label"] == 0) & (df["source"].isin(PSEUDO_SOURCES))] = "pseudo_harm"
    return groups


def load_hf_token(cli_token):
    if cli_token:
        return cli_token
    token_path = Path("~/.hf_token").expanduser()
    if token_path.exists():
        return token_path.read_text().strip()
    return None


def load_data(hf_dataset, checkpoints, layers, position, token,
              exclude_sources=None, behavioural=False):
    from datasets import load_dataset

    base_cols = ["label", "source", "checkpoint"]
    if behavioural:
        base_cols.append("predicted_refusal")
    act_cols = [f"layer_{l}_{position}" for l in layers]
    needed   = base_cols + act_cols

    dfs = []
    for ckpt in checkpoints:
        try:
            ds = load_dataset(
                hf_dataset,
                data_files={"train": f"data/{ckpt}/*.parquet"},
                split="train",
                token=token,
            )
            available = [c for c in needed if c in ds.column_names]
            if behavioural and "predicted_refusal" not in available:
                print(f"  [warn] {ckpt}: colonna 'predicted_refusal' non trovata, "
                      f"behavioural disabilitato per questo checkpoint")
            df = ds.select_columns(available).to_pandas()
            dfs.append(df)
            print(f"  {ckpt}: {len(df)} righe")
        except Exception as e:
            print(f"  [error] {ckpt}: {e}")

    if not dfs:
        return None

    df_full = pd.concat(dfs, ignore_index=True)

    if exclude_sources:
        before = len(df_full)
        df_full = df_full[~df_full["source"].isin(exclude_sources)].reset_index(drop=True)
        print(f"[.] Escluse source {exclude_sources}: {before} -> {len(df_full)} righe")

    df_full["group"] = assign_group(df_full)

    # normalizza predicted_refusal a bool se presente
    if behavioural and "predicted_refusal" in df_full.columns:
        df_full["predicted_refusal"] = df_full["predicted_refusal"].astype(bool)
    elif behavioural:
        df_full["predicted_refusal"] = False  # fallback silenzioso

    return df_full


# ---------------------------------------------------------------------------
# Scatter helpers
# ---------------------------------------------------------------------------

def _scatter_group_plain(ax, coords, groups, group):
    """Scatter senza distinzione behavioural."""
    mask = groups == group
    if mask.sum() == 0:
        return
    ax.scatter(coords[mask, 0], coords[mask, 1],
               color=GROUP_COLOR[group],
               marker=GROUP_MARKER[group],
               alpha=0.4, s=15, linewidths=0.5)
    cx, cy = coords[mask, 0].mean(), coords[mask, 1].mean()
    ax.scatter(cx, cy, color=GROUP_COLOR[group], marker="*", **CENTROID_STYLE)


def _scatter_group_behavioural(ax, coords, groups, refused, group):
    """
    Scatter con distinzione behavioural usando POINT_COLOR:
      colore = (group, refused) come in plot_2d_refusal_space.py
    """
    mask = groups == group
    if mask.sum() == 0:
        return

    for refused_val in [False, True]:
        m = mask & (refused == refused_val)
        if m.sum() == 0:
            continue
        color = POINT_COLOR[(group, refused_val)]
        ax.scatter(coords[m, 0], coords[m, 1],
                   color=color,
                   marker=GROUP_MARKER[group],
                   alpha=0.5, s=15, linewidths=0.5)
        # centroide per ciascuno dei 6 sottogruppi
        cx, cy = coords[m, 0].mean(), coords[m, 1].mean()
        ax.scatter(cx, cy, color=color, marker="*", **CENTROID_STYLE)


# ---------------------------------------------------------------------------
# Legenda
# ---------------------------------------------------------------------------

def make_legend_handles(behavioural=False):
    handles = []
    if not behavioural:
        for g in ["harmful", "pseudo_harm", "harmless"]:
            handles.append(plt.scatter([], [], color=GROUP_COLOR[g],
                                       marker=GROUP_MARKER[g], alpha=0.7,
                                       s=30, label=GROUP_LABEL[g]))
        handles.append(plt.scatter([], [], color="gray", marker="*", s=80,
                                   edgecolors="black", linewidths=1,
                                   label="centroide"))
    else:
        for g in ["harmful", "pseudo_harm", "harmless"]:
            for refused_val in [False, True]:
                color = POINT_COLOR[(g, refused_val)]
                label = GROUP_LABEL_BEH[(g, refused_val)]
                handles.append(plt.scatter([], [], color=color,
                                           marker=GROUP_MARKER[g],
                                           alpha=0.8, s=30, label=label))
        handles.append(plt.scatter([], [], color="gray", marker="*", s=80,
                                   edgecolors="black", linewidths=1,
                                   label="centroide"))
    return handles


# ---------------------------------------------------------------------------
# Plot grid
# ---------------------------------------------------------------------------

def plot_grid(proj_data, checkpoints, layers, method_name, position,
              output_path, behavioural=False, sample=None, rng=None):
    """
    proj_data: dict[(ckpt, layer)] -> dict con:
        'coords'  : (N, 2)
        'groups'  : (N,)  stringa categoria
        'refused' : (N,)  bool  [solo se behavioural]
    """
    n_rows = len(layers)
    n_cols = len(checkpoints)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.8 * n_cols, 3.2 * n_rows),
                             squeeze=False)

    for r, layer in enumerate(layers):
        for c, ckpt in enumerate(checkpoints):
            ax = axes[r][c]
            key = (ckpt, layer)
            if key not in proj_data:
                ax.set_visible(False)
                continue

            coords = proj_data[key]["coords"]
            groups = proj_data[key]["groups"]
            refused = proj_data[key].get("refused",
                                         np.zeros(len(groups), dtype=bool))

            if sample and len(coords) > sample:
                idx = rng.choice(len(coords), sample, replace=False)
                coords  = coords[idx]
                groups  = groups[idx]
                refused = refused[idx]

            for group in ["harmless", "pseudo_harm", "harmful"]:
                if behavioural:
                    _scatter_group_behavioural(ax, coords, groups, refused, group)
                else:
                    _scatter_group_plain(ax, coords, groups, group)

            if r == 0:
                ax.set_title(ckpt.replace("__", "\n"), fontsize=8, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"layer {layer}\n{method_name} 2", fontsize=7)
            if r == n_rows - 1:
                ax.set_xlabel(f"{method_name} 1", fontsize=7)
            ax.tick_params(labelsize=6)
            n = len(groups)
            ax.text(0.03, 0.97, f"n={n}", transform=ax.transAxes,
                    fontsize=6, ha="left", va="top", color="gray")

    ncol_legend = 4 if not behavioural else 3
    fig.legend(handles=make_legend_handles(behavioural),
               loc="lower center", ncol=ncol_legend,
               fontsize=8, bbox_to_anchor=(0.5, -0.03))

    beh_tag = " [behavioural]" if behavioural else ""
    fig.suptitle(
        f"OLMo2 — {method_name}  |  {position}{beh_tag}",
        fontsize=10, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  [ok] {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dataset",      default="saracandu/olmo-activations")
    parser.add_argument("--hf-token",        default=None)
    parser.add_argument("--checkpoints",     nargs="+",
                        default=["base__none", "sft__none", "final__none"])
    parser.add_argument("--layers",          nargs="+", type=int,
                        default=[8, 16, 19, 24, 26, 31])
    parser.add_argument("--position",        nargs="+",
                        default=["first_gen", "last_prompt"],
                        choices=["first_gen", "last_prompt"])
    parser.add_argument("--exclude-sources", nargs="*", default=None)
    parser.add_argument("--behavioural",     action="store_true",
                        help="Distingui compliant (pieno) da rifiutato (vuoto) "
                             "usando la colonna predicted_refusal")
    parser.add_argument("--sample",          type=int, default=None,
                        help="Max punti per subplot (dopo la proiezione)")
    parser.add_argument("--no-umap",         action="store_true",
                        help="Salta UMAP (più veloce, solo PCA)")
    parser.add_argument("--umap-neighbors",  type=int, default=15)
    parser.add_argument("--umap-min-dist",   type=float, default=0.1)
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--output-dir",      default="figures/pca_umap/")
    args = parser.parse_args()

    rng   = np.random.default_rng(args.seed)
    token = load_hf_token(args.hf_token)
    excl  = set(args.exclude_sources) if args.exclude_sources else None

    for position in args.position:
        print(f"\n{'='*60}")
        print(f"Position: {position}  |  behavioural={args.behavioural}")
        print(f"{'='*60}")
        print(f"[.] Carico dati ...")
        df_full = load_data(
            args.hf_dataset, args.checkpoints, args.layers,
            position, token, excl, behavioural=args.behavioural
        )
        if df_full is None:
            continue

        # suffix per distinguere i file output
        beh_suffix = "_behavioural" if args.behavioural else ""

        # ---------------------------------------------------------------
        # PCA
        # ---------------------------------------------------------------
        print(f"[.] PCA ...")
        pca_data = {}
        for ckpt in args.checkpoints:
            df_ckpt = df_full[df_full["checkpoint"] == ckpt]
            for layer in args.layers:
                col = f"layer_{layer}_{position}"
                if col not in df_ckpt.columns:
                    continue
                X = np.stack(df_ckpt[col].apply(
                    lambda v: np.asarray(v, dtype=np.float32)).values)
                groups  = df_ckpt["group"].values
                refused = df_ckpt["predicted_refusal"].values.astype(bool) \
                          if "predicted_refusal" in df_ckpt.columns \
                          else np.zeros(len(groups), dtype=bool)

                pca    = PCA(n_components=2, random_state=args.seed)
                coords = pca.fit_transform(X)
                pca_data[(ckpt, layer)] = {
                    "coords":  coords,
                    "groups":  groups,
                    "refused": refused,
                }
                var = pca.explained_variance_ratio_
                print(f"  {ckpt} layer {layer}: var={var[0]:.2f}+{var[1]:.2f}")

        plot_grid(
            pca_data, args.checkpoints, args.layers,
            "PCA", position,
            Path(args.output_dir) / f"pca_{position}{beh_suffix}.png",
            behavioural=args.behavioural,
            sample=args.sample, rng=rng,
        )

        # ---------------------------------------------------------------
        # UMAP
        # ---------------------------------------------------------------
        if not args.no_umap:
            try:
                import umap
            except ImportError:
                print("[!] umap-learn non installato. Salta UMAP.")
                print("    Installa con: pip install umap-learn")
                continue

            print(f"[.] UMAP (n_neighbors={args.umap_neighbors}, "
                  f"min_dist={args.umap_min_dist}) ...")
            umap_data = {}
            for ckpt in args.checkpoints:
                df_ckpt = df_full[df_full["checkpoint"] == ckpt]
                for layer in args.layers:
                    col = f"layer_{layer}_{position}"
                    if col not in df_ckpt.columns:
                        continue
                    X = np.stack(df_ckpt[col].apply(
                        lambda v: np.asarray(v, dtype=np.float32)).values)
                    groups  = df_ckpt["group"].values
                    refused = df_ckpt["predicted_refusal"].values.astype(bool) \
                              if "predicted_refusal" in df_ckpt.columns \
                              else np.zeros(len(groups), dtype=bool)

                    reducer = umap.UMAP(
                        n_components=2,
                        n_neighbors=args.umap_neighbors,
                        min_dist=args.umap_min_dist,
                        random_state=args.seed,
                        verbose=False,
                    )
                    coords = reducer.fit_transform(X)
                    umap_data[(ckpt, layer)] = {
                        "coords":  coords,
                        "groups":  groups,
                        "refused": refused,
                    }
                    print(f"  {ckpt} layer {layer}: ok")

            plot_grid(
                umap_data, args.checkpoints, args.layers,
                "UMAP", position,
                Path(args.output_dir) / f"umap_{position}{beh_suffix}.png",
                behavioural=args.behavioural,
                sample=args.sample, rng=rng,
            )

    print("\n[ok] Fatto.")


if __name__ == "__main__":
    main()