"""
plot_pca_umap.py
Produce una griglia di scatter plot PCA e UMAP per le attivazioni di OLMo2.
Righe = layer, colonne = checkpoint (base / sft / final per default).
Due figure separate: una per PCA, una per UMAP.

Colori = 3 categorie (harmful / pseudo_harm / harmless).
Con --behavioural: il colore distingue decisione corretta / errore.

--split-groups:
    Invece di una figura con tutti i punti sovrapposti, produce TRE figure
    per posizione (harmless / pseudo_harm / harmful). La proiezione e' la
    STESSA (PCA/UMAP fittata una volta sola su tutti i punti di quel
    checkpoint+layer) e i limiti degli assi sono IDENTICI fra le tre figure,
    quindi i pannelli sono direttamente sovrapponibili.
    In ogni pannello:
      - nuvola grigia di sfondo = tutti i punti (riferimento visivo)
      - punti colorati = solo il gruppo della figura
      - centroidi = TUTTI i sottogruppi, pieni se appartengono alla figura,
        tenui altrimenti (per confrontare le posizioni relative)

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

    # tre figure per gruppo, scale compatibili:
    python analysis/plot_pca_umap.py \
        --behavioural --split-groups --no-umap \
        --exclude-sources beavertails \
        --position first_gen last_prompt \
        --raw-results-csv results/olmo2/raw_results.csv \
        --output-dir figures/pca_umap/
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA

# Fonti i cui prompt label=0 sono pseudo-harmful (benigni che SEMBRANO dannosi).
PSEUDO_SOURCES = {"or_bench", "false_reject"}

# XSTest va trattato a parte: nel dataset TUTTE le 450 righe hanno label=0,
# ma le 18 categorie si dividono in due meta' di significato opposto:
#   - 10 categorie "safe"  (homonyms, definitions, privacy_public, ...)
#     = i 250 prompt benigni che sembrano dannosi        -> pseudo_harm
#   - 8  categorie contrast_* = i 200 prompt REALMENTE unsafe costruiti come
#     contrasto ("kill a Python process" vs la versione con bersaglio umano)
#                                                        -> harmful
# Senza questa distinzione i 200 contrast_* finiscono fra harmless/pseudo e
# un loro rifiuto — che e' la decisione CORRETTA — viene contato come
# over-refusal, contaminando sia le etichette sia il polo harmless dell'asse.
XSTEST_UNSAFE_PREFIX = "contrast_"

GROUPS = ["harmless", "pseudo_harm", "harmful"]

GROUP_MARKER = {
    "harmful":     "x",
    "pseudo_harm": "^",
    "harmless":    "o",
}

# Palette ridisegnata attorno all'asse CORRETTO vs ERRORE, che e' il pattern
# che conta davvero (dissociazione comportamento/verita') — non attorno al
# gruppo da solo. Prima, due rossi diversi (harmful-rifiutato-corretto vs
# harmless-rifiutato-errore) erano percettivamente troppo simili, nascondendo
# esattamente la distinzione piu' importante da vedere in un plot denso.
#
# Famiglia "corretto" (freddi, blu/verde) — decisione giusta:
#   harmful rifiutato, pseudo-harmful accettato, harmless accettato
# Famiglia "errore" (caldi, arancio/rosso) — decisione sbagliata:
#   harmful accettato (miss), pseudo-harmful rifiutato (over-r),
#   harmless rifiutato (over-r grave)
#
# pseudo_harm ACCETTATE = verde scuro pieno (#1b5e20): sono la categoria
# chiave per l'overrefusal e devono staccarsi nettamente sia dal verde-teal
# delle harmless accettate sia dall'arancio delle pseudo rifiutate.
CORRECT_COLOR = {
    "harmful":     "#1a759f",  # blu
    "pseudo_harm": "#1b5e20",  # verde scuro — pseudo-harmful ACCETTATE
    "harmless":    "#52b69a",  # verde-teal
}
ERROR_COLOR = {
    "harmful":     "#f4a261",  # arancio chiaro — miss (meno grave: harmful come pseudo)
    "pseudo_harm": "#e76f51",  # arancio-rosso — over-refusal su pseudo (grave)
    "harmless":    "#9d0208",  # rosso scuro — over-refusal su harmless (piu' grave)
}


def _is_correct(group: str, refused: bool) -> bool:
    """Decisione corretta: harmful va rifiutato, pseudo/harmless vanno accettati."""
    should_refuse = (group == "harmful")
    return refused == should_refuse


def _point_color(group: str, refused: bool) -> str:
    return CORRECT_COLOR[group] if _is_correct(group, refused) else ERROR_COLOR[group]


POINT_COLOR = {
    (g, r): _point_color(g, r)
    for g in GROUPS
    for r in [True, False]
}

# Colore gruppo senza distinzione behavioural (usa la versione "corretta" come default)
GROUP_COLOR = {g: CORRECT_COLOR[g] for g in GROUPS}

GROUP_LABEL_BEH = {
    ("harmful",     True):  "harmful — rifiutato ✓",
    ("harmful",     False): "harmful — accettato (miss)",
    ("pseudo_harm", True):  "pseudo-harmful — rifiutato (over-r)",
    ("pseudo_harm", False): "pseudo-harmful — accettato ✓",
    ("harmless",    True):  "harmless — rifiutato (over-r grave)",
    ("harmless",    False): "harmless — accettato ✓",
}
GROUP_LABEL = {
    "harmful":     "harmful",
    "pseudo_harm": "pseudo-harmful",
    "harmless":    "harmless",
}

# Errori disegnati sopra e leggermente piu' grandi (vedi _scatter_group_behavioural):
# sono tipicamente la classe minoritaria e rischiano di sparire sotto la
# massa di punti "corretti" se disegnati con lo stesso zorder/size.
ERROR_ZORDER_BOOST = 2
ERROR_SIZE_BOOST = 1.3  # moltiplicatore di s per i punti di errore

CENTROID_STYLE = dict(s=180, edgecolors="black", linewidths=0.6, zorder=5)
# centroide di un sottogruppo NON appartenente al pannello corrente
CENTROID_STYLE_FAINT = dict(s=110, edgecolors="black", linewidths=0.4,
                            zorder=4, alpha=0.30)



def assign_group(df):
    groups = pd.Series("harmless", index=df.index)
    groups[df["label"] == 1] = "harmful"
    groups[(df["label"] == 0) & (df["source"].isin(PSEUDO_SOURCES))] = "pseudo_harm"

    # XSTest: split per categoria, non per label (vedi commento sopra).
    is_x = df["source"].eq("xstest")
    if is_x.any():
        if "category" not in df.columns:
            raise SystemExit(
                "xstest presente ma manca la colonna 'category': impossibile "
                "separare i 200 prompt contrast_* (unsafe) dai 250 safe. "
                "Carica 'category', oppure escludi xstest con --exclude-sources."
            )
        unsafe = df["category"].astype(str).str.startswith(XSTEST_UNSAFE_PREFIX)
        groups[is_x & unsafe] = "harmful"
        groups[is_x & ~unsafe] = "pseudo_harm"
    return groups


def load_hf_token(cli_token):
    if cli_token:
        return cli_token
    token_path = Path("~/.hf_token").expanduser()
    if token_path.exists():
        return token_path.read_text().strip()
    return None


def load_data(hf_dataset, checkpoints, layers, position, token,
              exclude_sources=None, behavioural=False,
              model_family=None, nested_only=False,
              refusal_source="judge", raw_results_csv=None):
    from datasets import load_dataset

    base_cols = ["label", "source", "checkpoint", "prompt", "category"]
    if behavioural:
        base_cols.append("predicted_refusal")
    act_cols = [f"layer_{l}_{position}" for l in layers]
    needed   = base_cols + act_cols

    def _data_paths(ckpt):
        flat = f"data/{ckpt}/*.parquet"
        nested = f"data/{model_family}/{ckpt}/*.parquet" if model_family else None
        if nested_only:
            if not nested:
                raise SystemExit("--nested-only requires --model-family to be set.")
            return [nested]
        return [flat, nested] if nested else [flat]

    dfs = []
    for ckpt in checkpoints:
        try:
            ckpt_dfs = []
            for path in _data_paths(ckpt):
                ds = load_dataset(
                    hf_dataset, data_files={"train": path}, split="train", token=token,
                )
                available = [c for c in needed if c in ds.column_names]
                if behavioural and "predicted_refusal" not in available:
                    print(f"  [warn] {ckpt} {path}: colonna 'predicted_refusal' non trovata, "
                          f"behavioural disabilitato per questo checkpoint")
                ckpt_dfs.append(ds.select_columns(available).to_pandas())
            df = pd.concat(ckpt_dfs, ignore_index=True) if len(ckpt_dfs) > 1 else ckpt_dfs[0]
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
    print("[.] Composizione gruppi (tutti i checkpoint):")
    for g, n in df_full["group"].value_counts().items():
        srcs = df_full.loc[df_full["group"] == g, "source"].value_counts().to_dict()
        print(f"      {g:<12} n={n:<6} {srcs}")
    for g in GROUPS:
        if (df_full["group"] == g).sum() == 0:
            print(f"    [warn] gruppo '{g}' VUOTO: i suoi pannelli avranno "
                  f"solo i centroidi degli altri gruppi.")

    if behavioural:
        if refusal_source == "judge":
            if not raw_results_csv:
                raise SystemExit("--refusal-source judge richiede --raw-results-csv")
            import sys as _sys, os as _os
            _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
            from analysis.judge_utils import attach_judge_refusal
            n_before = len(df_full)
            df_full = attach_judge_refusal(df_full, raw_results_csv, drop_missing=True)
            df_full["predicted_refusal"] = df_full["judge_refusal"]
            print(f"[.] Refusal source: JUDGE. {n_before} -> {len(df_full)} righe "
                  f"dopo merge con {raw_results_csv}.")
        elif "predicted_refusal" not in df_full.columns:
            df_full["predicted_refusal"] = False  # fallback silenzioso

        df_full["predicted_refusal"] = df_full["predicted_refusal"].astype(bool)

    return df_full


# ---------------------------------------------------------------------------
# Scatter helpers
# ---------------------------------------------------------------------------

def _xy(coords, mask=None):
    """Colonne di coords come tupla -> funziona identico in 2D e 3D."""
    c = coords if mask is None else coords[mask]
    return tuple(c[:, k] for k in range(c.shape[1]))


def _cent(coords, mask):
    """Centroide come tupla di scalari, 2D o 3D."""
    return tuple(coords[mask][:, k].mean() for k in range(coords.shape[1]))


def _scatter_group_plain(ax, coords, groups, group, draw_centroid=True):
    """Scatter senza distinzione behavioural."""
    mask = groups == group
    if mask.sum() == 0:
        return
    ax.scatter(*_xy(coords, mask),
               color=GROUP_COLOR[group],
               marker=GROUP_MARKER[group],
               alpha=0.4, s=15, linewidths=0.5, zorder=3)
    if draw_centroid:
        ax.scatter(*_cent(coords, mask), color=GROUP_COLOR[group],
                   marker="*", **CENTROID_STYLE)


def _scatter_group_behavioural(ax, coords, groups, refused, group,
                               draw_centroid=True):
    """
    Scatter con distinzione behavioural usando POINT_COLOR (asse corretto/errore).
    Gli errori sono disegnati con zorder e size maggiori: sono tipicamente la
    classe minoritaria e sparirebbero sotto la massa di punti corretti.

    Ogni sottogruppo (gruppo x refused) ha il proprio centroide: in particolare
    pseudo-harmful ACCETTATE e RIFIUTATE hanno due centroidi distinti.
    """
    mask = groups == group
    if mask.sum() == 0:
        return

    for refused_val in [False, True]:
        m = mask & (refused == refused_val)
        if m.sum() == 0:
            continue
        color = POINT_COLOR[(group, refused_val)]
        is_correct = _is_correct(group, refused_val)
        ax.scatter(*_xy(coords, m),
                   color=color,
                   marker=GROUP_MARKER[group],
                   alpha=0.45 if is_correct else 0.75,
                   s=15 if is_correct else int(15 * ERROR_SIZE_BOOST),
                   linewidths=0.5,
                   zorder=3 if is_correct else 3 + ERROR_ZORDER_BOOST)
        if draw_centroid:
            ax.scatter(*_cent(coords, m), color=color,
                       marker="*", **CENTROID_STYLE)


def _draw_all_centroids(ax, coords, groups, refused, own_group, behavioural):
    """
    Disegna i centroidi di TUTTI i sottogruppi presenti.
    Pieni se appartengono a `own_group`, tenui altrimenti: serve a leggere la
    posizione relativa (es. quanto le pseudo rifiutate si avvicinino al
    centroide harmful) restando dentro un pannello con un solo gruppo.
    """
    for g in GROUPS:
        gm = groups == g
        if gm.sum() == 0:
            continue
        subs = [(None, gm)] if not behavioural else [
            (rv, gm & (refused == rv)) for rv in [False, True]
        ]
        for rv, m in subs:
            if m.sum() == 0:
                continue
            color = GROUP_COLOR[g] if rv is None else POINT_COLOR[(g, rv)]
            style = CENTROID_STYLE if g == own_group else CENTROID_STYLE_FAINT
            ax.scatter(*_cent(coords, m), color=color, marker="*", **style)


def compute_limits(proj_data, pad_frac=0.05):
    """Limiti assi per (ckpt, layer), calcolati su TUTTI i punti di quella cella."""
    limits = {}
    for key, d in proj_data.items():
        c = d["coords"]
        span = c.max(0) - c.min(0)
        pad = pad_frac * np.where(span > 0, span, 1.0)
        limits[key] = tuple((c[:, k].min() - pad[k], c[:, k].max() + pad[k])
                            for k in range(c.shape[1]))
    return limits


# ---------------------------------------------------------------------------
# Legenda
# ---------------------------------------------------------------------------

def make_legend_handles(behavioural=False, only_group=None):
    handles = []
    groups = GROUPS if only_group is None else [only_group]
    if not behavioural:
        for g in groups:
            handles.append(plt.scatter([], [], color=GROUP_COLOR[g],
                                       marker=GROUP_MARKER[g], alpha=0.7,
                                       s=30, label=GROUP_LABEL[g]))
    else:
        for g in groups:
            for refused_val in [False, True]:
                handles.append(plt.scatter([], [], color=POINT_COLOR[(g, refused_val)],
                                           marker=GROUP_MARKER[g],
                                           alpha=0.8, s=30,
                                           label=GROUP_LABEL_BEH[(g, refused_val)]))
    handles.append(plt.scatter([], [], color="gray", marker="*", s=80,
                               edgecolors="black", linewidths=1,
                               label="centroide (gruppo in figura)"))
    if only_group is not None:
        handles.append(plt.scatter([], [], color="gray", marker="*", s=55,
                                   edgecolors="black", linewidths=0.4, alpha=0.30,
                                   label="centroide (altri gruppi)"))
    return handles


# ---------------------------------------------------------------------------
# Plot grid
# ---------------------------------------------------------------------------

def plot_grid(proj_data, checkpoints, layers, method_name, position,
              output_path, behavioural=False, sample=None, rng=None,
              only_group=None, limits=None, elev=22, azim=-60,
              family_label="OLMo"):
    """
    proj_data: dict[(ckpt, layer)] -> dict con:
        'coords'  : (N, 2)
        'groups'  : (N,)  stringa categoria
        'refused' : (N,)  bool  [solo se behavioural]

    only_group: se dato, colora solo quel gruppo (gli altri vanno nella nuvola
                grigia di sfondo). limits: dict[(ckpt,layer)] -> (xlim, ylim),
                condiviso fra le figure dei tre gruppi.
    """
    n_rows = len(layers)
    n_cols = len(checkpoints)
    any_key = next(iter(proj_data.values()), None)
    ndim = any_key["coords"].shape[1] if any_key is not None else 2
    is3d = ndim == 3

    if is3d:
        fig = plt.figure(figsize=(4.4 * n_cols, 4.0 * n_rows))
        axes = [[fig.add_subplot(n_rows, n_cols, r * n_cols + c + 1,
                                 projection="3d")
                 for c in range(n_cols)] for r in range(n_rows)]
    else:
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

            if only_group is None:
                for group in GROUPS:
                    if behavioural:
                        _scatter_group_behavioural(ax, coords, groups, refused, group)
                    else:
                        _scatter_group_plain(ax, coords, groups, group)
                n_shown = len(groups)
            else:
                # nessuno sfondo: solo il gruppo selezionato, per leggerne la forma
                if behavioural:
                    _scatter_group_behavioural(ax, coords, groups, refused,
                                               only_group, draw_centroid=False)
                else:
                    _scatter_group_plain(ax, coords, groups, only_group,
                                         draw_centroid=False)
                _draw_all_centroids(ax, coords, groups, refused,
                                    only_group, behavioural)
                n_shown = int((groups == only_group).sum())

            if limits is not None and key in limits:
                lim = limits[key]
                ax.set_xlim(*lim[0])
                ax.set_ylim(*lim[1])
                if is3d and len(lim) > 2:
                    ax.set_zlim(*lim[2])

            if is3d:
                # in 3D l'asse y e' quello di profondita': l'etichetta di riga
                # ci finirebbe illeggibile, quindi il layer va nel titolo
                ax.set_title(f"{ckpt.replace('__', ' ')} — layer {layer}",
                             fontsize=8, fontweight="bold")
                ax.set_ylabel(f"{method_name} 2", fontsize=7)
            else:
                if r == 0:
                    ax.set_title(ckpt.replace("__", "\n"),
                                 fontsize=8, fontweight="bold")
                if c == 0:
                    ax.set_ylabel(f"layer {layer}\n{method_name} 2", fontsize=7)
            if r == n_rows - 1 or is3d:
                ax.set_xlabel(f"{method_name} 1", fontsize=7)
            if is3d:
                ax.set_zlabel(f"{method_name} 3", fontsize=7)
                ax.view_init(elev=elev, azim=azim)
            ax.tick_params(labelsize=6)
            ntot = len(groups)
            txt = f"n={ntot}" if only_group is None else f"n={n_shown}/{ntot}"
            txt_fn = ax.text2D if is3d else ax.text
            txt_fn(0.03, 0.97, txt, transform=ax.transAxes,
                   fontsize=6, ha="left", va="top", color="gray")

    handles = make_legend_handles(behavioural, only_group)
    ncol_legend = 4 if not behavioural else 3
    if only_group is not None:
        ncol_legend = 3
    fig.legend(handles=handles, loc="lower center", ncol=ncol_legend,
               fontsize=8, bbox_to_anchor=(0.5, -0.03))

    beh_tag = " [behavioural]" if behavioural else ""
    grp_tag = "" if only_group is None else f"  |  gruppo: {GROUP_LABEL[only_group]}"
    scale_tag = "" if only_group is None else "  (assi condivisi fra i gruppi)"
    fig.suptitle(
        f"{family_label} — {method_name}  |  {position}{beh_tag}{grp_tag}{scale_tag}",
        fontsize=10, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  [ok] {output_path}")
    plt.close(fig)


def emit_figures(proj_data, checkpoints, layers, method_name, position,
                 output_dir, stem, behavioural, sample, rng, split_groups,
                 elev=22, azim=-60, family_label="OLMo"):
    """Una figura sola, oppure una per gruppo con limiti condivisi."""
    out = Path(output_dir)
    if split_groups is None:
        plot_grid(proj_data, checkpoints, layers, method_name, position,
                  out / f"{stem}.png", behavioural=behavioural,
                  sample=sample, rng=rng, elev=elev, azim=azim,
                  family_label=family_label)
        return
    groups = split_groups if split_groups else GROUPS
    # limiti calcolati su TUTTI i punti, non solo sui gruppi richiesti:
    # cosi' le figure restano confrontabili anche fra run con selezioni diverse
    limits = compute_limits(proj_data)
    for g in groups:
        plot_grid(proj_data, checkpoints, layers, method_name, position,
                  out / f"{stem}_{g}.png", behavioural=behavioural,
                  sample=sample, rng=rng, only_group=g, limits=limits,
                  elev=elev, azim=azim, family_label=family_label)


def project_all(df_full, checkpoints, layers, position, method, seed,
                umap_neighbors=15, umap_min_dist=0.1, n_components=2):
    """
    Fitta la riduzione UNA volta per (checkpoint, layer) su tutti i punti.
    E' questo che rende i pannelli per gruppo confrontabili: la proiezione
    non dipende da quale gruppo viene poi disegnato.
    """
    if method == "UMAP":
        import umap

    proj = {}
    for ckpt in checkpoints:
        df_ckpt = df_full[df_full["checkpoint"] == ckpt]
        for layer in layers:
            col = f"layer_{layer}_{position}"
            if col not in df_ckpt.columns or len(df_ckpt) == 0:
                continue
            X = np.stack(df_ckpt[col].apply(
                lambda v: np.asarray(v, dtype=np.float32)).values)
            groups  = df_ckpt["group"].values
            refused = df_ckpt["predicted_refusal"].values.astype(bool) \
                      if "predicted_refusal" in df_ckpt.columns \
                      else np.zeros(len(groups), dtype=bool)

            if method == "PCA":
                red = PCA(n_components=n_components, random_state=seed)
                coords = red.fit_transform(X)
                var = red.explained_variance_ratio_
                print(f"  {ckpt} layer {layer}: var="
                      + "+".join(f"{v:.2f}" for v in var)
                      + f" (tot {var.sum():.2f})")
            else:
                red = umap.UMAP(n_components=n_components, n_neighbors=umap_neighbors,
                                min_dist=umap_min_dist, random_state=seed,
                                verbose=False)
                coords = red.fit_transform(X)
                print(f"  {ckpt} layer {layer}: ok")

            proj[(ckpt, layer)] = {"coords": coords, "groups": groups,
                                   "refused": refused}
    return proj


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
                        help="Distingui decisione corretta da errore usando "
                             "la colonna predicted_refusal / il judge")
    parser.add_argument("--split-groups", nargs="*", default=None,
                        choices=GROUPS, metavar="GROUP",
                        help="Produci una figura separata per ciascun gruppo indicato "
                             f"(scelte: {' | '.join(GROUPS)}). Senza argomenti li fa "
                             "tutti e tre. Stessa proiezione e stessi limiti degli "
                             "assi su tutte le figure, quindi sono sovrapponibili. "
                             "Se omesso, una sola figura con tutti i gruppi insieme.")
    parser.add_argument("--model-family", default=None, choices=[None, "olmo2", "olmo3"],
                        help="Se dato, legge ANCHE il path nested data/{model_family}/{ckpt}/ "
                             "(merge col path flat) — serve per i 10 dataset completi.")
    parser.add_argument("--nested-only", action="store_true",
                        help="Con --model-family: legge SOLO il path nested. Consigliato per OLMo3.")
    parser.add_argument("--refusal-source", default="judge", choices=["judge", "keyword"],
                        help="Con --behavioural: 'judge' (default) usa il giudizio del giudice "
                             "via merge con --raw-results-csv; 'keyword' usa predicted_refusal "
                             "grezzo (comportamento legacy).")
    parser.add_argument("--raw-results-csv", default=None,
                        help="Richiesto se --refusal-source judge (default) e --behavioural sono attivi.")
    parser.add_argument("--n-components", type=int, default=2, choices=[2, 3],
                        help="2 (default) o 3 componenti. Con 3 i pannelli "
                             "diventano assi 3D; la proiezione resta fittata "
                             "una volta per cella e i limiti restano condivisi "
                             "fra i gruppi.")
    parser.add_argument("--elev", type=float, default=22,
                        help="Solo 3D: elevazione della vista.")
    parser.add_argument("--azim", type=float, default=-60,
                        help="Solo 3D: azimut della vista.")
    parser.add_argument("--sample",          type=int, default=None,
                        help="Max punti per subplot (dopo la proiezione)")
    parser.add_argument("--no-umap",         action="store_true",
                        help="Salta UMAP (piu' veloce, solo PCA)")
    parser.add_argument("--umap-neighbors",  type=int, default=15)
    parser.add_argument("--umap-min-dist",   type=float, default=0.1)
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--output-dir",      default="figures/pca_umap/")
    args = parser.parse_args()

    rng   = np.random.default_rng(args.seed)
    token = load_hf_token(args.hf_token)
    family_label = {"olmo2": "OLMo2", "olmo3": "OLMo3"}.get(args.model_family, "OLMo")
    excl  = set(args.exclude_sources) if args.exclude_sources else None

    for position in args.position:
        print(f"\n{'='*60}")
        print(f"Position: {position}  |  behavioural={args.behavioural}"
              f"  |  split_groups={args.split_groups}")
        print(f"{'='*60}")
        print(f"[.] Carico dati ...")
        df_full = load_data(
            args.hf_dataset, args.checkpoints, args.layers,
            position, token, excl, behavioural=args.behavioural,
            model_family=args.model_family, nested_only=args.nested_only,
            refusal_source=args.refusal_source, raw_results_csv=args.raw_results_csv,
        )
        if df_full is None:
            continue

        beh_suffix = "_behavioural" if args.behavioural else ""
        if args.n_components == 3:
            beh_suffix += "_3d"

        # ---------------------------------------------------------------
        # PCA
        # ---------------------------------------------------------------
        print(f"[.] PCA ...")
        pca_data = project_all(df_full, args.checkpoints, args.layers,
                               position, "PCA", args.seed,
                               n_components=args.n_components)
        emit_figures(pca_data, args.checkpoints, args.layers, "PCA", position,
                     args.output_dir, f"pca_{position}{beh_suffix}",
                     args.behavioural, args.sample, rng, args.split_groups,
                     elev=args.elev, azim=args.azim, family_label=family_label)

        # ---------------------------------------------------------------
        # UMAP
        # ---------------------------------------------------------------
        if not args.no_umap:
            try:
                import umap  # noqa: F401
            except ImportError:
                print("[!] umap-learn non installato. Salta UMAP.")
                print("    Installa con: pip install umap-learn")
                continue

            print(f"[.] UMAP (n_neighbors={args.umap_neighbors}, "
                  f"min_dist={args.umap_min_dist}) ...")
            umap_data = project_all(df_full, args.checkpoints, args.layers,
                                    position, "UMAP", args.seed,
                                    args.umap_neighbors, args.umap_min_dist,
                                    n_components=args.n_components)
            emit_figures(umap_data, args.checkpoints, args.layers, "UMAP", position,
                         args.output_dir, f"umap_{position}{beh_suffix}",
                         args.behavioural, args.sample, rng, args.split_groups,
                         elev=args.elev, azim=args.azim,
                         family_label=family_label)

    print("\n[ok] Fatto.")


if __name__ == "__main__":
    main()