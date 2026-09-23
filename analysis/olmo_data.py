"""
olmo_data.py — caricamento unico delle attivazioni OLMo2 / OLMo3.

Dopo la migrazione (analysis/migrate_to_unified_repo.py) entrambe le
famiglie vivono in un'unica repo, con path sempre prefissati dalla
famiglia — nessuna possibilita' di mescolarle come successo con i path
"flat" della vecchia saracandu/olmo-activations:

    saracandu/overrefusal-activations
        data/olmo2/{ckpt}/shard_*.parquet
        data/olmo3/{ckpt}/shard_*.parquet

La migrazione e' stata una copia DIRETTA degli shard originali (nessuna
ricostruzione), quindi lo schema per-riga e' identico a quello delle repo
sorgente: prompt, label, category, source, checkpoint, response,
predicted_refusal, layer_{L}_{pos}, piu' gli eventuali post_instr_*
(7 posizioni per gli shard che venivano dal flat OLMo2, 3 per il nested
OLMo2, 5 per OLMo3 — tutti conservati, numero non uniforme fra le fonti).

Uso:
    from analysis.olmo_data import load_activations, stack

    df = load_activations("olmo2", layers=[19, 26],
                          positions=["last_prompt", "first_gen"])
    X  = stack(df[df.checkpoint == "sft__none"], "layer_19_last_prompt")

Colonne garantite nel DataFrame restituito:
    prompt, source, category, checkpoint,
    label     gold del dataset sorgente, MAI derivato dal judge
    group     in {"harmless", "pseudo_harm", "harmful"}, da assign_group()
             (split XSTest contrast_* -> harmful)
    refused   bool, SEMPRE dal judge (merge su raw_results.csv) — mai da
             predicted_refusal grezzo
    layer_{L}_{pos} per ogni layer/posizione richiesti

Per cambiare dove stanno i dati, modificare SOLO `FAMILIES`.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Dove stanno i dati
# ---------------------------------------------------------------------------
REPO = "saracandu/overrefusal-activations"

FAMILIES = {
    "olmo2": {
        "repo": REPO,
        "path": "data/olmo2/{ckpt}/*.parquet",
        "raw_results": "results/olmo2/raw_results.csv",
    },
    "olmo3": {
        "repo": REPO,
        "path": "data/olmo3/{ckpt}/*.parquet",
        "raw_results": "results/olmo3/raw_results.csv",
    },
}

CHECKPOINTS = ["base__none", "sft__none", "dpo__none", "final__none"]
GROUPS = ["harmless", "pseudo_harm", "harmful"]

# ---------------------------------------------------------------------------
# Etichettatura
# ---------------------------------------------------------------------------
# Fonti i cui prompt label=0 sono pseudo-harmful (benigni che sembrano dannosi).
PSEUDO_SOURCES = {"or_bench", "false_reject"}

# XSTest: tutte le 450 righe arrivano con label=0, ma le 8 categorie
# contrast_* (200 prompt) sono REALMENTE unsafe; le 10 restanti (250) sono
# le pseudo-harmful. Senza questo split un rifiuto corretto su un contrast_*
# verrebbe contato come over-refusal.
XSTEST_UNSAFE_PREFIX = "contrast_"


def assign_group(df: pd.DataFrame) -> pd.Series:
    g = pd.Series("harmless", index=df.index)
    g[df["label"] == 1] = "harmful"
    g[(df["label"] == 0) & df["source"].isin(PSEUDO_SOURCES)] = "pseudo_harm"

    is_x = df["source"].eq("xstest")
    if is_x.any():
        if "category" not in df.columns:
            raise ValueError("xstest presente senza colonna 'category': "
                             "impossibile separare contrast_* (unsafe) dai safe.")
        unsafe = df["category"].astype(str).str.startswith(XSTEST_UNSAFE_PREFIX)
        g[is_x & unsafe] = "harmful"
        g[is_x & ~unsafe] = "pseudo_harm"
    return g


# ---------------------------------------------------------------------------
# Caricamento
# ---------------------------------------------------------------------------
META_COLS = ["prompt", "source", "category", "label", "checkpoint"]


def hf_token() -> str | None:
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    p = Path("~/.hf_token").expanduser()
    return p.read_text().strip() if p.exists() else None


def _read(repo: str, path: str, cols: list[str], token) -> pd.DataFrame | None:
    """Legge un glob di parquet chiedendo solo le colonne esistenti."""
    from datasets import load_dataset
    try:
        ds = load_dataset(repo, data_files={"train": path}, split="train",
                          token=token)
    except FileNotFoundError:
        return None
    keep = [c for c in cols if c in ds.column_names]
    return ds.select_columns(keep).to_pandas()


def load_activations(
    family: str,
    layers: list[int],
    positions: list[str],
    checkpoints: list[str] = CHECKPOINTS,
    exclude_sources: list[str] | None = None,
    raw_results_csv: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Carica un checkpoint alla volta dalla repo unificata, unisce, etichetta.

    `label`   e' sempre il gold del dataset sorgente, mai toccato.
    `group`   e' derivato da label/source/category via assign_group()
              (split XSTest contrast_* -> harmful).
    `refused` viene SEMPRE dal judge: merge su raw_results.csv. Le righe
              senza un giudizio corrispondente vengono scartate e il
              conteggio stampato — mai un rifiuto stimato da keyword.
    """
    if family not in FAMILIES:
        raise ValueError(f"famiglia sconosciuta: {family} (note: {list(FAMILIES)})")
    cfg = FAMILIES[family]
    token = hf_token()
    act_cols = [f"layer_{l}_{p}" for l in layers for p in positions]
    cols = META_COLS + act_cols

    parts = []
    for ck in checkpoints:
        d = _read(cfg["repo"], cfg["path"].format(ckpt=ck), cols, token)
        if d is None or len(d) == 0:
            if verbose:
                print(f"  [skip] {family}/{ck}: nessun dato")
            continue
        d["checkpoint"] = ck
        missing = [c for c in act_cols if c not in d.columns]
        if missing and verbose:
            print(f"  [warn] {family}/{ck}: mancano {missing}")
        parts.append(d)
        if verbose:
            print(f"  {family}/{ck}: {len(d)} righe")

    if not parts:
        raise RuntimeError(f"nessun dato caricato per {family}")
    df = pd.concat(parts, ignore_index=True)

    if exclude_sources:
        df = df[~df["source"].isin(set(exclude_sources))].reset_index(drop=True)

    df["group"] = assign_group(df)

    from analysis.judge_utils import attach_judge_refusal
    csv = raw_results_csv or cfg["raw_results"]
    n0 = len(df)
    df = attach_judge_refusal(df, csv, drop_missing=True)
    df["refused"] = df["judge_refusal"].astype(bool)
    if verbose:
        print(f"  judge ({csv}): {n0} -> {len(df)} righe")
        summarize(df)

    return df


def summarize(df: pd.DataFrame) -> None:
    """Composizione per gruppo e source, piu' tasso di rifiuto per checkpoint."""
    print("  composizione (tutti i checkpoint):")
    for g in GROUPS:
        sub = df[df["group"] == g]
        if len(sub) == 0:
            print(f"    {g:<12} VUOTO")
            continue
        print(f"    {g:<12} n={len(sub):<6} {sub['source'].value_counts().to_dict()}")
    if "refused" in df.columns:
        rate = df.pivot_table(index="checkpoint", columns="group",
                              values="refused", aggfunc="mean")
        print("  tasso di rifiuto:")
        for line in rate.reindex(columns=GROUPS).round(3).to_string().splitlines():
            print("    " + line)


def stack(df: pd.DataFrame, col: str, dtype=np.float32) -> np.ndarray:
    """Colonna di vettori -> matrice (n, d)."""
    return np.stack(df[col].values).astype(dtype)