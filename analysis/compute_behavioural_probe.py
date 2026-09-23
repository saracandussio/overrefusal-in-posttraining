"""
compute_behavioral_probe.py

Addestra una probe logistica che predice predicted_refusal (0/1)
dalle attivazioni first_gen, separatamente per ogni checkpoint e layer.

Poi fa cross-checkpoint transfer: addestra sul base e testa su SFT/DPO/Final.

Se la probe semantica (cat3) è stabile tra checkpoint ma quella comportamentale
migliora da SFT a DPO, la dissociazione geometria/comportamento è dimostrata
su due probe indipendenti.

Usage:
    python compute_behavioral_probe.py
    python compute_behavioral_probe.py --exclude-sources beavertails
    python compute_behavioral_probe.py --out results/olmo2/geometry/behavioral_probe.csv
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, cross_val_predict

HF_REPO = "saracandu/olmo-activations"
CHECKPOINTS = ["base__none", "sft__none", "dpo__none", "final__none"]
LAYERS = [8, 16, 19, 24, 26, 31]

SAVE_DIR = Path("results/olmo2/classifiers")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def _data_paths(ckpt: str, model_family, nested_only: bool) -> list[str]:
    """Same flat/nested logic as explore_comprehension_decision.py / check_sources.py."""
    flat = f"data/{ckpt}/*.parquet"
    nested = f"data/{model_family}/{ckpt}/*.parquet" if model_family else None
    if nested_only:
        if not nested:
            raise SystemExit("--nested-only requires --model-family to be set.")
        return [nested]
    if nested:
        return [flat, nested]
    return [flat]


def load_checkpoint(hf_repo, ckpt, cols, token, exclude_sources=None,
                     model_family=None, nested_only=False):
    """Carica un checkpoint (merge flat+nested se --model-family è dato) e filtra le source."""
    paths = _data_paths(ckpt, model_family, nested_only)
    dfs = []
    for path in paths:
        try:
            ds = load_dataset(hf_repo, data_files={"train": path}, split="train", token=token)
        except Exception as e:
            print(f"    [warn] {ckpt} {path}: skipping ({e})")
            continue
        available = [c for c in cols if c in ds.column_names]
        dfs.append(ds.select_columns(available).to_pandas())

    if not dfs:
        raise RuntimeError(f"No data loaded for checkpoint {ckpt} (paths tried: {paths})")

    df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

    if exclude_sources:
        before = len(df)
        df = df[~df["source"].isin(exclude_sources)].reset_index(drop=True)
        print(f"    [exclude] {ckpt}: {before} -> {len(df)} righe (rimosso {exclude_sources})")
    return df


def apply_refusal_source(df, refusal_source, raw_results_csv):
    """
    Se refusal_source == 'judge', sovrascrive predicted_refusal con
    judge_refusal (merge con raw_results_csv su checkpoint+source+prompt).
    Se 'keyword', lascia predicted_refusal com'e' (comportamento legacy).
    """
    if refusal_source != "judge":
        return df
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from judge_utils import attach_judge_refusal
    n_before = len(df)
    df = attach_judge_refusal(df, raw_results_csv, drop_missing=True)
    df["predicted_refusal"] = df["judge_refusal"]
    if len(df) != n_before:
        print(f"    [judge merge] {n_before} -> {len(df)} righe "
              f"(droppate quelle senza giudizio/incoerenti)")
    return df


def make_clf(balanced: bool) -> LogisticRegression:
    return LogisticRegression(
        max_iter=1000, C=0.1, random_state=42,
        class_weight="balanced" if balanced else None,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-repo", default=HF_REPO)
    parser.add_argument("--model-family", default=None, choices=[None, "olmo2", "olmo3"],
                        help="If set, ALSO reads the nested data/{model_family}/{ckpt}/ path "
                             "(merged with the flat path) — needed to get all 10 datasets, "
                             "not just the 5 original ones.")
    parser.add_argument("--nested-only", action="store_true",
                        help="With --model-family set, read ONLY the nested path. "
                             "Recommended for OLMo3.")
    parser.add_argument("--layers", nargs="+", type=int, default=LAYERS)
    parser.add_argument("--out", default="results/olmo2/geometry/behavioral_probe.csv")
    parser.add_argument("--exclude-sources", nargs="*", default=None,
                        metavar="SOURCE",
                        help="Source da escludere completamente "
                             "(es. --exclude-sources beavertails)")
    parser.add_argument("--raw-results-csv", default="results/olmo2/raw_results.csv",
                        help="CSV con judge_ga/judge_pd per il target basato sul giudice.")
    parser.add_argument("--refusal-source", default="judge", choices=["judge", "keyword"],
                        help="'judge' (default): target = judge_refusal (via merge "
                             "con --raw-results-csv). 'keyword': comportamento legacy, "
                             "target = predicted_refusal.")
    parser.add_argument("--balanced", action="store_true",
                        help="Usa class_weight='balanced' nella regressione logistica "
                             "(ripesa ogni esempio per l'inverso della frequenza della "
                             "sua classe) — utile quando il dataset concatenato mescola "
                             "source con tassi di rifiuto molto diversi.")
    args = parser.parse_args()

    exclude = set(args.exclude_sources) if args.exclude_sources else None

    token = os.environ.get("HF_TOKEN") or open(
        os.path.expanduser("~/.hf_token")
    ).read().strip()

    rows = []

    # -----------------------------------------------------------------------
    # Phase 1: train per checkpoint, cross-val accuracy
    # -----------------------------------------------------------------------
    print("=" * 65)
    print("PHASE 1 — Behavioral probe per checkpoint (cross-val, per-source breakdown)")
    print("=" * 65)

    for ckpt in CHECKPOINTS:
        act_cols = [f"layer_{l}_first_gen" for l in args.layers]
        base_cols = ["label", "source", "predicted_refusal", "checkpoint", "prompt"]
        df = load_checkpoint(args.hf_repo, ckpt, base_cols + act_cols, token, exclude,
                              model_family=args.model_family, nested_only=args.nested_only)
        df = apply_refusal_source(df, args.refusal_source, args.raw_results_csv)

        for l in args.layers:
            col = f"layer_{l}_first_gen"

            X_all = np.stack(df[col].values).astype(np.float32)
            y_all = df["predicted_refusal"].values.astype(int)

            clf = make_clf(args.balanced)
            auroc_all = cross_val_score(clf, X_all, y_all, cv=5, scoring="roc_auc").mean()
            acc_all = cross_val_score(clf, X_all, y_all, cv=5, scoring="accuracy").mean()

            # Probabilità out-of-fold (ogni riga scorata da un fold in cui NON
            # era nel training) — serve per AUROC e accuratezza per-source.
            y_proba = cross_val_predict(clf, X_all, y_all, cv=5, method="predict_proba")[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)
            df_pred = df[["source"]].copy()
            df_pred["y_true"] = y_all
            df_pred["y_proba"] = y_proba
            df_pred["y_pred"] = y_pred

            per_source_rows = []
            for source, g in df_pred.groupby("source"):
                if g["y_true"].nunique() < 2:
                    auroc_src = float("nan")  # AUROC non definita se la source ha una sola classe
                else:
                    auroc_src = roc_auc_score(g["y_true"], g["y_proba"])
                acc_src = (g["y_pred"] == g["y_true"]).mean()
                refusal_rate = g["y_true"].mean()
                baseline_acc = max(refusal_rate, 1 - refusal_rate)  # indovina sempre la classe maggioritaria
                per_source_rows.append({
                    "source": source,
                    "n": len(g),
                    "auroc": auroc_src,
                    "acc": acc_src,
                    "acc_vs_baseline": acc_src - baseline_acc,  # margine sopra il banale
                    "refusal_rate": refusal_rate,
                })
            per_source = pd.DataFrame(per_source_rows).set_index("source").sort_index()

            print(f"\n{ckpt}  layer={l}  auroc_all={auroc_all:.3f}  acc_all={acc_all:.3f}  (n={len(df)})")
            print(per_source.to_string(float_format=lambda x: f"{x:.3f}"))

            rows.append({
                "phase":      "within",
                "train_on":   ckpt,
                "test_on":    ckpt,
                "layer":      l,
                "auroc_all":  auroc_all,
                "acc_all":    acc_all,
            })
            for source, r in per_source.iterrows():
                rows.append({
                    "phase":           "within_by_source",
                    "train_on":        ckpt,
                    "test_on":         ckpt,
                    "layer":           l,
                    "source":          source,
                    "n":               int(r["n"]),
                    "auroc":           r["auroc"],
                    "acc":             r["acc"],
                    "acc_vs_baseline": r["acc_vs_baseline"],
                    "refusal_rate":    r["refusal_rate"],
                })

            # Salva clf addestrato su tutti i dati
            clf_full = make_clf(args.balanced)
            clf_full.fit(X_all, y_all)
            save_path = SAVE_DIR / f"clf_beh_{ckpt}_layer{l}.pkl"
            with open(save_path, "wb") as f:
                pickle.dump({"clf": clf_full, "trained_on": ckpt, "layer": l}, f)

        print()

    # -----------------------------------------------------------------------
    # Phase 2: cross-checkpoint transfer
    # Addestra su base → testa su tutti
    # Addestra su SFT  → testa su DPO/Final
    # -----------------------------------------------------------------------
    print("=" * 65)
    print("PHASE 2 — Cross-checkpoint transfer")
    print("=" * 65)

    for train_ckpt, test_ckpts in [
        ("base__none", ["base__none", "sft__none", "dpo__none", "final__none"]),
        ("sft__none",  ["sft__none",  "dpo__none", "final__none"]),
    ]:
        print(f"\nTrained on: {train_ckpt}")

        # Load each test checkpoint ONCE (all layers' columns together),
        # instead of once per (layer, test_ckpt) — was causing 6x redundant
        # HF downloads + judge merges per train_ckpt.
        act_cols_all = [f"layer_{l}_first_gen" for l in args.layers]
        test_dfs = {}
        for ckpt in test_ckpts:
            df_ckpt = load_checkpoint(
                args.hf_repo, ckpt,
                ["source", "predicted_refusal", "checkpoint", "prompt"] + act_cols_all,
                token, exclude,
                model_family=args.model_family, nested_only=args.nested_only
            )
            df_ckpt = apply_refusal_source(df_ckpt, args.refusal_source, args.raw_results_csv)
            test_dfs[ckpt] = df_ckpt

        for l in args.layers:
            clf_path = SAVE_DIR / f"clf_beh_{train_ckpt}_layer{l}.pkl"
            with open(clf_path, "rb") as f:
                saved = pickle.load(f)
            clf_train = saved["clf"]

            col = f"layer_{l}_first_gen"
            for ckpt in test_ckpts:
                df = test_dfs[ckpt]
                X = np.stack(df[col].values).astype(np.float32)
                y = df["predicted_refusal"].values.astype(int)
                if len(set(y)) < 2:
                    auroc = float("nan")
                else:
                    y_proba = clf_train.predict_proba(X)[:, 1]
                    auroc = roc_auc_score(y, y_proba)
                print(f"  layer={l:>2}  test_on={ckpt:<20}  auroc={auroc:.3f}")

                rows.append({
                    "phase":      "cross",
                    "train_on":   train_ckpt,
                    "test_on":    ckpt,
                    "layer":      l,
                    "auroc":      auroc,
                })

    # Salva CSV
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSalvato in {out}")


if __name__ == "__main__":
    main()
