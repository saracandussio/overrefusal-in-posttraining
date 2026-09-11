"""
check_sources.py

Conta le righe per (checkpoint, group, source) SENZA scaricare nessuna
colonna layer_*_{last_prompt,first_gen,post_instr_k} — solo label, source,
checkpoint. Queste tre colonne pesano pochi byte a riga contro le migliaia
di float delle attivazioni, quindi anche caricando l'intero dataset (tutti
i checkpoint, tutte le righe) il costo in RAM/rete è trascurabile rispetto
a un load_dataset() completo.

Usa `columns=` per fare column pruning a livello di lettura Parquet, non
select_columns() dopo — quello scaricherebbe comunque tutto.

Uso
---
    python check_sources.py
    python check_sources.py --checkpoints sft__none dpo__none
"""

import argparse
import os

import pandas as pd


CHECKPOINT_ORDER = [
    "base__none", "base__mistral_safety",
    "sft__none", "sft__mistral_safety",
    "dpo__none", "dpo__mistral_safety",
    "final__none", "final__mistral_safety",
]

PSEUDO_HARM_SOURCES = {"or_bench", "false_reject"}


def assign_group(df: pd.DataFrame) -> pd.Series:
    g = pd.Series("harmless", index=df.index)
    g[df["label"] == 1] = "harmful"
    g[(df["label"] == 0) & (df["source"].isin(PSEUDO_HARM_SOURCES))] = "pseudo_harm"
    return g


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-repo", default="saracandu/olmo-activations")
    parser.add_argument("--checkpoints", nargs="+", default=CHECKPOINT_ORDER)
    parser.add_argument(
        "--model-family", default=None, choices=[None, "olmo2", "olmo3"],
        help="If set, ALSO reads the NESTED path data/{model_family}/{ckpt}/*.parquet "
             "in addition to the old FLAT path data/{ckpt}/*.parquet, and merges both "
             "(unless --nested-only is also given). The repo has BOTH structures "
             "coexisting for historical reasons: an old flat one (5 original sources, "
             "extracted before v3 existed — but NOTE the flat path is NOT model-specific, "
             "it can silently contain data from whichever model was pushed there, "
             "olmo2 or olmo3 or an unrelated earlier experiment) and a new nested one "
             "written by extract_and_push.py v3, which IS model-specific. Neither path "
             "alone has the full picture per checkpoint in general — this flag merges "
             "both so you don't have to run this script twice and add numbers by hand.",
    )
    parser.add_argument(
        "--nested-only", action="store_true",
        help="With --model-family set, read ONLY the nested data/{model_family}/{ckpt}/ "
             "path, skipping the flat data/{ckpt}/ path entirely. Use this to isolate "
             "what a specific model's v3 extraction actually contains, without any "
             "risk of the flat path's contents (which may belong to a different model "
             "or an unrelated old experiment) muddying the picture.",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or open(
        os.path.expanduser("~/.hf_token")
    ).read().strip()

    from datasets import load_dataset

    def data_paths(ckpt: str) -> list[str]:
        flat = f"data/{ckpt}/*.parquet"
        nested = f"data/{args.model_family}/{ckpt}/*.parquet" if args.model_family else None
        if args.nested_only:
            if not nested:
                raise SystemExit("--nested-only requires --model-family to be set.")
            return [nested]
        if nested:
            return [flat, nested]
        return [flat]

    def load_one(path: str):
        try:
            # columns= prunes at the Parquet-read level: only these 3
            # columns are ever fetched over the network / materialized.
            return load_dataset(
                args.hf_repo,
                data_files={"train": path},
                split="train",
                token=token,
                columns=["label", "source", "checkpoint"],
            )
        except TypeError:
            # older `datasets` versions may not support columns= here;
            # fall back to select_columns after a normal load (still much
            # cheaper than loading activations, but not column-pruned at
            # the network level).
            ds = load_dataset(
                args.hf_repo,
                data_files={"train": path},
                split="train",
                token=token,
            )
            return ds.select_columns(["label", "source", "checkpoint"])

    all_rows = []
    for ckpt in args.checkpoints:
        ckpt_dfs = []
        for path in data_paths(ckpt):
            try:
                ds = load_one(path)
            except Exception as e:
                print(f"[{ckpt}] {path}: SKIPPED ({e})")
                continue
            ckpt_dfs.append(ds.to_pandas())

        if not ckpt_dfs:
            continue
        df = pd.concat(ckpt_dfs, ignore_index=True) if len(ckpt_dfs) > 1 else ckpt_dfs[0]
        df["group"] = assign_group(df)
        all_rows.append(df)
        print(f"[{ckpt}] {len(df)} rows loaded (source+label only)")

    if not all_rows:
        print("Nothing loaded.")
        return

    full = pd.concat(all_rows, ignore_index=True)

    print("\n=== rows per (checkpoint, group, source) ===")
    tab = full.groupby(["checkpoint", "group", "source"]).size().reset_index(name="n")
    print(tab.to_string(index=False))

    print("\n=== which sources appear at all, across all loaded checkpoints ===")
    print(sorted(full["source"].unique()))

    print("\n=== sources missing per checkpoint vs. the union of all sources seen ===")
    all_sources = set(full["source"].unique())
    for ckpt, sub in full.groupby("checkpoint"):
        missing = all_sources - set(sub["source"].unique())
        if missing:
            print(f"  {ckpt}: missing {sorted(missing)}")
        else:
            print(f"  {ckpt}: all sources present")


if __name__ == "__main__":
    main()
