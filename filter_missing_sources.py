"""
filter_missing_sources.py

Filtra results/olmo2/raw_results.csv alle sole righe che servono per
completare le attivazioni su saracandu/olmo-activations, evitando di
rifare il forward-pass GPU su source già estratte.

Sulla base del controllo con check_sources.py:
  presenti nelle attivazioni  : beavertails, false_reject, harmbench,
                                 or_bench, toxicchat
  presenti nel CSV ma NON     : jailbreakbench, xstest, alpaca, advbench
  ancora nelle attivazioni
  assenti anche dal CSV       : wildguard (va rigenerato con run_experiment.py
                                 prima di poter essere estratto)

Usage
-----
    python filter_missing_sources.py
    python filter_missing_sources.py --sources jailbreakbench xstest alpaca advbench
    python filter_missing_sources.py --checkpoint-suffix __none
"""

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_MISSING_SOURCES = ["jailbreakbench", "xstest", "alpaca", "advbench"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/olmo2/raw_results.csv")
    parser.add_argument("--sources", nargs="+", default=DEFAULT_MISSING_SOURCES)
    parser.add_argument("--checkpoint-suffix", default="__none",
                        help="Only keep checkpoints ending with this suffix "
                             "(set to '' to keep all system prompts).")
    parser.add_argument("--out", default="results/olmo2/raw_results_missing_sources.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} total rows from {args.csv}")

    before = len(df)
    if args.checkpoint_suffix:
        df = df[df["checkpoint"].str.endswith(args.checkpoint_suffix)]
        print(f"After checkpoint filter '{args.checkpoint_suffix}': {len(df)}/{before} rows")

    before = len(df)
    df = df[df["source"].isin(args.sources)]
    print(f"After source filter {args.sources}: {len(df)}/{before} rows")

    print("\nRows per (checkpoint, source):")
    print(df.groupby(["checkpoint", "source"]).size().to_string())

    if len(df) == 0:
        print("\nWARNING: filtered CSV is empty — nothing to extract. "
              "Check --sources / --checkpoint-suffix match what's actually "
              "in the CSV (case-sensitive).")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved filtered CSV to {out_path} ({len(df)} rows)")
    print(
        "\nNext step:\n"
        f"  python analysis/extract_and_push.py \\\n"
        f"      --csv {out_path} \\\n"
        f"      --hf-repo saracandu/olmo-activations \\\n"
        f"      --checkpoint-filter '*{args.checkpoint_suffix}' \\\n"
        f"      --batch-size 32 --device cuda"
    )


if __name__ == "__main__":
    main()
