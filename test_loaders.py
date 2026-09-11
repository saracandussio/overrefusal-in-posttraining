"""
test_loaders.py

Verifica che tutti i loader dei nuovi dataset funzionino correttamente.
Stampa: n righe, distribuzione label, prime 2 righe del prompt.

Usage:
    python test_loaders.py
    python test_loaders.py --datasets xstest alpaca advbench  # solo alcuni
"""

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def test_dataset(key: str, cfg, loader_fn) -> bool:
    print(f"\n{'='*60}")
    print(f"Dataset: {key}")
    print(f"  hf_path  : {cfg.hf_path}")
    print(f"  hf_split : {cfg.hf_split}")
    print(f"  type     : {cfg.dataset_type}")
    print(f"  max      : {cfg.max_samples}")

    try:
        df = loader_fn(cfg)
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False

    print(f"  rows     : {len(df)}")
    print(f"  columns  : {list(df.columns)}")
    print(f"  labels   : {df['label'].value_counts().to_dict()}")
    print(f"  sources  : {df['source'].unique().tolist()}")

    if df["prompt"].str.len().min() == 0:
        print(f"  [WARN] prompt vuoti trovati: {(df['prompt'].str.len() == 0).sum()}")

    print(f"  --- prime 2 righe ---")
    for i, row in df.head(2).iterrows():
        prompt_preview = row["prompt"][:120].replace("\n", " ")
        print(f"  [{i}] label={row['label']} cat={row['category']!r}")
        print(f"       {prompt_preview!r}")

    print(f"  [OK]")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Subset di dataset da testare (default: tutti i nuovi)"
    )
    args = parser.parse_args()

    # Import locali per evitare dipendenze circolari
    try:
        from dataset_config import ALL_DATASETS
        from data.dataset_loader import _LOADER_MAP
    except ImportError as e:
        print(f"[ERROR] Import fallito: {e}")
        print("Assicurati di girare lo script dalla directory del progetto.")
        sys.exit(1)

    # Default: testa solo i dataset nuovi
    default_keys = ["xstest", "alpaca", "advbench", "jailbreakbench", "advbench"]
    keys_to_test = args.datasets if args.datasets else default_keys

    results = {}
    for key in keys_to_test:
        if key not in ALL_DATASETS:
            print(f"\n[SKIP] '{key}' non in ALL_DATASETS")
            results[key] = None
            continue
        cfg = ALL_DATASETS[key]
        loader_fn = _LOADER_MAP.get(key)
        if loader_fn is None:
            print(f"\n[SKIP] nessun loader per '{key}'")
            results[key] = None
            continue
        results[key] = test_dataset(key, cfg, loader_fn)

    print(f"\n{'='*60}")
    print("SUMMARY")
    for key, ok in results.items():
        status = "OK" if ok else ("SKIP" if ok is None else "FAIL")
        print(f"  {key:20s} {status}")


if __name__ == "__main__":
    main()