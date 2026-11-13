"""Start a variant run (OVERREFUSAL_VARIANT=nosys) by copying the base rows.

The base model has no chat template, so its generations and judgements are
the same in every run. This copies the base__none rows of the original
results/<family>/raw_results.csv (read only) into the variant's
raw_results.csv, so the variant run does not regenerate the base. It refuses
to run without a variant, and refuses if the variant file already has rows.

    OVERREFUSAL_VARIANT=nosys python scripts/seed_variant.py --family olmo3
"""
import argparse
from pathlib import Path

import pandas as pd

from overrefusal import config
from overrefusal.io import save_atomic


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", required=True, choices=list(config.MODELS))
    args = p.parse_args()
    if not config.VARIANT:
        raise SystemExit("set OVERREFUSAL_VARIANT first: this only seeds a variant run")
    source = Path("results") / args.family / "raw_results.csv"
    target = config.raw_results_csv(args.family)
    if target.resolve() == source.resolve():
        raise SystemExit("source and target are the same file")
    if target.exists():
        raise SystemExit(f"{target} already exists; nothing written")
    raw = pd.read_csv(source)
    base = raw[raw.checkpoint == "base__none"]
    if base.empty:
        raise SystemExit(f"no base__none rows in {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    save_atomic(base, target)
    print(f"{len(base)} base rows copied from {source} to {target}")


if __name__ == "__main__":
    main()
