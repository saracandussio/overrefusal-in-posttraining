"""Put new numbers next to the old ones, cell by cell.

Not everything is expected to match: groups now include XSTest (split into
contrast_* harmful and the rest pseudo), beavertails is gone everywhere, and
refusal always comes from the judge. So this prints the differences and
flags the large ones, for you to explain one by one rather than to pass/fail.

    python scripts/check_old.py --family olmo2 --old-dir ../old-repo/results/olmo2
"""
from pathlib import Path

import pandas as pd

from overrefusal import cli, config

TOLERANCE = 0.05

# new file, old file, how to turn old rows into new keys and columns
COMPARISONS = [
    ("geometry/axis.csv", "geometry/ent_first_meandiff.csv",
     lambda o: o.assign(position="first_gen", t=(o.boundary_margin_n + 1) / 2),
     ["checkpoint", "layer", "position"],
     ["entanglement", "t", "cos_vbeh_vref", "cos_vbeh_vover"]),
    ("geometry/axis.csv", "geometry/ent_last_meandiff.csv",
     lambda o: o.assign(position="last_prompt", t=(o.boundary_margin_n + 1) / 2),
     ["checkpoint", "layer", "position"],
     ["entanglement", "t"]),
]


def compare(new: pd.DataFrame, old: pd.DataFrame, keys, columns) -> pd.DataFrame:
    if "pseudo_source" in new:
        new = new[new.pseudo_source == "all"]
    both = new[keys + columns].merge(old[keys + columns], on=keys, suffixes=("", "_old"))
    for c in columns:
        both[f"{c}_delta"] = both[c] - both[f"{c}_old"]
    return both


def main():
    p = cli.parser(__doc__)
    p.add_argument("--old-dir", required=True, type=Path)
    args = p.parse_args()
    new_dir = config.results_dir(args.family)

    for new_file, old_file, adapt, keys, columns in COMPARISONS:
        if not (new_dir / new_file).exists() or not (args.old_dir / old_file).exists():
            print(f"skip {new_file} vs {old_file}: file missing")
            continue
        both = compare(pd.read_csv(new_dir / new_file),
                       adapt(pd.read_csv(args.old_dir / old_file)), keys, columns)
        deltas = both[[f"{c}_delta" for c in columns]].abs()
        big = both[(deltas > TOLERANCE).any(axis=1)]
        print(f"\n{new_file} vs {old_file}: {len(both)} cells, {len(big)} differ by > {TOLERANCE}")
        if len(big):
            print(big[keys + [f"{c}_delta" for c in columns]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
