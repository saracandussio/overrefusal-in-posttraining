"""Step 2 — score every response with the coherence + GA/PD judge.

Only rows never judged (is_coherent empty) are sent, so rerunning resumes.
Progress is saved every 50 rows.

    python scripts/judge.py --family olmo2
    python scripts/judge.py --family olmo2 --backend local --workers 1
"""
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from overrefusal import cli, config, judge
from overrefusal.io import save_atomic

COLUMNS = ["is_coherent", "judge_ga", "judge_pd", "judge_ga_reason", "judge_pd_reason"]


def main():
    p = cli.parser(__doc__)
    p.add_argument("--backend", choices=["api", "local"], default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--workers", type=int, default=8, help="parallel API calls")
    args = p.parse_args()

    path = config.raw_results_csv(args.family)
    df = pd.read_csv(path)
    for c in COLUMNS:
        if c not in df:
            df[c] = None
    todo = df.index[df.is_coherent.isna() & df.checkpoint.isin(args.checkpoints)]
    print(f"{len(todo)} rows to judge")

    j = judge.build(args.backend, args.model)
    work = lambda i: (i, j.evaluate(str(df.at[i, "prompt"]), str(df.at[i, "response"])))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for n, (i, result) in enumerate(pool.map(work, todo), 1):
            for c in COLUMNS:
                df.at[i, c] = result.get(c)
            if n % 50 == 0 or n == len(todo):
                save_atomic(df, path)
                print(f"  {n}/{len(todo)} saved")
    j.close()


if __name__ == "__main__":
    main()
