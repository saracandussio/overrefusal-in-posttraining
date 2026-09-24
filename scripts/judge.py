"""Step 2 — score every response with the coherence + GA/PD judge.

Main mode writes into raw_results.csv: only rows never judged (is_coherent
empty) are sent, so rerunning resumes. Progress is saved every 50 rows.

--tag NAME runs a *second* judge instead: results go to
results/<family>/judges/NAME.csv, raw_results.csv is only read, and rows
already in that file are skipped. Use it for calibration (with --sample) and
for cross-judge robustness; compare with scripts/judge_agreement.py.

    python scripts/judge.py --family olmo2
    python scripts/judge.py --family olmo2 --tag deepseek --model deepseek-ai/DeepSeek-V4-Flash-0731
    python scripts/judge.py --family olmo2 --tag gptoss_local --sample 300 \\
        --checkpoints sft__none dpo__none --model openai/gpt-oss-120b
"""
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from overrefusal import cli, config, judge
from overrefusal.io import save_atomic
from overrefusal.refusal import KEY

COLUMNS = ["is_coherent", "judge_ga", "judge_pd", "judge_ga_reason", "judge_pd_reason"]


def rows_for_tag(raw: pd.DataFrame, checkpoints, path, sample, model) -> pd.DataFrame:
    """The tag file: every row to judge, with results filled in as they come."""
    rows = raw[raw.checkpoint.isin(checkpoints)]
    if sample:  # calibration: only rows the main judge has scored
        rows = rows[rows.is_coherent.notna()].sample(n=sample, random_state=0)
    out = rows[KEY + ["response"]].drop_duplicates(KEY).reset_index(drop=True)
    if path.exists():
        done = pd.read_csv(path)
        out = out.merge(done.drop(columns="response", errors="ignore"), on=KEY, how="left")
    for c in COLUMNS:
        if c not in out:
            out[c] = None
    out["judge_model"] = model
    return out


def main():
    p = cli.parser(__doc__)
    p.add_argument("--backend", choices=["api", "local"], default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--workers", type=int, default=8, help="parallel API calls")
    p.add_argument("--tag", default=None, help="second judge: write to judges/<tag>.csv")
    p.add_argument("--sample", type=int, default=0, help="with --tag: judge only N scored rows")
    args = p.parse_args()

    raw = pd.read_csv(config.raw_results_csv(args.family))
    if args.tag:
        path = config.results_dir(args.family) / "judges" / f"{args.tag}.csv"
        df = rows_for_tag(raw, args.checkpoints, path, args.sample, args.model)
    else:
        path, df = config.raw_results_csv(args.family), raw
        for c in COLUMNS:
            if c not in df:
                df[c] = None

    todo = df.index[df.is_coherent.isna() & df.checkpoint.isin(args.checkpoints)]
    print(f"{len(todo)} rows to judge -> {path}")

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
