"""How much two judges agree, on the rows both have scored.

Compares a second judge (results/<family>/judges/<tag>.csv) with the main
one (raw_results.csv): coherence, GA, PD, and the derived refused/answered,
overall and per checkpoint. Cohen's kappa corrects for chance agreement;
as a rule of thumb above 0.8 the two judges are interchangeable, 0.6-0.8
they agree on the picture but not on every row.

    python scripts/judge_agreement.py --family olmo2 --tag gptoss_local
"""
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from overrefusal import cli, config
from overrefusal.refusal import KEY, judge_refusal


def agreement(a: pd.Series, b: pd.Series) -> dict:
    ok = a.notna() & b.notna()
    a, b = a[ok].astype(int), b[ok].astype(int)
    return {"n": int(ok.sum()), "exact": float((a == b).mean()),
            "kappa": float(cohen_kappa_score(a, b)) if a.nunique() > 1 or b.nunique() > 1 else float("nan")}


def main():
    p = cli.parser(__doc__)
    p.add_argument("--tag", required=True)
    args = p.parse_args()

    main_ = pd.read_csv(config.raw_results_csv(args.family))
    other = pd.read_csv(config.results_dir(args.family) / "judges" / f"{args.tag}.csv")
    cols = ["is_coherent", "judge_ga", "judge_pd"]
    both = main_[KEY + cols].merge(other[KEY + cols], on=KEY, suffixes=("", "_2"))
    both = both[both.is_coherent.notna() & both.is_coherent_2.notna()]
    for side in ("", "_2"):
        view = both[[f"{c}{side}" for c in cols]].set_axis(cols, axis=1)
        both[f"refused{side}"] = judge_refusal(view)

    rows = []
    for ckpt, g in [("all", both)] + list(both.groupby("checkpoint")):
        for name in ["is_coherent", "judge_ga", "judge_pd", "refused"]:
            rows.append({"checkpoint": ckpt, "measure": name, **agreement(g[name], g[f"{name}_2"])})
    table = pd.DataFrame(rows)
    print(f"{args.tag} vs main judge, {len(both)} rows scored by both")
    print(table.pivot_table(index="measure", columns="checkpoint", values=["exact", "kappa"],
                            sort=False).round(2).to_string())
    rate = both.groupby("checkpoint")[["refused", "refused_2"]].mean().round(3)
    print("\nrefusal rate, main vs second judge\n" + rate.to_string())


if __name__ == "__main__":
    main()
