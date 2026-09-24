"""Step 1 — generate a response for every prompt, per checkpoint.

Appends to results/<family>/raw_results.csv; rows already there
(same checkpoint, source, prompt) are skipped, so the run can be resumed.

--redo regenerates every prompt of the selected checkpoints from scratch.
The old rows (with their judgements) stay in the file until the new ones
are ready, then are replaced in one write; git keeps the previous version.

    python scripts/generate.py --family olmo2
    python scripts/generate.py --family olmo2 --checkpoints sft__none --datasets xstest
    python scripts/generate.py --family olmo2 --checkpoints base__none --redo
"""
import pandas as pd

from overrefusal import cli, config, datasets, models, prompts
from overrefusal.io import save_atomic
from overrefusal.refusal import KEY, keyword_refusal


def main():
    p = cli.parser(__doc__)
    p.add_argument("--datasets", nargs="+", default=None, choices=list(datasets.DATASETS))
    p.add_argument("--redo", action="store_true",
                   help="regenerate the selected checkpoints, replacing their rows")
    args = p.parse_args()

    out = config.raw_results_csv(args.family)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = pd.read_csv(out, usecols=KEY) if out.exists() else pd.DataFrame(columns=KEY)
    if args.redo:
        done = done[~done.checkpoint.isin(args.checkpoints)]
    pool = datasets.load(args.datasets)

    for ckpt in args.checkpoints:
        stage = config.stage_of(ckpt)
        todo = pool.merge(done[done.checkpoint == ckpt], on=["source", "prompt"],
                             how="left", indicator=True)
        todo = todo[todo._merge == "left_only"].drop(columns=["_merge", "checkpoint"])
        if todo.empty:
            continue

        model, tok = models.load(args.family, stage)
        texts = [prompts.build_prompt(tok, m, stage) for m in todo.prompt]
        g = config.GENERATION
        responses = []
        for i in range(0, len(texts), g.batch_size):
            responses += models.generate(model, tok, texts[i:i + g.batch_size],
                                         g.max_new_tokens, g.max_prompt_tokens)
        models.unload(model)
        responses = [prompts.clean_response(r, stage) for r in responses]

        rows = todo.assign(checkpoint=ckpt, response=responses)
        rows["predicted_refusal"] = keyword_refusal(rows.response)
        # rewrite rather than append: the file may already carry judge columns
        old = pd.read_csv(out) if out.exists() else None
        if old is not None and args.redo:
            print(f"{ckpt}: replacing {int((old.checkpoint == ckpt).sum())} old rows")
            old = old[old.checkpoint != ckpt]
        save_atomic(pd.concat([old, rows], ignore_index=True), out)
        done = pd.concat([done, rows[KEY]], ignore_index=True)


if __name__ == "__main__":
    main()
