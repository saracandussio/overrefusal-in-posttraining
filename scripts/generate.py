"""Step 1 — generate a response for every prompt, per checkpoint.

Appends to results/<family>/raw_results.csv; rows already there
(same checkpoint, source, prompt) are skipped, so the run can be resumed.

    python scripts/generate.py --family olmo2
    python scripts/generate.py --family olmo2 --checkpoints sft__none --datasets xstest
"""
import pandas as pd

from overrefusal import cli, config, datasets, models
from overrefusal.io import save_atomic
from overrefusal.refusal import KEY, keyword_refusal


def main():
    p = cli.parser(__doc__)
    p.add_argument("--datasets", nargs="+", default=None, choices=list(datasets.DATASETS))
    args = p.parse_args()

    out = config.raw_results_csv(args.family)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = pd.read_csv(out, usecols=KEY) if out.exists() else pd.DataFrame(columns=KEY)
    prompts = datasets.load(args.datasets)

    for ckpt in args.checkpoints:
        stage = config.stage_of(ckpt)
        todo = prompts.merge(done[done.checkpoint == ckpt], on=["source", "prompt"],
                             how="left", indicator=True)
        todo = todo[todo._merge == "left_only"].drop(columns=["_merge", "checkpoint"])
        if todo.empty:
            continue

        model, tok = models.load(args.family, stage)
        texts = [models.build_prompt(tok, p, stage) for p in todo.prompt]
        g = config.GENERATION
        responses = []
        for i in range(0, len(texts), g.batch_size):
            responses += models.generate(model, tok, texts[i:i + g.batch_size],
                                         g.max_new_tokens, g.max_prompt_tokens)
        models.unload(model)

        rows = todo.assign(checkpoint=ckpt, response=responses)
        rows["predicted_refusal"] = keyword_refusal(rows.response)
        # rewrite rather than append: the file may already carry judge columns
        old = pd.read_csv(out) if out.exists() else None
        save_atomic(pd.concat([old, rows], ignore_index=True), out)
        done = pd.concat([done, rows[KEY]], ignore_index=True)


if __name__ == "__main__":
    main()
