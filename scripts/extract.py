"""Step 3 — extract activations (last_prompt, every template token, first_gen)
and push them to the Hub, one folder per checkpoint.

Writes to <repo>/data/<family>/<checkpoint>/ (repo defaults to
config.ACTIVATIONS_REPO). It refuses to write into a folder that already has
shards, so two extractions are never mixed. --replace deletes the old shards
of that folder in the same commit that adds the new ones; the Hub keeps the
previous revision, so nothing is lost for good.

    python scripts/extract.py --family olmo2 --checkpoints base__none --replace
    python scripts/extract.py --family olmo2 --repo saracandu/overrefusal-activations-v2
"""
import tempfile

import pandas as pd
from datasets import Dataset
from huggingface_hub import HfApi

from overrefusal import activations, cli, config, models, prompts

SHARD_ROWS = 500


def main():
    p = cli.parser(__doc__)
    p.add_argument("--repo", default=config.ACTIVATIONS_REPO)
    p.add_argument("--replace", action="store_true",
                   help="swap out existing shards of the selected checkpoints")
    args = p.parse_args()

    api = HfApi(token=activations.hf_token())
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    existing = set(api.list_repo_files(args.repo, repo_type="dataset"))
    raw = pd.read_csv(config.raw_results_csv(args.family))
    raw = raw[~raw.source.isin(config.EXCLUDED_SOURCES)]

    for ckpt in args.checkpoints:
        folder = config.ACTIVATIONS_PATH.format(family=args.family, checkpoint=ckpt).rsplit("/", 1)[0]
        if any(f.startswith(folder + "/") for f in existing) and not args.replace:
            raise SystemExit(f"{args.repo}/{folder} already has data; "
                             "use --replace or a new --repo")

        stage = config.stage_of(ckpt)
        rows = raw[raw.checkpoint == ckpt]
        model, tok = models.load(args.family, stage)
        layers = config.select_layers(model.config.num_hidden_layers)

        with tempfile.TemporaryDirectory() as tmp:
            buffer, shard = [], 0
            for n, r in enumerate(rows.itertuples(), 1):
                text = prompts.build_prompt(tok, r.prompt, stage)
                acts = activations.extract(model, tok, text, r.prompt, str(r.response), layers)
                meta = {c: getattr(r, c) for c in activations.META}
                buffer.append(meta | {k: v.tolist() for k, v in acts.items()})
                if len(buffer) == SHARD_ROWS or n == len(rows):
                    Dataset.from_list(buffer).to_parquet(f"{tmp}/shard_{shard:05d}.parquet")
                    buffer, shard = [], shard + 1
            api.upload_folder(repo_id=args.repo, repo_type="dataset",
                              folder_path=tmp, path_in_repo=folder,
                              delete_patterns="*.parquet" if args.replace else None,
                              commit_message=f"{'re-extract' if args.replace else 'extract'} "
                                             f"{args.family}/{ckpt}")
        models.unload(model)
        print(f"{ckpt}: {len(rows)} rows, layers {layers}")


if __name__ == "__main__":
    main()
