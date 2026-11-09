"""Regenerate the first original batch of each block of DPO rows and compare
with the stored responses. A block = contiguous rows of the checkpoint in
raw_results.csv; batches start at the beginning of each block.

    python scripts/check_batches.py
"""
import pandas as pd

from overrefusal import config, models, prompts

FAMILY, STAGE = "olmo2", "dpo"
g = config.GENERATION

raw = pd.read_csv(config.raw_results_csv(FAMILY))
idx = pd.Series(raw.index[raw.checkpoint == f"{STAGE}__none"])
starts = idx[idx.diff() != 1].tolist()

model, tok = models.load(FAMILY, STAGE)
for s in starts:
    rows = list(range(s, s + g.batch_size))
    texts = [prompts.build_prompt(tok, m, STAGE) for m in raw.prompt[rows]]
    new = models.generate(model, tok, texts, g.max_new_tokens, g.max_prompt_tokens)
    new = [prompts.clean_response(r, STAGE).strip() for r in new]
    old = raw.response[rows].fillna("").str.strip().tolist()
    print(f"block from row {s} ({raw.source[s]}): identical {sum(a == b for a, b in zip(new, old))}/{len(rows)}")
