"""Residual-stream activations: extracting them and reading them back.

On the Hub every row is one (checkpoint, prompt): metadata plus one column
per (layer, position), `layer_{L}_{position}`, with positions last_prompt,
post_instr_0..k (the chat-template tokens) and first_gen. See config.py.

Missing vectors are stored as all zeros (a template token cut off by
truncation, or padding in older shards that reserved extra post_instr
slots). `load` turns them into None, and `matrix` skips them, so a zero
vector never enters a mean.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from overrefusal import config
from overrefusal.groups import assign_group

log = logging.getLogger(__name__)
META = ["prompt", "label", "category", "source", "checkpoint"]
_POST = re.compile(r"post_instr_(\d+)")


def col(layer: int, position: str) -> str:
    return f"layer_{layer}_{position}"


def hf_token() -> str | None:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    path = Path("~/.hf_token").expanduser()
    return path.read_text().strip() if path.exists() else None


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------
def token_positions(tok, prompt_text: str, user_message: str, response: str,
                    max_len: int) -> tuple[list[int], dict[str, int]]:
    """Full token ids (prompt + response) and the index of every position."""
    special = config.ADD_SPECIAL_TOKENS
    prompt_ids = tok(prompt_text, add_special_tokens=special).input_ids
    response_ids = tok(response, add_special_tokens=False).input_ids
    ids = (prompt_ids + response_ids)[:max_len]

    user_end = prompt_text.rfind(user_message)
    if user_end < 0:
        raise ValueError("user message not found in prompt text")
    n_prefix = len(tok(prompt_text[:user_end + len(user_message)],
                       add_special_tokens=special).input_ids)

    positions = {"last_prompt": n_prefix - 1}
    positions |= {f"post_instr_{k}": i for k, i in enumerate(range(n_prefix, len(prompt_ids)))}
    positions["first_gen"] = len(prompt_ids)
    return ids, positions


def extract(model, tok, prompt_text: str, user_message: str, response: str,
            layers: list[int], max_len: int = 2048) -> dict[str, np.ndarray]:
    import torch
    ids, pos = token_positions(tok, prompt_text, user_message, response, max_len)
    with torch.no_grad():
        hidden = model(torch.tensor([ids], device=model.device),
                       output_hidden_states=True, use_cache=False).hidden_states
    # hidden_states[0] is the embedding output, so layer L is index L + 1.
    # Positions cut off by max_len are stored as zeros (read back as missing),
    # which keeps every row of a checkpoint on the same schema.
    d = hidden[0].shape[-1]
    return {col(L, p): (hidden[L + 1][0, i].float().cpu().numpy() if i < len(ids)
                        else np.zeros(d, np.float32))
            for L in layers for p, i in pos.items()}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def _template_columns(columns, layer: int) -> list[str]:
    found = [(int(m.group(1)), c) for c in columns
             if c.startswith(f"layer_{layer}_") and (m := _POST.search(c))]
    return [c for _, c in sorted(found)]


def _blank_zeros(series: pd.Series) -> pd.Series:
    return series.map(lambda v: v if v is not None and np.any(v) else None)


def _pre_gen(df: pd.DataFrame, layer: int) -> pd.Series:
    """Last non-empty template token of each row; last_prompt if there is none."""
    template = _template_columns(df.columns, layer)
    last = df[col(layer, "last_prompt")].copy()
    for c in template:  # later tokens overwrite earlier ones
        last = df[c].where(df[c].notna(), last)
    return last


def load(family: str, layers: list[int], positions: list[str],
         checkpoints: list[str] = config.CHECKPOINTS) -> pd.DataFrame:
    """Activations for the requested cells, with `group`; excluded sources dropped.

    `positions` may name post_instr_k directly, or "post_instr" for all of
    them, or "pre_gen" for the last one.
    """
    from datasets import load_dataset
    parts = []
    for ckpt in checkpoints:
        ds = load_dataset(config.ACTIVATIONS_REPO, split="train", token=hf_token(),
                          data_files={"train": config.ACTIVATIONS_PATH.format(
                              family=family, checkpoint=ckpt)})
        wanted = set(META)
        for L in layers:
            template = _template_columns(ds.column_names, L)
            for p in positions:
                if p in ("post_instr", "pre_gen"):
                    wanted |= set(template)
                if p == "pre_gen":
                    wanted.add(col(L, "last_prompt"))
                if p not in ("post_instr", "pre_gen"):
                    wanted.add(col(L, p))
        df = ds.select_columns([c for c in ds.column_names if c in wanted]).to_pandas()
        df["checkpoint"] = ckpt
        for c in df.columns:
            if c.startswith("layer_"):
                df[c] = _blank_zeros(df[c])
        if "pre_gen" in positions:
            for L in layers:
                df[col(L, "pre_gen")] = _pre_gen(df, L)
        log.info("%s/%s: %d rows", family, ckpt, len(df))
        parts.append(df)

    df = pd.concat(parts, ignore_index=True)
    df = df[~df["source"].isin(config.EXCLUDED_SOURCES)].reset_index(drop=True)
    df["group"] = assign_group(df)
    return df


def reading_order(positions) -> list[str]:
    """last_prompt, post_instr_0, post_instr_1, ..., pre_gen, first_gen."""
    positions = set(positions)
    template = sorted((p for p in positions if _POST.fullmatch(p)),
                      key=lambda p: int(p.rsplit("_", 1)[1]))
    return [p for p in ["last_prompt", *template, "pre_gen", "first_gen"] if p in positions]


def positions_in(df: pd.DataFrame, layer: int) -> list[str]:
    """Positions actually present for a layer, in reading order."""
    prefix = f"layer_{layer}_"
    return reading_order(c.removeprefix(prefix) for c in df.columns if c.startswith(prefix))


def has(df: pd.DataFrame, column: str) -> pd.Series:
    """Rows where `column` holds a real vector."""
    return df[column].notna() if column in df else pd.Series(False, index=df.index)


def matrix(df: pd.DataFrame, column: str) -> np.ndarray:
    """Column of vectors -> (n, d) float32 matrix, skipping missing ones."""
    values = df.loc[has(df, column), column].to_numpy()
    return np.stack(values).astype(np.float32) if len(values) else np.empty((0, 0), np.float32)


def expand(df: pd.DataFrame, layer: int, requested: list[str]) -> list[str]:
    """Requested positions present for this layer; "post_instr" means all of them."""
    present = positions_in(df, layer)
    return [p for p in present
            if p in requested or ("post_instr" in requested and _POST.fullmatch(p))]
