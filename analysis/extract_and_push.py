"""
extract_and_push.py  (v3 — olmo2 + olmo3, no mistral_safety)

Extract residual-stream activations from OLMo-2 and OLMo-3 checkpoints at
selected layers and push them incrementally to a HuggingFace dataset.

Usage
-----
    # OLMo-2
    python extract_and_push.py \
        --csv results/olmo2/raw_results.csv \
        --model-family olmo2 \
        --hf-repo saracandu/olmo-activations \
        --hf-token $HF_TOKEN \
        --batch-size 32 --device cuda

    # OLMo-3
    python extract_and_push.py \
        --csv results/olmo3/raw_results.csv \
        --model-family olmo3 \
        --hf-repo saracandu/olmo-activations \
        --hf-token $HF_TOKEN \
        --batch-size 32 --device cuda

    # Post-instruction columns only (non-destructive)
    python extract_and_push.py \
        --csv results/olmo2/raw_results.csv \
        --model-family olmo2 \
        --hf-repo saracandu/olmo-activations \
        --hf-token $HF_TOKEN \
        --post-instr-only
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import re
import shutil
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, Features, Sequence, Value
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

def upload_folder_with_retry(
    api: HfApi,
    max_retries: int = 8,
    extra_buffer: int = 30,
    **kwargs,
) -> None:
    backoff = 60
    for attempt in range(1, max_retries + 1):
        try:
            api.upload_folder(**kwargs)
            return
        except HfHubHTTPError as e:
            if e.response.status_code != 429 or attempt == max_retries:
                raise
            match = re.search(r"[Rr]etry after (\d+) seconds", str(e))
            wait = int(match.group(1)) + extra_buffer if match else backoff
            log.warning(
                "429 rate-limit on attempt %d/%d — waiting %ds ...",
                attempt, max_retries, wait,
            )
            time.sleep(wait)
            backoff = min(backoff * 2, 3600)


# ---------------------------------------------------------------------------
# Checkpoint → HF model id
# ---------------------------------------------------------------------------

CHECKPOINT_TO_HF: dict[str, dict[str, str]] = {
    "olmo2": {
        "base__none":  "allenai/OLMo-2-1124-7B",
        "sft__none":   "allenai/OLMo-2-1124-7B-SFT",
        "dpo__none":   "allenai/OLMo-2-1124-7B-DPO",
        "final__none": "allenai/OLMo-2-1124-7B-Instruct",
    },
    "olmo3": {
        "base__none":  "allenai/Olmo-3-1025-7B",
        "sft__none":   "allenai/Olmo-3-7B-Instruct-SFT",
        "dpo__none":   "allenai/Olmo-3-7B-Instruct-DPO",
        "final__none": "allenai/Olmo-3-7B-Instruct",
    },
}

# Only none — mistral_safety dropped entirely
SYSTEM_PROMPTS: dict[str, str | None] = {
    "none": None,
}


# ---------------------------------------------------------------------------
# Layer selection
# ---------------------------------------------------------------------------

def select_layers(num_layers: int) -> list[int]:
    percentiles = [25, 50, 60, 75, 80, 100]
    layers = sorted({
        min(int(round(p / 100 * num_layers)), num_layers - 1)
        for p in percentiles
    })
    log.info("Selected layers: %s (out of %d)", layers, num_layers)
    return layers


# ---------------------------------------------------------------------------
# Prompt building — mirrors olmo_loader._build_prompt exactly
# ---------------------------------------------------------------------------

def build_prompt_string(
    user_message: str,
    tokenizer,
    system_prompt: str | None = None,
) -> str:
    """
    Replicate olmo_loader.CheckpointModel._build_prompt exactly.

    For instruct models (chat_template present):
        apply_chat_template with optional system prompt.
    For base model (no chat_template):
        raw pass-through — no framing added.
    """
    if getattr(tokenizer, "chat_template", None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass

    # Base model — raw pass-through (matches updated olmo_loader._build_prompt)
    return user_message


# ---------------------------------------------------------------------------
# Tokenisation and index detection
# ---------------------------------------------------------------------------

def build_full_ids(
    prompt: str,
    response: str,
    tokenizer,
    system_prompt: str | None = None,
    max_length: int = 2048,
    device: str = "cpu",
) -> tuple[torch.Tensor, int, int, list[int]]:
    """
    Build the full token sequence replicating the generation context,
    and identify the positions of interest.

    Returns
    -------
    full_ids : Tensor (1, T)
    last_prompt_idx : int
    first_gen_idx : int
    post_instr_indices : list[int]
    """
    prompt_string = build_prompt_string(
        user_message=prompt,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
    )

    prompt_ids   = tokenizer.encode(prompt_string, add_special_tokens=False)
    response_ids = tokenizer.encode(response,      add_special_tokens=False)

    full = prompt_ids + response_ids
    if len(full) > max_length:
        full = full[:max_length]
    T = len(full)

    if prompt in prompt_string:
        user_end_char = prompt_string.index(prompt) + len(prompt)
        prefix_ids    = tokenizer.encode(prompt_string[:user_end_char], add_special_tokens=False)
        last_prompt_idx = len(prefix_ids) - 1
        post_start      = len(prefix_ids)
    else:
        log.warning("Could not locate user text in prompt_string, using fallback.")
        last_prompt_idx = len(prompt_ids) - 1
        post_start      = len(prompt_ids)

    post_instr_indices = [i for i in range(post_start, len(prompt_ids)) if i < T]
    last_prompt_idx    = min(last_prompt_idx, T - 1)
    first_gen_idx      = min(len(prompt_ids), T - 1)

    return (
        torch.tensor([full], dtype=torch.long, device=device),
        last_prompt_idx,
        first_gen_idx,
        post_instr_indices,
    )


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_activations(
    full_ids: torch.Tensor,
    last_prompt_idx: int,
    first_gen_idx: int,
    post_instr_indices: list[int],
    n_post_instr: int,
    model,
    layers: list[int],
    post_instr_only: bool = False,
) -> dict[int, dict[str, np.ndarray]]:
    out = model(full_ids, output_hidden_states=True, use_cache=False)

    result: dict[int, dict[str, np.ndarray]] = {}
    for l in layers:
        hs    = out.hidden_states[l + 1]
        d     = hs.shape[-1]
        entry: dict[str, np.ndarray] = {}

        if not post_instr_only:
            entry["last_prompt"] = (
                hs[0, last_prompt_idx, :].float().cpu().numpy().astype(np.float16)
            )
            entry["first_gen"] = (
                hs[0, first_gen_idx, :].float().cpu().numpy().astype(np.float16)
            )

        for k in range(n_post_instr):
            if k < len(post_instr_indices):
                entry[f"post_instr_{k}"] = (
                    hs[0, post_instr_indices[k], :]
                    .float().cpu().numpy().astype(np.float16)
                )
            else:
                entry[f"post_instr_{k}"] = np.zeros(d, dtype=np.float16)

        result[l] = entry

    del out
    return result


# ---------------------------------------------------------------------------
# HuggingFace helpers
# ---------------------------------------------------------------------------

def build_features(
    layers: list[int],
    d_model: int,
    n_post_instr: int,
    post_instr_only: bool = False,
) -> Features:
    feats: dict = {
        "prompt":            Value("string"),
        "label":             Value("int32"),
        "category":          Value("string"),
        "source":            Value("string"),
        "checkpoint":        Value("string"),
        "response":          Value("string"),
        "predicted_refusal": Value("int32"),
    }
    for l in layers:
        if not post_instr_only:
            feats[f"layer_{l}_last_prompt"] = Sequence(Value("float32"), length=d_model)
            feats[f"layer_{l}_first_gen"]   = Sequence(Value("float32"), length=d_model)
        for k in range(n_post_instr):
            feats[f"layer_{l}_post_instr_{k}"] = Sequence(Value("float32"), length=d_model)
    return Features(feats)


def rows_to_hf_batch(
    rows: list[dict],
    acts_batch: list[dict[int, dict[str, np.ndarray]]],
    layers: list[int],
    n_post_instr: int,
    post_instr_only: bool = False,
) -> dict[str, list]:
    out: dict[str, list] = defaultdict(list)
    for row, acts in zip(rows, acts_batch):
        out["prompt"].append(row["prompt"])
        out["label"].append(int(row["label"]))
        out["category"].append(row["category"])
        out["source"].append(row["source"])
        out["checkpoint"].append(row["checkpoint"])
        out["response"].append(row["response"])
        out["predicted_refusal"].append(int(row["predicted_refusal"]))
        for l in layers:
            if not post_instr_only:
                out[f"layer_{l}_last_prompt"].append(
                    acts[l]["last_prompt"].astype(np.float32).tolist()
                )
                out[f"layer_{l}_first_gen"].append(
                    acts[l]["first_gen"].astype(np.float32).tolist()
                )
            for k in range(n_post_instr):
                out[f"layer_{l}_post_instr_{k}"].append(
                    acts[l][f"post_instr_{k}"].astype(np.float32).tolist()
                )
    return dict(out)


# ---------------------------------------------------------------------------
# Post-instr count detection
# ---------------------------------------------------------------------------

def detect_n_post_instr(
    tokenizer,
    system_prompt: str | None = None,
    sample_prompt: str = "Hello",
) -> int:
    prompt_string = build_prompt_string(sample_prompt, tokenizer, system_prompt)
    prompt_ids    = tokenizer.encode(prompt_string, add_special_tokens=False)

    if sample_prompt in prompt_string:
        user_end_char = prompt_string.index(sample_prompt) + len(sample_prompt)
        prefix_ids    = tokenizer.encode(prompt_string[:user_end_char], add_special_tokens=False)
        n = len(prompt_ids) - len(prefix_ids)
        log.info("Detected %d post-instruction token(s).", n)
        return n

    log.warning("Could not detect post-instruction count, defaulting to 0.")
    return 0


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def iter_csv_by_checkpoint(csv_path: str) -> Iterator[tuple[str, pd.DataFrame]]:
    df = pd.read_csv(csv_path)
    for ckpt, group in df.groupby("checkpoint"):
        yield str(ckpt), group.reset_index(drop=True)


def _parse_checkpoint_name(checkpoint_name: str) -> str:
    parts = checkpoint_name.split("__", 1)
    return parts[1] if len(parts) == 2 else "none"


def process_checkpoint(
    checkpoint_name: str,
    df: pd.DataFrame,
    model_family: str,
    hf_repo: str,
    hf_token: str,
    batch_size: int,
    device: str,
    max_length: int,
    first_checkpoint: bool,
    post_instr_only: bool = False,
) -> None:
    hf_model_id = CHECKPOINT_TO_HF[model_family].get(checkpoint_name)
    if hf_model_id is None:
        log.warning(
            "No HF model id for '%s' in family '%s', skipping.",
            checkpoint_name, model_family,
        )
        return

    sp_key        = _parse_checkpoint_name(checkpoint_name)
    system_prompt = SYSTEM_PROMPTS.get(sp_key)
    log.info(
        "Checkpoint '%s' → model %s, system_prompt key '%s'",
        checkpoint_name, hf_model_id, sp_key,
    )

    log.info("Loading model %s ...", hf_model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_id, token=hf_token, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_id,
        torch_dtype=torch.float16,
        device_map=device,
        token=hf_token,
        trust_remote_code=True,
    )
    model.eval()

    num_layers   = model.config.num_hidden_layers
    d_model      = model.config.hidden_size
    layers       = select_layers(num_layers)
    n_post_instr = detect_n_post_instr(tokenizer, system_prompt)
    features     = build_features(layers, d_model, n_post_instr, post_instr_only)

    api = HfApi(token=hf_token)
    if first_checkpoint:
        api.create_repo(
            repo_id=hf_repo, repo_type="dataset", exist_ok=True, private=True,
        )

    log.info(
        "Processing '%s' — %d rows, layers %s, %d post-instr tokens%s",
        checkpoint_name, len(df), layers, n_post_instr,
        " [post_instr only]" if post_instr_only else "",
    )

    staging_dir = Path(tempfile.mkdtemp(prefix=f"act_{checkpoint_name}_"))
    rows_buf: list[dict] = []
    acts_buf: list[dict[int, dict[str, np.ndarray]]] = []
    shard_count = 0

    def write_shard() -> None:
        nonlocal shard_count
        if not rows_buf:
            return
        batch_dict = rows_to_hf_batch(
            rows_buf, acts_buf, layers, n_post_instr, post_instr_only
        )
        ds         = Dataset.from_dict(batch_dict, features=features)
        shard_path = staging_dir / f"shard_{shard_count:05d}.parquet"
        ds.to_parquet(str(shard_path))
        log.info("  staged shard %d (%d rows)", shard_count, len(rows_buf))
        shard_count += 1
        rows_buf.clear()
        acts_buf.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for i, row in df.iterrows():
        try:
            full_ids, last_prompt_idx, first_gen_idx, post_instr_indices = (
                build_full_ids(
                    str(row["prompt"]),
                    str(row["response"]),
                    tokenizer,
                    system_prompt=system_prompt,
                    max_length=max_length,
                    device=device,
                )
            )
            acts = extract_activations(
                full_ids,
                last_prompt_idx,
                first_gen_idx,
                post_instr_indices,
                n_post_instr,
                model,
                layers,
                post_instr_only=post_instr_only,
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower():
                log.warning("OOM at row %d, skipping.", i)
                gc.collect()
                torch.cuda.empty_cache()
                continue
            raise

        rows_buf.append(row.to_dict())
        acts_buf.append(acts)
        del full_ids, acts

        if len(rows_buf) >= batch_size:
            write_shard()

    write_shard()

    path_in_repo = (
        f"data/{model_family}/{checkpoint_name}_post_instr"
        if post_instr_only
        else f"data/{model_family}/{checkpoint_name}"
    )
    log.info("Uploading %d shards → %s ...", shard_count, path_in_repo)
    upload_folder_with_retry(
        api,
        folder_path=str(staging_dir),
        path_in_repo=path_in_repo,
        repo_id=hf_repo,
        repo_type="dataset",
        commit_message=(
            f"Add post_instr activations for {model_family}/{checkpoint_name}"
            if post_instr_only
            else f"Add all activations for {model_family}/{checkpoint_name} (v3)"
        ),
    )
    log.info("Upload complete for '%s'.", checkpoint_name)

    shutil.rmtree(staging_dir, ignore_errors=True)
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract OLMo activations → HF dataset (v3 — olmo2 + olmo3)."
    )
    p.add_argument("--csv",              required=True)
    p.add_argument("--model-family",     required=True, choices=["olmo2", "olmo3"])
    p.add_argument("--hf-repo",          required=True)
    p.add_argument("--hf-token",         default=os.environ.get("HF_TOKEN", ""))
    p.add_argument("--batch-size",       type=int, default=32)
    p.add_argument("--device",           default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-length",       type=int, default=2048)
    p.add_argument("--checkpoint-filter", default=None)
    p.add_argument("--dataset-filter",   default=None,
                   help="Filter by source dataset name (e.g. xstest, advbench).")
    p.add_argument(
        "--post-instr-only",
        action="store_true",
        dest="post_instr_only",
        help="Extract only post_instr_* columns to a separate folder.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.hf_token:
        raise ValueError("Provide --hf-token or set HF_TOKEN.")

    first = True
    for checkpoint_name, df in iter_csv_by_checkpoint(args.csv):
        if args.checkpoint_filter and checkpoint_name != args.checkpoint_filter:
            continue
        # Skip mistral_safety entirely
        if checkpoint_name.endswith("__mistral_safety"):
            log.info("Skipping mistral_safety checkpoint: %s", checkpoint_name)
            continue
        # Filter by dataset if requested
        if args.dataset_filter:
            df = df[df["source"] == args.dataset_filter].reset_index(drop=True)
            if len(df) == 0:
                log.warning(
                    "No rows for checkpoint '%s' and dataset '%s', skipping.",
                    checkpoint_name, args.dataset_filter,
                )
                continue
        process_checkpoint(
            checkpoint_name=checkpoint_name,
            df=df,
            model_family=args.model_family,
            hf_repo=args.hf_repo,
            hf_token=args.hf_token,
            batch_size=args.batch_size,
            device=args.device,
            max_length=args.max_length,
            first_checkpoint=first,
            post_instr_only=args.post_instr_only,
        )
        first = False

    log.info("All done. Dataset: https://huggingface.co/datasets/%s", args.hf_repo)


if __name__ == "__main__":
    main()