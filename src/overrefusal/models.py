"""Loading checkpoints and generating with them (prompts are in prompts.py)."""
from __future__ import annotations

import gc
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from overrefusal import config

log = logging.getLogger(__name__)


def load(family: str, stage: str):
    name = config.MODELS[family][stage]
    log.info("loading %s", name)
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"  # decoder-only batched generation
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float16, device_map="auto").eval()
    return model, tok


def unload(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.inference_mode()
def generate(model, tok, texts: list[str], max_new_tokens: int, max_prompt_tokens: int) -> list[str]:
    batch = tok(texts, return_tensors="pt", padding=True, truncation=True,
                max_length=max_prompt_tokens,
                add_special_tokens=config.ADD_SPECIAL_TOKENS).to(model.device)
    batch.pop("token_type_ids", None)
    out = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    return tok.batch_decode(out[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
