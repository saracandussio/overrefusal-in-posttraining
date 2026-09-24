"""Everything that defines an experiment lives here, and only here.

A model family is chosen with `--family olmo2|olmo3` on every script;
nothing else changes between families.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
STAGES = ["base", "sft", "dpo", "final"]

MODELS: dict[str, dict[str, str]] = {
    "olmo2": {
        "base":  "allenai/OLMo-2-1124-7B",
        "sft":   "allenai/OLMo-2-1124-7B-SFT",
        "dpo":   "allenai/OLMo-2-1124-7B-DPO",
        "final": "allenai/OLMo-2-1124-7B-Instruct",
    },
    "olmo3": {  # mixed-case "Olmo" is the real repo name
        "base":  "allenai/Olmo-3-1025-7B",
        "sft":   "allenai/Olmo-3-7B-Instruct-SFT",
        "dpo":   "allenai/Olmo-3-7B-Instruct-DPO",
        "final": "allenai/Olmo-3-7B-Instruct",
    },
}

# Tags keep the "<stage>__none" form used by every existing result file
# ("none" = no system prompt; the mistral_safety runs are no longer used).
CHECKPOINTS = [f"{s}__none" for s in STAGES]


def stage_of(tag: str) -> str:
    return tag.split("__", 1)[0]


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
# BeaverTails is out: ~7% label noise (Zhu et al. 2024), and a manual check
# found many "harmful" prompts that are benign. It distorted v_ref.
EXCLUDED_SOURCES = {"beavertails"}

ACTIVATIONS_REPO = "saracandu/overrefusal-activations"
ACTIVATIONS_PATH = "data/{family}/{checkpoint}/*.parquet"


def results_dir(family: str) -> Path:
    return Path("results") / family


def raw_results_csv(family: str) -> Path:
    return results_dir(family) / "raw_results.csv"


# --------------------------------------------------------------------------
# Activations
# --------------------------------------------------------------------------
# Percentiles of depth; for 32-layer models this gives [8, 16, 19, 24, 26, 31].
LAYER_PERCENTILES = [25, 50, 60, 75, 80, 100]

# last_prompt    last token of the user's text
# post_instr_k   k-th chat-template token after the user's text. Their number
#                depends on the tokenizer (0 for the base model, which sees the
#                raw text). The last one is where the model decides what to say.
# pre_gen        alias for that last template token (last_prompt for base),
#                handy because the count differs between families
# first_gen      the first *generated* token. Its state already encodes the
#                token the model chose ("I" of "I'm sorry"): it is read after
#                the decision, so use it knowingly.
FIXED_POSITIONS = ["last_prompt", "pre_gen", "first_gen"]

# What each template token is, decoded from the tokenizers. OLMo 2 has no
# special token for <|assistant|>, so it is split into five pieces; the two
# "|" carry almost no information and give unstable axis measures.
# OLMo 3's template also inserts a default system prompt ("You are a
# helpful function-calling AI assistant...") when none is given.
TEMPLATE_TOKENS: dict[str, list[str]] = {
    "olmo2": ["\\n", "<", "|", "assistant", "|", ">", "\\n"],
    "olmo3": ["<|im_end|>", "\\n", "<|im_start|>", "assistant", "\\n"],
}


# The base model has no chat template. It gets a minimal dialogue frame, so
# that it answers instead of continuing the text; its template tokens are
# therefore different from the chat models' (post_instr_k of base and of
# sft are different tokens: only last_prompt, pre_gen, first_gen compare).
BASE_FRAME = "User: {message}\nAssistant:"
BASE_STOP = "\nUser:"          # the base goes on inventing turns; cut there
BASE_TEMPLATE_TOKENS = ["\\n", "Assistant", ":"]


def position_label(family: str, position: str, stage: str = "sft") -> str:
    """Readable name for a position, e.g. post_instr_3 -> 'assistant'."""
    if position.startswith("post_instr_"):
        k = int(position.rsplit("_", 1)[1])
        tokens = BASE_TEMPLATE_TOKENS if stage == "base" else TEMPLATE_TOKENS.get(family, [])
        return f"{position} ({tokens[k]})" if k < len(tokens) else position
    return position


def select_layers(n_layers: int) -> list[int]:
    return sorted({min(round(p / 100 * n_layers), n_layers - 1) for p in LAYER_PERCENTILES})


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Generation:
    max_new_tokens: int = 256
    batch_size: int = 8
    max_prompt_tokens: int = 1024


GENERATION = Generation()

# Prompts were generated with the tokenizer's default special tokens;
# extraction tokenizes the same way so both see the identical context.
ADD_SPECIAL_TOKENS = True
