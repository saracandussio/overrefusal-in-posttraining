"""
dataset_config.py
Single source of truth for all benchmark datasets.
Imported by every model config (config.py, config_olmo2.py, …).

NOTE: this file must be named `dataset_config.py` (singular) on disk,
since dataset_loader.py does `from dataset_config import DatasetConfig`
and `from dataset_config import ALL_DATASETS`.

Dataset types
-------------
  "over_refusal"  — all prompts are SAFE (label=0). Measures FP (over-refusal).
                    A good model should answer all of them.
  "harmful"       — all prompts are HARMFUL (label=1). Measures FN (under-refusal).
                    A good model should refuse all of them.
  "mixed"         — contains both safe (0) and harmful (1) prompts.
                    Measures both FP and FN simultaneously.
"""
from dataclasses import dataclass
from typing import Literal, Optional

DatasetType = Literal["over_refusal", "harmful", "mixed"]


@dataclass
class DatasetConfig:
    name: str
    dataset_type: DatasetType       # drives which metrics are computed
    hf_path: str                    # HuggingFace dataset path
    hf_split: str = "test"
    prompt_column: str = "prompt"
    label_column: Optional[str] = None   # None → use dataset_type to assign fixed label
    max_samples: Optional[int] = None    # None = use all available


# ---------------------------------------------------------------------------
# Over-refusal benchmarks  (label = 0, FP measurement)
# ---------------------------------------------------------------------------

OR_BENCH = DatasetConfig(
    name="or_bench",
    dataset_type="over_refusal",
    hf_path="bench-llm/or-bench",
    hf_split="train",
    prompt_column="prompt",
    max_samples=500,
)

FALSE_REJECT = DatasetConfig(
    name="false_reject",
    dataset_type="over_refusal",
    hf_path="AmazonScience/FalseReject",
    hf_split="test",
    prompt_column="prompt",
    max_samples=500,
)

XSTEST = DatasetConfig(
    name="xstest",
    dataset_type="over_refusal",
    hf_path="walledai/XSTest",
    hf_split="test",
    prompt_column="prompt",
    # gated — richiede HF token e accettazione condizioni su HF
)

ALPACA = DatasetConfig(
    name="alpaca",
    dataset_type="over_refusal",
    # Usato come baseline harmless genuino (istruzioni ordinarie).
    # Il loader concatena instruction + input quando input non è vuoto.
    hf_path="tatsu-lab/alpaca",
    hf_split="train",
    prompt_column="instruction",   # il loader gestisce input opzionale
    max_samples=1000,
)


# ---------------------------------------------------------------------------
# Harmful benchmarks  (label = 1, FN measurement)
# ---------------------------------------------------------------------------

HARMBENCH = DatasetConfig(
    name="harmbench",
    dataset_type="harmful",
    hf_path="allenai/tulu-3-harmbench-eval",
    hf_split="test",
    prompt_column="prompt",
)

ADVBENCH = DatasetConfig(
    name="advbench",
    dataset_type="harmful",
    hf_path="walledai/AdvBench",
    hf_split="train",
    prompt_column="prompt",
    # gated — richiede HF token e accettazione condizioni su HF
    # 500 esempi, jailbreak diretto — il riferimento più citato in letteratura
)

JAILBREAKBENCH = DatasetConfig(
    name="jailbreakbench",
    dataset_type="harmful",
    hf_path="JailbreakBench/JBB-Behaviors",
    hf_split="harmful",   # split corretto — non "train"
    prompt_column="Goal",
)


# ---------------------------------------------------------------------------
# Mixed benchmarks  (label = 0 and 1, FP + FN measurement)
# ---------------------------------------------------------------------------

WILDGUARD = DatasetConfig(
    name="wildguard",
    dataset_type="mixed",
    hf_path="allenai/wildguardmix",
    hf_split="test",
    label_column="prompt_harm_label",
    max_samples=1000,
)

TOXICCHAT = DatasetConfig(
    name="toxicchat",
    dataset_type="mixed",
    hf_path="lmsys/toxic-chat",
    hf_split="test",
    prompt_column="user_input",
    label_column="toxicity",
    max_samples=500,
)

BEAVERTAILS = DatasetConfig(
    name="beavertails",
    dataset_type="mixed",
    hf_path="PKU-Alignment/BeaverTails",
    hf_split="30k_test",
    label_column="is_safe",
    max_samples=1000,
)


# ---------------------------------------------------------------------------
# Canonical registry
# ---------------------------------------------------------------------------

ALL_DATASETS: dict[str, DatasetConfig] = {
    "or_bench":       OR_BENCH,
    "false_reject":   FALSE_REJECT,
    "xstest":         XSTEST,
    "alpaca":         ALPACA,
    "harmbench":      HARMBENCH,
    "advbench":       ADVBENCH,
    "jailbreakbench": JAILBREAKBENCH,
    "wildguard":      WILDGUARD,
    "toxicchat":      TOXICCHAT,
    # "beavertails":  RIMOSSO -- 6.96% label error rate (Zhu et al. 2024)
}

OVER_REFUSAL_DATASETS: list[str] = ["or_bench", "false_reject", "xstest", "alpaca"]
HARMFUL_DATASETS:      list[str] = ["harmbench", "advbench", "jailbreakbench"]
MIXED_DATASETS:        list[str] = ["wildguard", "toxicchat"]