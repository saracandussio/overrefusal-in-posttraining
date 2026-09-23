"""Benchmark prompts as one DataFrame: prompt, label, category, source.

label is the dataset's own gold: 0 = should be answered, 1 = should be refused.
Sampling (seeds, sizes, order) is identical to the original loaders, so the
prompt set matches the one already in results/*/raw_results.csv.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import pandas as pd

log = logging.getLogger(__name__)
SEED = 42


@dataclass(frozen=True)
class Dataset:
    hf_path: str
    split: str
    kind: str                 # "safe" (all 0), "harmful" (all 1), "mixed"
    config: str | None = None
    max_samples: int | None = None


DATASETS: dict[str, Dataset] = {
    # safe prompts that look harmful: the over-refusal probes
    "or_bench":       Dataset("bench-llm/or-bench", "train", "safe", "or-bench-80k", 500),
    "false_reject":   Dataset("AmazonScience/FalseReject", "test", "safe", None, 500),
    "xstest":         Dataset("walledai/XSTest", "test", "safe"),  # gated
    # ordinary instructions
    "alpaca":         Dataset("tatsu-lab/alpaca", "train", "safe", None, 1000),
    # harmful
    "harmbench":      Dataset("allenai/tulu-3-harmbench-eval", "test", "harmful"),
    "advbench":       Dataset("walledai/AdvBench", "train", "harmful"),  # gated
    "jailbreakbench": Dataset("JailbreakBench/JBB-Behaviors", "harmful", "harmful", "behaviors"),
    # both
    "wildguard":      Dataset("allenai/wildguardmix", "test", "mixed", "wildguardtest", 1000),
    "toxicchat":      Dataset("lmsys/toxic-chat", "test", "mixed", "toxicchat0124", 500),
}


def _hf(d: Dataset) -> pd.DataFrame:
    from datasets import load_dataset
    args = (d.hf_path, d.config) if d.config else (d.hf_path,)
    return load_dataset(*args, split=d.split).to_pandas()


def _first(df: pd.DataFrame, names: list[str]) -> str | None:
    return next((c for c in names if c in df.columns), None)


def _frame(prompts: pd.Series, labels, categories, source: str) -> pd.DataFrame:
    return pd.DataFrame({
        "prompt": prompts.values,
        "label": labels if not hasattr(labels, "values") else labels.values,
        "category": categories if not hasattr(categories, "values") else categories.values,
        "source": source,
    })


def _balanced(df: pd.DataFrame, n: int | None) -> pd.DataFrame:
    """Half safe, half harmful, shuffled."""
    if not n:
        return df.reset_index(drop=True)
    parts = [df[df.label == y].sample(n=min(n // 2, (df.label == y).sum()), random_state=SEED)
             for y in (0, 1)]
    return pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)


def _simple(name: str, prompt_cols: list[str], cat_cols: list[str]) -> Callable:
    def load(d: Dataset) -> pd.DataFrame:
        raw = _hf(d)
        cat = _first(raw, cat_cols)
        out = _frame(raw[_first(raw, prompt_cols)],
                     0 if d.kind == "safe" else 1,
                     raw[cat] if cat else "unknown", name)
        if d.max_samples:
            out = out.sample(n=min(d.max_samples, len(out)), random_state=SEED)
        return out.reset_index(drop=True)
    return load


def _alpaca(d: Dataset) -> pd.DataFrame:
    raw = _hf(d)
    has_input = raw["input"].fillna("").str.strip() != ""
    prompt = raw["instruction"].where(~has_input, raw["instruction"] + "\n" + raw["input"])
    out = _frame(prompt, 0, "unknown", "alpaca")
    return out.sample(n=min(d.max_samples, len(out)), random_state=SEED).reset_index(drop=True)


def _wildguard(d: Dataset) -> pd.DataFrame:
    raw = _hf(d)
    cat = _first(raw, ["category", "subcategory"])
    label = (raw["prompt_harm_label"].str.lower() == "harmful").astype(int)
    return _balanced(_frame(raw["prompt"], label, raw[cat] if cat else "unknown", "wildguard"),
                     d.max_samples)


def _toxicchat(d: Dataset) -> pd.DataFrame:
    raw = _hf(d)
    cat = raw["jailbreaking"].astype(int).map({1: "jailbreak", 0: "direct"})
    return _balanced(_frame(raw["user_input"], raw["toxicity"].astype(int), cat, "toxicchat"),
                     d.max_samples)


LOADERS: dict[str, Callable[[Dataset], pd.DataFrame]] = {
    "or_bench":       _simple("or_bench", ["prompt", "text", "question"], ["category", "type"]),
    "false_reject":   _simple("false_reject", ["prompt", "text", "question"],
                              ["category", "domain", "type"]),
    # xstest's "type" column carries the contrast_* marker used by groups.py
    "xstest":         _simple("xstest", ["prompt", "text"], ["type", "category", "note"]),
    "alpaca":         _alpaca,
    "harmbench":      _simple("harmbench", ["prompt", "Behavior", "behavior", "goal"],
                              ["category", "SemanticCategory", "FunctionalCategory"]),
    "advbench":       _simple("advbench", ["prompt", "goal"], ["category"]),
    "jailbreakbench": _simple("jailbreakbench", ["Goal", "goal", "prompt"], ["Category", "category"]),
    "wildguard":      _wildguard,
    "toxicchat":      _toxicchat,
}


def load(names: list[str] | None = None) -> pd.DataFrame:
    names = names or list(DATASETS)
    frames = []
    for name in names:
        df = LOADERS[name](DATASETS[name])
        log.info("%-15s %5d prompts (%d harmful)", name, len(df), int(df.label.sum()))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
