"""Arguments every script shares, so they read the same everywhere."""
from __future__ import annotations

import argparse
import logging

from overrefusal import config

DEFAULT_LAYERS = [8, 16, 19, 24, 26, 31]


def parser(description: str, activations: bool = False) -> argparse.ArgumentParser:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--family", required=True, choices=list(config.MODELS))
    p.add_argument("--checkpoints", nargs="+", default=config.CHECKPOINTS)
    if activations:
        p.add_argument("--layers", nargs="+", type=int, default=DEFAULT_LAYERS)
        p.add_argument("--positions", nargs="+",
                       default=["last_prompt", "post_instr", "first_gen"],
                       help="last_prompt, post_instr (all template tokens), post_instr_K, "
                            "pre_gen (last template token), first_gen")
    return p
