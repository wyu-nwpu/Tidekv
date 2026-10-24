
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List, Union

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "readkv"))
from layer_budget import (  # noqa: E402
    layer_budgets_from_config,
    load_budget_config,
    pyramid_layer_budgets,
)

MaxCapacityType = Union[int, List[int]]

def pyramidkv_layer_budgets(
    budget: int,
    num_layers: int,
    *,
    source_gqa_ratio: int,
    window_size: int,
    beta: int,
) -> List[int]:
    """Match the compact PyramidKV-A layer schedule, including integer rounding."""
    if source_gqa_ratio < 1:
        raise ValueError("source_gqa_ratio must be >= 1")
    if budget % source_gqa_ratio:
        raise ValueError(
            f"budget={budget} must be divisible by "
            f"source_gqa_ratio={source_gqa_ratio}"
        )
    reference_capacity = budget // source_gqa_ratio
    base_prefix = reference_capacity - int(window_size)
    if base_prefix <= 0:
        raise ValueError(
            f"Pyramid reference capacity={reference_capacity} must exceed "
            f"PyramidKV window_size={window_size}"
        )
    if beta < 1:
        raise ValueError("beta must be >= 1")
    min_prefix = base_prefix // beta
    max_prefix = base_prefix * 2 - min_prefix
    denominator = max(int(num_layers) - 1, 1)
    step = (max_prefix - min_prefix) // denominator
    return [
        (max_prefix - layer_idx * step + int(window_size)) * source_gqa_ratio
        for layer_idx in range(int(num_layers))
    ]

def resolve_tidekv_max_capacity(args: Any, num_layers: int) -> MaxCapacityType:

    config_path = getattr(args, "layer_budget_config", None)
    if config_path:
        cfg = load_budget_config(config_path)
        return layer_budgets_from_config(cfg, num_layers)
    cap = getattr(args, "max_capacity_prompts", None)
    if cap is None:
        raise ValueError(
            "TideKV requires --max_capacity_prompts or --layer_budget_config"
        )
    mode = str(getattr(args, "tidekv_layer_budget", "uniform") or "uniform").lower()
    if mode == "pyramid":
        beta = int(getattr(args, "pyram_beta", 20) or 20)
        source_gqa_ratio = int(
            getattr(args, "tidekv_pyramid_source_gqa_ratio", 1) or 1
        )
        # Keep every layer budget strictly above the obs window so num_keep > 0.
        obs = getattr(args, "obs_window_size", None) or getattr(args, "window_sizes", None) or 32
        if source_gqa_ratio > 1:
            return pyramidkv_layer_budgets(
                int(cap),
                num_layers,
                source_gqa_ratio=source_gqa_ratio,
                window_size=int(obs),
                beta=beta,
            )
        return pyramid_layer_budgets(cap, num_layers, beta=beta, min_floor=int(obs) + 1)
    return cap
