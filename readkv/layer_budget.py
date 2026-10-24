"""Uniform and pyramidal layer budgets used during inference."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence, Union

MaxCapacitySpec = Union[int, Sequence[int]]

def get_layer_budget(max_capacity: MaxCapacitySpec, layer_idx: int) -> int:
    if isinstance(max_capacity, int):
        return max_capacity
    if isinstance(max_capacity, (list, tuple)):
        return int(max_capacity[layer_idx])
    raise TypeError(f"max_capacity must be int or list, got {type(max_capacity)}")

def infer_base_budget(layer_budgets: Sequence[int]) -> int:

    budgets = [int(x) for x in layer_budgets]
    if not budgets:
        raise ValueError("empty layer_budgets")
    return sum(budgets) // len(budgets)

def resolve_prefill_max_capacity(
    seq_len: int,
    max_capacity: MaxCapacitySpec,
    *,
    base_budget: int | None = None,
) -> MaxCapacitySpec:

    if isinstance(max_capacity, int):
        return max_capacity
    budgets = [int(x) for x in max_capacity]
    k_base = base_budget if base_budget is not None else infer_base_budget(budgets)
    if seq_len <= k_base:
        return k_base
    return budgets

def effective_layer_budget(layer_budget: int, seq_len: int) -> int:

    return min(int(layer_budget), int(seq_len))

def pyramid_layer_budgets(
    base_budget: int,
    num_layers: int,
    beta: int = 20,
    min_floor: int = 1,
) -> List[int]:
    """Create a linear layer schedule and correct its rounding drift."""
    base = max(1, int(base_budget))
    n = int(num_layers)
    floor = max(1, int(min_floor))
    if n <= 1:
        return [max(floor, base)]
    min_b = max(floor, base // max(1, int(beta)))
    max_b = 2 * base - min_b
    step = (max_b - min_b) / (n - 1)
    budgets = [max(floor, int(round(max_b - i * step))) for i in range(n)]
    drift = n * base - sum(budgets)            # restore exact total after rounding
    budgets[n // 2] = max(floor, budgets[n // 2] + drift)
    return budgets

def normalize_max_capacity(max_capacity: MaxCapacitySpec, num_layers: int) -> List[int]:
    if isinstance(max_capacity, int):
        return [max_capacity] * num_layers
    budgets = [int(x) for x in max_capacity]
    if len(budgets) != num_layers:
        raise ValueError(
            f"layer_budgets length {len(budgets)} != num_layers {num_layers}"
        )
    return budgets

def load_budget_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def layer_budgets_from_config(config: dict, num_layers: int) -> List[int]:
    budgets = config.get("layer_budgets")
    if budgets is None:
        base = int(config["base_budget"])
        return [base] * num_layers
    return normalize_max_capacity(budgets, num_layers)
