"""Validation of TideKV model, cache, and selector parameters."""

from __future__ import annotations

from collections.abc import Sequence

def validate_tidekv_parameters(
    *,
    model,
    batch_size: int,
    seq_len: int | None = None,
    num_layers: int,
    max_capacity,
    chunk_size: int,
    score_block_size: int,
    obs_window_size: int,
    kernel_size: int,
    pooling: str,
    warmup_layers: int,
    tidekv_head_adaptive: bool,
    tidekv_head_floor_alpha: float,
    selection_mode: str = "snapkv",
    chunk_length: int = 20,
    attention_mask=None,
) -> None:
    """Fail before model computation when a TideKV configuration is invalid."""

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if score_block_size <= 0:
        raise ValueError(f"score_block_size must be > 0, got {score_block_size}")
    if obs_window_size <= 0:
        raise ValueError(f"obs_window_size must be > 0, got {obs_window_size}")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
    if pooling not in {"avgpool", "maxpool"}:
        raise ValueError(f"pooling must be one of {{'avgpool', 'maxpool'}}, got {pooling!r}")
    selection_mode = str(selection_mode).lower()
    if selection_mode not in {"snapkv", "chunkkv"}:
        raise ValueError(
            "selection_mode must be one of {'snapkv', 'chunkkv'}, "
            f"got {selection_mode!r}"
        )
    if chunk_length <= 0:
        raise ValueError(f"chunk_length must be > 0, got {chunk_length}")
    if tidekv_head_adaptive and selection_mode != "snapkv":
        raise ValueError("tidekv_head_adaptive currently requires selection_mode='snapkv'")
    if not 0 <= warmup_layers <= num_layers:
        raise ValueError(
            f"warmup_layers must be in [0, {num_layers}], got {warmup_layers}"
        )
    if not 0.0 <= float(tidekv_head_floor_alpha) <= 1.0:
        raise ValueError(
            "tidekv_head_floor_alpha must be in [0, 1], "
            f"got {tidekv_head_floor_alpha}"
        )
    if getattr(model, "training", False):
        raise ValueError("model.training must be False; call model.eval() before TideKV")

    if isinstance(max_capacity, int):
        budgets = [max_capacity]
    elif isinstance(max_capacity, Sequence):
        budgets = [int(x) for x in max_capacity]
        if len(budgets) != num_layers:
            raise ValueError(
                f"max_capacity list length must equal num_layers={num_layers}, "
                f"got {len(budgets)}"
            )
    else:
        raise ValueError(
            f"max_capacity must be a positive int or per-layer sequence, got {type(max_capacity)}"
        )
    if any(budget <= 0 for budget in budgets):
        raise ValueError(f"all max_capacity layer budget values must be > 0, got {budgets}")
    if seq_len is not None:
        obs_len = min(int(seq_len), int(obs_window_size))
        for layer_idx in range(warmup_layers, num_layers):
            budget = budgets[0] if len(budgets) == 1 else budgets[layer_idx]
            if seq_len > budget and budget <= obs_len:
                raise ValueError(
                    f"max_capacity layer budget must be > obs_window_size when compression "
                    f"occurs; layer={layer_idx}, budget={budget}, obs_len={obs_len}"
                )

    if attention_mask is not None and not bool(attention_mask.bool().all().item()):
        raise NotImplementedError(
            "attention_mask with padding is not supported by tidekv_prefill"
        )

    layers = model.model.layers
    for layer_idx, layer in enumerate(layers):
        attn = layer.self_attn
        num_heads = int(attn.num_heads)
        num_kv_heads = int(attn.num_key_value_heads)
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads; "
                f"layer={layer_idx}, got {num_heads} and {num_kv_heads}"
            )

    if tidekv_head_adaptive:
        if batch_size != 1:
            raise NotImplementedError(
                f"tidekv_head_adaptive requires batch_size=1, got {batch_size}"
            )
        model_type = str(getattr(model.config, "model_type", "")).lower()
        if model_type != "llama":
            raise NotImplementedError(
                "tidekv_head_adaptive currently supports model_type='llama' only, "
                f"got {model_type!r}"
            )
        impl = getattr(model.config, "attn_implementation", None) or getattr(
            model.config, "_attn_implementation", None
        )
        if str(impl).lower() != "flash_attention_2":
            raise NotImplementedError(
                "tidekv_head_adaptive requires attention backend "
                f"'flash_attention_2', got {impl!r}"
            )
