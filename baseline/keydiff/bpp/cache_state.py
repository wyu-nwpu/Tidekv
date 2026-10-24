"""Transformers cache API adapter for BPP."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers.cache_utils import DynamicCache

from .policies.base import CompressionResult

def build_model_positions(
    *,
    seen_tokens: int,
    physical_cache_length: int,
    query_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build absolute RoPE positions and compressed physical cache slots."""
    if seen_tokens < 0 or physical_cache_length < 0 or query_length <= 0:
        raise ValueError(
            "seen_tokens and physical_cache_length must be non-negative; "
            "query_length must be positive"
        )
    absolute = torch.arange(
        seen_tokens,
        seen_tokens + query_length,
        device=device,
        dtype=torch.long,
    )
    physical = torch.arange(
        physical_cache_length,
        physical_cache_length + query_length,
        device=device,
        dtype=torch.long,
    )
    return absolute, physical

@dataclass
class LayerCacheView:
    keys: torch.Tensor
    values: torch.Tensor

class BPPCacheState:
    """Owns DynamicCache plus absolute token positions for each layer."""

    def __init__(self, num_layers: int, model_config=None) -> None:
        try:
            self.cache = DynamicCache(config=model_config)
        except TypeError:
            self.cache = DynamicCache()
        self.seen_tokens = 0
        self.positions: list[torch.Tensor | None] = [None] * num_layers

    def layer(self, layer_idx: int) -> LayerCacheView:
        if hasattr(self.cache, "layers"):
            layer = self.cache.layers[layer_idx]
            return LayerCacheView(keys=layer.keys, values=layer.values)
        return LayerCacheView(
            keys=self.cache.key_cache[layer_idx],
            values=self.cache.value_cache[layer_idx],
        )

    def append_positions(
        self,
        layer_idx: int,
        block_positions: torch.Tensor,
        batch: int,
        kv_heads: int,
    ) -> None:
        current = block_positions.view(1, 1, -1).expand(batch, kv_heads, -1)
        old = self.positions[layer_idx]
        self.positions[layer_idx] = current if old is None else torch.cat([old, current], dim=-1)

    def replace_layer(self, layer_idx: int, result: CompressionResult) -> None:
        if hasattr(self.cache, "layers"):
            layer = self.cache.layers[layer_idx]
            layer.keys = result.keys
            layer.values = result.values
        else:
            self.cache.key_cache[layer_idx] = result.keys
            self.cache.value_cache[layer_idx] = result.values
        self.positions[layer_idx] = result.positions

    def assert_budget(self, budget: int) -> None:
        for layer_idx, key, value in self._iter_layer_tensors():
            positions = self.positions[layer_idx]
            if positions is None:
                raise AssertionError(f"missing positions for layer {layer_idx}")
            assert key.shape[2] <= budget, (layer_idx, key.shape, budget)
            assert key.shape[2] == value.shape[2], (layer_idx, key.shape, value.shape)
            assert key.shape[2] == positions.shape[2], (layer_idx, key.shape, positions.shape)

    def max_cache_length(self) -> int:
        lengths = [key.shape[2] for _layer_idx, key, _value in self._iter_layer_tensors()]
        return max(lengths, default=0)

    def physical_cache_length(self) -> int:
        lengths = [key.shape[2] for _layer_idx, key, _value in self._iter_layer_tensors()]
        if not lengths:
            return 0
        if len(set(lengths)) != 1:
            raise AssertionError(f"layer cache lengths differ: {lengths}")
        return lengths[0]

    def _iter_layer_tensors(self):
        if hasattr(self.cache, "layers"):
            for layer_idx, layer in enumerate(self.cache.layers):
                if getattr(layer, "is_initialized", False):
                    yield layer_idx, layer.keys, layer.values
            return
        for layer_idx, key in enumerate(self.cache.key_cache):
            yield layer_idx, key, self.cache.value_cache[layer_idx]
