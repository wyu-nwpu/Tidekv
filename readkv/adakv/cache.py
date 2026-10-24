"""Flatten per-head KV cache used by AdaKV (ported from AdaKV-main)."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from transformers.cache_utils import Cache

_CODE_ROOT = Path(__file__).resolve().parents[3]
for _candidate in (
    _CODE_ROOT / "tidekv-baseline/adakv/AdaKV-main/csrc/build/lib.linux-x86_64-cpython-310",
    _CODE_ROOT / "AdaKV-main/csrc/build/lib.linux-x86_64-cpython-310",
    _CODE_ROOT / "KVCache-Factory-main/csrc",
    _CODE_ROOT / "KVCache-Factory-main/csrc/build/lib.linux-x86_64-cpython-310",
):
    if _candidate.exists():
        _candidate_str = str(_candidate)
        if _candidate_str not in sys.path:
            sys.path.append(_candidate_str)

try:
    from tiny_api_cuda import update_flatten_view as _update_flatten_view_cuda
except ImportError as exc:
    _update_flatten_view_cuda = None
    warnings.warn(
        "tiny_api_cuda.update_flatten_view is unavailable; flattened per-head KV "
        "decode will use a slow PyTorch cat fallback. Build/install tiny_api_cuda "
        f"for good AdaKV/TideKV head-adaptive decode speed. Import error: {exc}",
        RuntimeWarning,
    )

def update_flatten_view_torch(
    cache: torch.Tensor,
    state: torch.Tensor,
    head_lens: torch.Tensor,
    cu_klen: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch fallback for ``tiny_api_cuda.update_flatten_view``."""
    head_lens_list = head_lens.tolist()
    cu = cu_klen.tolist()
    head_num = len(head_lens_list)
    chunks = []
    for i in range(head_num):
        start = cu[i]
        end = start + head_lens_list[i]
        chunks.append(cache[start:end])
        chunks.append(state[i : i + 1])
    return torch.cat(chunks, dim=0)

def update_flatten_view(
    cache: torch.Tensor,
    state: torch.Tensor,
    head_lens: torch.Tensor,
    cu_klen: torch.Tensor,
) -> torch.Tensor:
    if _update_flatten_view_cuda is not None:
        return _update_flatten_view_cuda(cache, state, head_lens, cu_klen)
    return update_flatten_view_torch(cache, state, head_lens, cu_klen)

class DynamicCacheSplitHeadFlatten(Cache):
    """Flattened per-head KV layout for AdaKV varlen attention."""

    def __init__(self) -> None:
        super().__init__()
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self._seen_tokens = 0

    def __len__(self):
        return len(self.key_cache)

    def __iter__(self):
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        raise KeyError(
            f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}"
        )

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            assert self.key_cache[layer_idx].dim() == 2
            bs, head, seqlen, dim = key_states.shape
            assert bs == 1 and seqlen == 1
            head_lens = cache_kwargs["head_lens"]
            cu_klen = cache_kwargs["cu_klen"]
            new_key_cache = update_flatten_view(
                self.key_cache[layer_idx].view(-1, dim),
                key_states.view(-1, dim),
                head_lens,
                cu_klen,
            )
            new_value_cache = update_flatten_view(
                self.value_cache[layer_idx].view(-1, dim),
                value_states.view(-1, dim),
                head_lens,
                cu_klen,
            )
            self.key_cache[layer_idx] = new_key_cache
            self.value_cache[layer_idx] = new_value_cache

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if len(self.key_cache) <= layer_idx:
            return 0
        return 1

    def get_max_length(self) -> Optional[int]:
        return None

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values=None) -> "DynamicCacheSplitHeadFlatten":
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache
