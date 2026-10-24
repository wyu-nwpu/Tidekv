"""Shared policy interfaces and tensor helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

@dataclass
class CompressionContext:
    layer_idx: int
    keys: torch.Tensor
    values: torch.Tensor
    positions: torch.Tensor
    budget: int
    query_states: torch.Tensor | None = None
    query_positions: torch.Tensor | None = None
    num_key_value_groups: int = 1

@dataclass
class CompressionResult:
    keys: torch.Tensor
    values: torch.Tensor
    positions: torch.Tensor
    keep_indices: torch.Tensor

class BPPPolicy(ABC):
    requires_query_capture = False

    def reset(self, num_layers: int) -> None:
        """Reset per-request state."""

    @abstractmethod
    def compress(self, ctx: CompressionContext) -> CompressionResult:
        raise NotImplementedError

def gather_cache(ctx: CompressionContext, keep: torch.Tensor) -> CompressionResult:
    """Gather K/V/positions with identical sorted indices."""
    keep = keep.sort(dim=-1).values
    head_dim = ctx.keys.shape[-1]
    gather_idx = keep.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    return CompressionResult(
        keys=ctx.keys.gather(2, gather_idx).contiguous(),
        values=ctx.values.gather(2, gather_idx).contiguous(),
        positions=ctx.positions.gather(2, keep).contiguous(),
        keep_indices=keep,
    )

def no_compression(ctx: CompressionContext) -> CompressionResult:
    length = ctx.keys.shape[2]
    keep = torch.arange(length, device=ctx.keys.device)
    keep = keep.view(1, 1, -1).expand(ctx.keys.shape[0], ctx.keys.shape[1], -1)
    return CompressionResult(ctx.keys, ctx.values, ctx.positions, keep)

class NoCompressionPolicy(BPPPolicy):
    """Internal dense-cache policy used for smoke/equivalence tests."""

    def __init__(self, spec) -> None:
        self.spec = spec

    @torch.no_grad()
    def compress(self, ctx: CompressionContext) -> CompressionResult:
        return no_compression(ctx)

