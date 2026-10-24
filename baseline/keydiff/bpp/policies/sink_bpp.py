"""Sink-BPP cache selection."""

from __future__ import annotations

import torch

from .base import BPPPolicy, CompressionContext, gather_cache, no_compression

class SinkBPPPolicy(BPPPolicy):
    def __init__(self, spec) -> None:
        self.spec = spec

    @torch.no_grad()
    def compress(self, ctx: CompressionContext):
        length = ctx.keys.shape[2]
        if length <= ctx.budget:
            return no_compression(ctx)

        sink = self.spec.sink_tokens
        recent = ctx.budget - sink
        first = torch.arange(sink, device=ctx.keys.device)
        last = torch.arange(length - recent, length, device=ctx.keys.device)
        keep_1d = torch.cat([first, last]).unique(sorted=True)
        if keep_1d.numel() != ctx.budget:
            raise RuntimeError("Sink selection did not produce exactly budget tokens")
        keep = keep_1d.view(1, 1, -1).expand(ctx.keys.shape[0], ctx.keys.shape[1], -1)
        return gather_cache(ctx, keep)

