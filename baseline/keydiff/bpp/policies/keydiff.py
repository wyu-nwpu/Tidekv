"""KeyDiff-BPP cache selection."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import BPPPolicy, CompressionContext, gather_cache, no_compression

class KeyDiffPolicy(BPPPolicy):
    score_chunk_tokens = 1024

    def __init__(self, spec) -> None:
        self.spec = spec

    @torch.no_grad()
    def compress(self, ctx: CompressionContext):
        if ctx.keys.shape[2] <= ctx.budget:
            return no_compression(ctx)

        if self.spec.keydiff_anchor == "raw_mean":
            # Accumulate only the small anchor in FP32.  The previous full
            # ``keys.float()`` duplicated the complete cache during every
            # compression and inflated the allocator peak by up to one K
            # tensor. The anchor remains FP32 and cosine scores below are
            # computed in bounded FP32 chunks, preserving the KeyDiff rule.
            anchor = ctx.keys.mean(dim=2, keepdim=True, dtype=torch.float32)
        elif self.spec.keydiff_anchor == "normalized_mean":
            anchor_sum = torch.zeros((*ctx.keys.shape[:2], 1, ctx.keys.shape[-1]),
                                     device=ctx.keys.device, dtype=torch.float32)
            for start in range(0, ctx.keys.shape[2], self.score_chunk_tokens):
                chunk = ctx.keys[:, :, start:start + self.score_chunk_tokens].float()
                anchor_sum.add_(F.normalize(chunk, p=2, dim=-1).sum(dim=2, keepdim=True))
            anchor = anchor_sum / ctx.keys.shape[2]
        else:
            raise ValueError(self.spec.keydiff_anchor)

        # Preserve FP32 cosine ranking without materializing a full FP32 copy
        # of the cache. At most ``score_chunk_tokens`` K vectors are promoted.
        scores = []
        for start in range(0, ctx.keys.shape[2], self.score_chunk_tokens):
            chunk = ctx.keys[:, :, start:start + self.score_chunk_tokens].float()
            scores.append(-F.cosine_similarity(chunk, anchor, dim=-1, eps=1e-6))
        score = torch.cat(scores, dim=-1)
        keep = score.topk(ctx.budget, dim=-1, largest=True, sorted=False).indices
        return gather_cache(ctx, keep)
