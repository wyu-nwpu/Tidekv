"""Snap-BPP cache selection."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .base import BPPPolicy, CompressionContext, gather_cache, no_compression

def compute_snap_attention_scores(
    ctx: CompressionContext,
    query_states: torch.Tensor,
    query_positions: torch.Tensor,
) -> torch.Tensor:
    """Compute post-softmax, post-GQA Snap scores for each KV token."""
    q = query_states.float()
    k = ctx.keys.float()
    groups = ctx.num_key_value_groups
    k_query_heads = k.repeat_interleave(groups, dim=1)
    key_pos = ctx.positions.repeat_interleave(groups, dim=1)

    logits = torch.matmul(q, k_query_heads.transpose(2, 3))
    logits = logits / math.sqrt(q.shape[-1])
    future = key_pos.unsqueeze(2) > query_positions.view(1, 1, -1, 1)
    logits = logits.masked_fill(future, float("-inf"))
    probs = F.softmax(logits, dim=-1, dtype=torch.float32)

    batch, _query_heads, obs, tokens = probs.shape
    kv_heads = ctx.keys.shape[1]
    probs = probs.view(batch, kv_heads, groups, obs, tokens).mean(dim=2)
    return probs.sum(dim=2)

class SnapBPPPolicy(BPPPolicy):
    requires_query_capture = True

    def __init__(self, spec) -> None:
        self.spec = spec
        self._query_buffers: list[torch.Tensor | None] = []
        self._position_buffers: list[torch.Tensor | None] = []

    def reset(self, num_layers: int) -> None:
        self._query_buffers = [None] * num_layers
        self._position_buffers = [None] * num_layers

    @torch.no_grad()
    def compress(self, ctx: CompressionContext):
        length = ctx.keys.shape[2]
        if length <= ctx.budget:
            self._update_query_buffer(ctx)
            return no_compression(ctx)
        if ctx.query_states is None or ctx.query_positions is None:
            raise RuntimeError("Snap-BPP requires captured query states and positions")

        q, q_pos = self._update_query_buffer(ctx)
        score = compute_snap_attention_scores(ctx, q, q_pos)
        batch = ctx.keys.shape[0]
        kv_heads = ctx.keys.shape[1]

        recent = self.spec.snap_recent_tokens
        prefix_len = length - recent
        prefix_budget = ctx.budget - recent
        prefix_score = score[..., :prefix_len]
        prefix_score = F.avg_pool1d(
            prefix_score,
            kernel_size=self.spec.snap_kernel_size,
            stride=1,
            padding=self.spec.snap_kernel_size // 2,
        )
        selected = prefix_score.topk(prefix_budget, dim=-1, largest=True, sorted=False).indices
        recent_idx = torch.arange(prefix_len, length, device=ctx.keys.device)
        recent_idx = recent_idx.view(1, 1, -1).expand(batch, kv_heads, -1)
        keep = torch.cat([selected, recent_idx], dim=-1)
        return gather_cache(ctx, keep)

    def _update_query_buffer(self, ctx: CompressionContext) -> tuple[torch.Tensor, torch.Tensor]:
        if ctx.query_states is None or ctx.query_positions is None:
            previous_q = self._query_buffers[ctx.layer_idx]
            previous_pos = self._position_buffers[ctx.layer_idx]
            if previous_q is None or previous_pos is None:
                raise RuntimeError("Snap query buffer is empty")
            return previous_q, previous_pos

        old_q = self._query_buffers[ctx.layer_idx]
        old_pos = self._position_buffers[ctx.layer_idx]
        q = ctx.query_states if old_q is None else torch.cat([old_q, ctx.query_states], dim=2)
        pos = ctx.query_positions if old_pos is None else torch.cat([old_pos, ctx.query_positions], dim=0)
        window = self.spec.snap_observation_window
        q = q[:, :, -window:, :].contiguous()
        pos = pos[-window:].contiguous()
        self._query_buffers[ctx.layer_idx] = q
        self._position_buffers[ctx.layer_idx] = pos
        return q, pos
