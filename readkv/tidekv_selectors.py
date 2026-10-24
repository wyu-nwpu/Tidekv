"""Chunk selection shared by ChunkKV and TideKV-ChunkKV."""

from __future__ import annotations

import torch

def select_chunk_indices(
    pooled_scores: torch.Tensor,
    num_keep: int,
    chunk_length: int,
) -> torch.Tensor:
    """ChunkKV-style shared chunk selector for TideKV-ChunkKV.

    ``pooled_scores`` is the TideKV prefix score tensor with shape
    ``[B, H_kv, prefix_len]``. ChunkKV ranks chunks after summing scores across
    heads and averaging within each chunk. The returned prefix indices are
    head-uniform, sorted, and have exactly ``num_keep`` positions per batch item.
    """
    if pooled_scores.ndim != 3:
        raise ValueError(
            "pooled_scores must have shape [batch, kv_heads, prefix_len], "
            f"got {tuple(pooled_scores.shape)}"
        )
    if chunk_length <= 0:
        raise ValueError(f"chunk_length must be > 0, got {chunk_length}")

    bsz, _, prefix_len = pooled_scores.shape
    if prefix_len <= 0:
        raise ValueError("prefix_len must be > 0 for ChunkKV selection")
    keep = max(1, min(int(num_keep), int(prefix_len)))
    device = pooled_scores.device

    token_scores = pooled_scores.float().sum(dim=1)
    n_full = prefix_len // int(chunk_length)
    rem = prefix_len - n_full * int(chunk_length)
    selected = []

    for b in range(bsz):
        scores = token_scores[b]
        chunk_parts = []
        if n_full > 0:
            chunk_parts.append(
                scores[: n_full * chunk_length].view(n_full, chunk_length).mean(dim=-1)
            )
        if rem > 0:
            chunk_parts.append(scores[-rem:].mean().view(1))
        if not chunk_parts:
            raise ValueError("ChunkKV selection produced no chunks")

        chunk_scores = torch.cat(chunk_parts)
        chunk_order = torch.argsort(chunk_scores, descending=True)
        kept = []
        total = 0
        for chunk_idx in chunk_order.tolist():
            if chunk_idx < n_full:
                idx = torch.arange(
                    chunk_idx * chunk_length,
                    (chunk_idx + 1) * chunk_length,
                    device=device,
                    dtype=torch.long,
                )
            else:
                idx = torch.arange(
                    n_full * chunk_length,
                    prefix_len,
                    device=device,
                    dtype=torch.long,
                )
            need = keep - total
            if idx.numel() <= need:
                kept.append(idx)
                total += int(idx.numel())
            else:
                local = torch.argsort(scores[idx], descending=True)[:need]
                kept.append(idx[local])
                total += int(need)
            if total >= keep:
                break

        selected.append(torch.cat(kept).sort().values)

    return torch.stack(selected, dim=0)
