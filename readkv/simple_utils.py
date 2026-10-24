from transformers.cache_utils import Cache
import torch
import torch.nn as nn
import math
import torch.nn.functional as F  
from typing import Optional, Tuple, Union,Dict,Any

def repeat_kv_heads(hidden_states: torch.Tensor, num_key_value_groups: int) -> torch.Tensor:
    """Expand compact KV heads to query-head layout without model-specific imports."""
    if num_key_value_groups == 1:
        return hidden_states
    return hidden_states.repeat_interleave(num_key_value_groups, dim=1)

def grouped_query_attention_weights(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    num_key_value_groups: int,
    causal: bool = True,
) -> torch.Tensor:
    """
    Compute attention per query head, then aggregate probabilities per KV group.

    This is the faithful GQA extension used by strict compact-cache baselines.
    Averaging query heads before softmax is intentionally avoided.
    """
    bsz, num_query_heads, q_len, head_dim = query_states.shape
    num_kv_heads = key_states.shape[1]
    if num_query_heads != num_kv_heads * num_key_value_groups:
        raise ValueError(
            "query/KV head mismatch: "
            f"{num_query_heads=} {num_kv_heads=} {num_key_value_groups=}"
        )
    expanded_keys = repeat_kv_heads(key_states, num_key_value_groups)
    attn_weights = torch.matmul(
        query_states, expanded_keys.transpose(2, 3)
    ) / math.sqrt(head_dim)
    if causal and q_len > 1:
        kv_len = key_states.shape[-2]
        mask = torch.full(
            (q_len, kv_len),
            torch.finfo(attn_weights.dtype).min,
            device=attn_weights.device,
            dtype=attn_weights.dtype,
        )
        mask = torch.triu(mask, diagonal=1 + kv_len - q_len)
        attn_weights = attn_weights + mask[None, None, :, :]
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    return attn_weights.view(
        bsz, num_kv_heads, num_key_value_groups, q_len, key_states.shape[-2]
    ).mean(dim=2)

def init_h2o(self):

    if not hasattr(self, "kv_cluster"):

        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 256
        if not hasattr(self.config, 'sink_size'):
            self.config.sink_size = 4
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
            
        self.kv_cluster = H2OKVCache(
            window_size=self.config.window_size,
            sink_size=0,
            max_capacity_prompt = self.config.max_capacity_prompt,
        )

def init_StreamingLLM(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'sink_size'):
            self.config.sink_size = 64
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'merge'):
            self.config.merge = None

        self.kv_cluster = StreamingLLMKVCluster(
            sink_size=self.config.sink_size,
            max_capacity_prompt=self.config.max_capacity_prompt,
            kernel_size=self.config.kernel_size,
            pooling=self.config.pooling,
            merge=self.config.merge,
        )



class H2OKVCache:
    def __init__(
        self, 
        window_size: int = 64, 
        sink_size: int = 4, 
        max_capacity_prompt: int = 2048,
        **kwargs
    ):
        self.window_size = window_size
        self.sink_size = sink_size
        self.max_capacity_prompt = max_capacity_prompt

        self.heavy_hitter_size = max_capacity_prompt - window_size - sink_size

        self.accumulated_scores: Optional[torch.Tensor] = None

        if self.heavy_hitter_size < 0:
            raise ValueError("max_capacity_prompt must be larger than window_size + sink_size")

    def reset(self):
        self.accumulated_scores = None

    def evict_and_compress(
        self, 
        key_states: torch.Tensor, 
        value_states: torch.Tensor, 
        attn_weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        bsz, num_heads, seq_len, head_dim = key_states.shape
        q_len = attn_weights.shape[2]

        if seq_len <= self.max_capacity_prompt:

            if q_len > 1:
                current_scores = attn_weights.sum(dim=-2).to(torch.float32)
            else:
                current_scores = attn_weights.squeeze(-2).to(torch.float32)

            if self.accumulated_scores is None:
                self.accumulated_scores = current_scores
            else:

                prev_len = self.accumulated_scores.shape[-1]
                if seq_len > prev_len:
                    padding = torch.zeros(
                        (bsz, num_heads, seq_len - prev_len),
                        device=key_states.device,
                        dtype=torch.float32
                    )
                    self.accumulated_scores = torch.cat([self.accumulated_scores, padding], dim=-1)

                self.accumulated_scores += current_scores

            return key_states, value_states

        if q_len > 1:
            current_scores = attn_weights.sum(dim=-2).to(torch.float32)
        else:
            current_scores = attn_weights.squeeze(-2).to(torch.float32)

        if self.accumulated_scores is None:

            self.accumulated_scores = current_scores
        else:
            prev_len = self.accumulated_scores.shape[-1]
            if seq_len > prev_len:
                padding = torch.zeros((bsz, num_heads, seq_len - prev_len), device=key_states.device, dtype=torch.float32)
                self.accumulated_scores = torch.cat([self.accumulated_scores, padding], dim=-1)
            self.accumulated_scores += current_scores

        sink_indices = torch.arange(0, self.sink_size, device=key_states.device)
        sink_indices = sink_indices.view(1, 1, -1).expand(bsz, num_heads, -1)

        window_start = seq_len - self.window_size
        window_indices = torch.arange(window_start, seq_len, device=key_states.device)
        window_indices = window_indices.view(1, 1, -1).expand(bsz, num_heads, -1)

        candidate_start = self.sink_size
        candidate_end = seq_len - self.window_size
        
        if candidate_end > candidate_start:

            candidate_scores = self.accumulated_scores[:, :, candidate_start:candidate_end]

            _, topk_indices = candidate_scores.topk(self.heavy_hitter_size, dim=-1)

            topk_indices = topk_indices + candidate_start
            topk_indices, _ = topk_indices.sort(dim=-1)

            keep_indices = torch.cat([sink_indices, topk_indices, window_indices], dim=-1)
        else:

            keep_indices = torch.cat([sink_indices, window_indices], dim=-1)

        kv_gather_idx = keep_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        
        compressed_k = key_states.gather(dim=2, index=kv_gather_idx)
        compressed_v = value_states.gather(dim=2, index=kv_gather_idx)

        self.accumulated_scores = self.accumulated_scores.gather(dim=2, index=keep_indices)
        
        return compressed_k, compressed_v

class StreamingLLMKVCluster():
    def __init__(self, sink_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', merge = None):
        self.sink_size = sink_size
        self.max_capacity_prompt = max_capacity_prompt
        self.window_size = max_capacity_prompt - sink_size
        assert self.window_size > 0, (
            f"window_size must be > 0, got max_capacity_prompt={max_capacity_prompt}, sink_size={sink_size}"
        )
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge

    def reset(self, sink_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', merge = None):
        self.sink_size = sink_size
        self.max_capacity_prompt = max_capacity_prompt
        self.window_size = max_capacity_prompt - sink_size
        assert self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge

    def compress_kv(self, key_states, value_states):
        seq_len = key_states.shape[-2]

        if seq_len <= self.max_capacity_prompt:
            return key_states, value_states

        sink_k = key_states[:, :, :self.sink_size, :]
        sink_v = value_states[:, :, :self.sink_size, :]

        window_k = key_states[:, :, -self.window_size:, :]
        window_v = value_states[:, :, -self.window_size:, :]

        key_states_compressed = torch.cat([sink_k, window_k], dim=-2)
        value_states_compressed = torch.cat([sink_v, window_v], dim=-2)

        return key_states_compressed, value_states_compressed

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):

        return self.compress_kv(key_states, value_states)

class _SnapKVConfig:
    def __init__(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', merge = None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', merge = None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge


def init_chunkkv(self):
    """Inject ChunkKV cluster — SnapKV scoring + chunk-wise token selection (kvpress ChunkKVPress)."""
    if hasattr(self, "kv_cluster"):
        return
    if not hasattr(self.config, "window_size"):
        self.config.window_size = 64
    if not hasattr(self.config, "max_capacity_prompt"):
        self.config.max_capacity_prompt = 1024
    if not hasattr(self.config, "kernel_size"):
        self.config.kernel_size = 5
    if not hasattr(self.config, "pooling"):
        self.config.pooling = "avgpool"
    if not hasattr(self.config, "chunk_length"):
        self.config.chunk_length = 20
    if not hasattr(self.config, "chunkkv_reuse_layers"):
        self.config.chunkkv_reuse_layers = 2

    self.kv_cluster = ChunkKVCluster(
        window_size=self.config.window_size,
        max_capacity_prompt=self.config.max_capacity_prompt,
        kernel_size=self.config.kernel_size,
        pooling=self.config.pooling,
        chunk_length=self.config.chunk_length,
        layer_idx=getattr(self, "layer_idx", None),
        reuse_layers=self.config.chunkkv_reuse_layers,
    )

class ChunkKVCluster:
    """
    ChunkKV (kvpress): SnapKV global importance scores + chunk-wise selection.

    Ported from kvpress ``ChunkKVPress`` + ``SnapKVPress`` (evaluate_registry:
    ChunkKVPress(press=SnapKVPress(), chunk_length=20)).

    1. Compute per-token SnapKV scores over full sequence (obs window padded with max+1).
    2. Aggregate scores per chunk; keep top chunks proportional to ``compression_ratio``.
    3. Gather all tokens from selected chunks (sorted indices).
    """

    _reuse_index_cache = {}

    def __init__(
        self,
        window_size=64,
        max_capacity_prompt=1024,
        kernel_size=5,
        pooling="avgpool",
        chunk_length=20,
        layer_idx=None,
        reuse_layers=2,
    ):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.chunk_length = chunk_length
        self.layer_idx = 0 if layer_idx is None else int(layer_idx)
        self.reuse_layers = max(1, int(reuse_layers))
        assert self.chunk_length > 0
        assert self.max_capacity_prompt > 0

    def reset(
        self,
        window_size=64,
        max_capacity_prompt=1024,
        kernel_size=5,
        pooling="avgpool",
        chunk_length=20,
        reuse_layers=None,
    ):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.chunk_length = chunk_length
        if reuse_layers is not None:
            self.reuse_layers = max(1, int(reuse_layers))
        if self.layer_idx == 0:
            type(self)._reuse_index_cache.clear()

    def _compute_snapkv_global_scores(
        self, key_states, query_states, num_key_value_groups=1
    ):
        """SnapKVPress.score semantics: shape [B, H_kv, seq_len]."""
        bsz, num_heads, q_len, head_dim = query_states.shape
        if q_len <= self.window_size:
            return torch.ones(
                bsz,
                key_states.shape[1],
                q_len,
                device=key_states.device,
                dtype=key_states.dtype,
            )

        obs_win = min(self.window_size, q_len)
        attn_weights = grouped_query_attention_weights(
            query_states[..., -obs_win:, :],
            key_states,
            num_key_value_groups,
            causal=True,
        )

        scores = attn_weights[:, :, -obs_win:, : -obs_win].mean(dim=-2)
        if self.pooling == "avgpool":
            scores = F.avg_pool1d(
                scores, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1
            )
        elif self.pooling == "maxpool":
            scores = F.max_pool1d(
                scores, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1
            )
        else:
            raise ValueError(f"Unsupported pooling: {self.pooling}")

        pad_val = scores.max().item() + 1.0
        scores = F.pad(scores, (0, self.window_size), value=pad_val)
        return scores

    def _compression_ratio(self, kv_len: int) -> float:
        if kv_len <= self.max_capacity_prompt:
            return 0.0
        return 1.0 - (self.max_capacity_prompt / kv_len)

    def _select_indices(self, global_scores: torch.Tensor, kv_len: int) -> torch.Tensor:
        """Return sorted token indices [B, H, n_kept] (same for all heads in reference batch loop)."""
        device = global_scores.device
        compression_ratio = self._compression_ratio(kv_len)
        bsz, num_heads, _ = global_scores.shape

        num_complete_chunks = kv_len // self.chunk_length
        remaining_tokens = kv_len % self.chunk_length

        if compression_ratio <= 0:
            return torch.arange(kv_len, device=device).view(1, 1, -1).expand(bsz, num_heads, -1)

        if num_complete_chunks == 0:
            n_kept = max(1, int(kv_len * (1.0 - compression_ratio)))
            return global_scores.topk(n_kept, dim=-1).indices.sort(dim=-1).values

        per_batch_indices = []
        for b in range(bsz):
            scores_b = global_scores[b : b + 1]
            if num_complete_chunks > 0:
                main_scores = scores_b[..., : num_complete_chunks * self.chunk_length]
                main_chunk_scores = main_scores.sum(dim=1).view(-1, num_complete_chunks, self.chunk_length)
                main_chunk_scores = main_chunk_scores.mean(dim=-1)
            else:
                main_chunk_scores = torch.empty((1, 0), device=device, dtype=scores_b.dtype)

            if remaining_tokens > 0:
                remaining_scores = scores_b[..., -remaining_tokens:]
                remaining_chunk_score = remaining_scores.sum(dim=1).mean(dim=-1, keepdim=True)
                chunk_scores = torch.cat([main_chunk_scores, remaining_chunk_score], dim=-1)
            else:
                chunk_scores = main_chunk_scores

            n_chunks_kept = max(
                1, int((num_complete_chunks + (remaining_tokens > 0)) * (1.0 - compression_ratio))
            )
            top_chunks = chunk_scores.topk(n_chunks_kept, dim=-1)

            batch_indices = []
            for chunk_idx in top_chunks.indices[0]:
                chunk_idx = int(chunk_idx.item())
                if chunk_idx < num_complete_chunks:
                    start_idx = chunk_idx * self.chunk_length
                    chunk_indices = torch.arange(
                        start_idx, start_idx + self.chunk_length, device=device
                    )
                else:
                    chunk_indices = torch.arange(
                        num_complete_chunks * self.chunk_length, kv_len, device=device
                    )
                batch_indices.append(chunk_indices)

            per_batch_indices.append(torch.cat(batch_indices).sort()[0])

        indices_1d = torch.stack(per_batch_indices, dim=0)
        return indices_1d.view(bsz, 1, -1).expand(bsz, num_heads, -1)

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = key_states.shape

        if q_len < self.max_capacity_prompt:
            return key_states, value_states

        reuse_group = self.layer_idx // self.reuse_layers
        cache_key = (reuse_group, q_len, self.max_capacity_prompt, self.chunk_length)
        reuse_from_previous = (
            self.reuse_layers > 1
            and self.layer_idx % self.reuse_layers != 0
            and cache_key in type(self)._reuse_index_cache
        )
        if reuse_from_previous:
            indices = type(self)._reuse_index_cache[cache_key].to(key_states.device)
        else:
            global_scores = self._compute_snapkv_global_scores(
                key_states, query_states, num_key_value_groups
            )
            indices = self._select_indices(global_scores, q_len)
            if self.reuse_layers > 1:
                type(self)._reuse_index_cache[cache_key] = indices.detach().cpu()
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

        key_states = key_states.gather(dim=2, index=indices).contiguous()
        value_states = value_states.gather(dim=2, index=indices).contiguous()
        return key_states, value_states


class _PyramidKVBase:
    """Shared configuration and compact K/V gathering for PyramidKV-A."""

    def __init__(
        self,
        num_hidden_layers=32,
        window_size=128,
        max_capacity_prompt=1024,
        kernel_size=7,
        pooling="avgpool",
        beta=20,
        layer_idx=None,
        merge=None,
    ):
        self.num_hidden_layers = int(num_hidden_layers)
        self.layer_idx = 0 if layer_idx is None else int(layer_idx)
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.beta = beta
        self.merge = merge
        assert self.max_capacity_prompt - self.window_size > 0

    def reset(
        self,
        window_size=128,
        max_capacity_prompt=1024,
        kernel_size=7,
        pooling="avgpool",
        beta=20,
        merge=None,
    ):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.beta = beta
        self.merge = merge
        assert self.max_capacity_prompt - self.window_size > 0



    def _compress_with_topk(self, key_states, value_states, attn_cache, prefix_topk: int):
        bsz, num_heads, _, head_dim = key_states.shape
        indices = attn_cache.topk(prefix_topk, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        if self.merge is not None:
            raise NotImplementedError("PyramidKV merge mode is not ported to ReadKV")
        k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim=2, index=indices)
        v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim=2, index=indices)
        k_cur = key_states[:, :, -self.window_size:, :]
        v_cur = value_states[:, :, -self.window_size:, :]
        return torch.cat([k_past_compress, k_cur], dim=2), torch.cat([v_past_compress, v_cur], dim=2)



def pool_center_block(
    score_kv,
    raw_start,
    raw_end,
    center_start,
    center_end,
    prefix_len,
    kernel_size,
    pooling="avgpool",
):

    B, H_kv, raw_len = score_kv.shape
    pad = kernel_size // 2
    center_len = center_end - center_start
    ext_len = center_len + 2 * pad
    ext_global_start = center_start - pad

    ext = torch.zeros(B, H_kv, ext_len, device=score_kv.device, dtype=score_kv.dtype)
    insert_start = raw_start - ext_global_start
    insert_end = insert_start + raw_len
    ext[:, :, insert_start:insert_end] = score_kv

    x = ext.reshape(B * H_kv, 1, ext_len)
    if pooling == "avgpool":
        y = F.avg_pool1d(x, kernel_size=kernel_size, stride=1)
    elif pooling == "maxpool":
        y = F.max_pool1d(x, kernel_size=kernel_size, stride=1)
    else:
        raise ValueError(f"Unsupported pooling: {pooling}")

    out_len = y.shape[-1]
    if out_len != center_len:
        raise RuntimeError(
            f"pool_center_block length mismatch: got {out_len}, expected {center_len} "
            f"(ext_len={ext_len}, kernel_size={kernel_size})"
        )
    return y.reshape(B, H_kv, center_len)

def update_streaming_topk(top_vals, top_idx, block_vals, block_idx, k):

    if top_vals is None:
        vals = block_vals
        idx = block_idx
    else:
        vals = torch.cat([top_vals, block_vals], dim=-1)
        idx = torch.cat([top_idx, block_idx], dim=-1)
    keep = min(k, vals.shape[-1])
    index_order = torch.argsort(idx, dim=-1, stable=True)
    vals = vals.gather(dim=-1, index=index_order)
    idx = idx.gather(dim=-1, index=index_order)
    # Block-wise logsumexp can differ by a few float32 ulps from a dense pass.
    # Treat those numerically indistinguishable values as ties, then prefer the
    # smaller original token index.
    tie_epsilon = 1e-6
    rank_vals = torch.round(vals.float() / tie_epsilon) * tie_epsilon
    value_order = torch.argsort(rank_vals, dim=-1, descending=True, stable=True)
    pos = value_order[..., :keep]
    new_vals = vals.gather(dim=-1, index=pos)
    new_idx = idx.gather(dim=-1, index=pos)
    return new_vals, new_idx

def build_causal_score_mask(query_positions, key_positions, dtype=torch.float32):
    """Build a block-local additive causal mask for observation-query scoring."""
    if query_positions.dim() != 1 or key_positions.dim() != 1:
        raise ValueError(
            "query_positions and key_positions must be rank-1 tensors, got "
            f"{query_positions.shape} and {key_positions.shape}"
        )
    blocked = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
    mask = torch.zeros(
        blocked.shape,
        device=query_positions.device,
        dtype=dtype,
    )
    return mask.masked_fill(blocked, torch.finfo(dtype).min)

def allocate_adaptive_head_keep(
    pooled_scores,
    prefix_keep,
    floor_alpha=0.5,
    normalize=False,
    sorted_scores=None,
):
    """
    AdaKV-style per-head prefix budget allocation.

    ``pooled_scores``: [B, H_kv, prefix_len].
    Returns [B, H_kv] keep counts. After AdaKV-style floor/vote blending, an
    exact-budget marginal-score correction enforces ``sum(head_keep) ==
    num_heads * prefix_keep`` for fair cache accounting.
    """
    bsz, num_heads, prefix_len = pooled_scores.shape
    if prefix_keep <= 0:
        raise ValueError(f"prefix_keep must be positive, got {prefix_keep}")
    if prefix_keep > prefix_len:
        return torch.full(
            (bsz, num_heads),
            prefix_len,
            device=pooled_scores.device,
            dtype=torch.long,
        )

    if sorted_scores is None:
        scores = pooled_scores.float()
        sorted_scores, _ = scores.sort(dim=-1, descending=True)
    else:
        sorted_scores = sorted_scores.float()
    length = sorted_scores.shape[-1]
    if normalize:
        denom = sorted_scores.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        ratio_weight = sorted_scores[..., :prefix_keep].sum(dim=-1, keepdim=True) / denom
        sorted_scores = sorted_scores * ratio_weight

    total_keep = num_heads * int(prefix_keep)
    flat = sorted_scores.reshape(bsz, num_heads * length)
    global_idx = torch.topk(flat, k=total_keep, dim=-1).indices
    head_votes = global_idx // length
    votes = torch.zeros(
        (bsz, num_heads), device=pooled_scores.device, dtype=torch.float32
    )
    votes.scatter_add_(
        -1,
        head_votes,
        torch.ones_like(head_votes, dtype=votes.dtype),
    )

    floor_alpha = float(floor_alpha)
    if floor_alpha < 0.0 or floor_alpha > 1.0:
        raise ValueError(f"floor_alpha must be in [0, 1], got {floor_alpha}")
    floor_keep = min(prefix_keep, int(prefix_keep * floor_alpha))
    raw = votes * (1.0 - floor_alpha) + floor_keep
    keep = torch.round(raw).to(torch.long).clamp_(min=floor_keep, max=prefix_len)

    # Rounding/clipping can change the total. Correct it using the next marginal
    # score to add, or the weakest currently retained score to remove.
    for batch_idx in range(bsz):
        diff = total_keep - int(keep[batch_idx].sum().item())
        while diff != 0:
            if diff > 0:
                candidates = []
                for head_idx in range(num_heads):
                    count = int(keep[batch_idx, head_idx].item())
                    if count < prefix_len:
                        candidates.append((
                            float(sorted_scores[batch_idx, head_idx, count].item()),
                            head_idx,
                        ))
                if not candidates:
                    raise RuntimeError(
                        f"cannot add {diff} head-adaptive entries to reach exact budget"
                    )
                _, head_idx = max(candidates)
                keep[batch_idx, head_idx] += 1
                diff -= 1
            else:
                candidates = []
                for head_idx in range(num_heads):
                    count = int(keep[batch_idx, head_idx].item())
                    if count > floor_keep:
                        candidates.append((
                            float(sorted_scores[batch_idx, head_idx, count - 1].item()),
                            head_idx,
                        ))
                if not candidates:
                    raise RuntimeError(
                        f"cannot remove {-diff} head-adaptive entries to reach exact budget"
                    )
                _, head_idx = min(candidates)
                keep[batch_idx, head_idx] -= 1
                diff += 1
    return keep

def pack_adaptive_heads_padded(
    full_k,
    full_v,
    sorted_prefix_indices,
    head_keep,
    obs_len,
):
    """
    Build a rectangular [B, H, max_head_len, D] cache from per-head keep counts.

    This v7.4-v0 packing intentionally pads shorter heads by repeating each
    head's last real KV entry. It preserves the existing DynamicCache/decode
    path and is meant to test accuracy impact before a varlen-cache version.
    """
    bsz, num_heads, _, head_dim = full_k.shape
    if bsz != 1:
        raise NotImplementedError("TideKV head-adaptive v0 currently supports batch_size=1")
    device, dtype = full_k.device, full_k.dtype
    prefix_len = full_k.shape[2] - obs_len
    head_lens = head_keep + obs_len
    max_len = int(head_lens.max().item())

    out_k = torch.empty(bsz, num_heads, max_len, head_dim, device=device, dtype=dtype)
    out_v = torch.empty_like(out_k)

    for h in range(num_heads):
        cap = int(head_keep[0, h].item())
        if cap > 0:
            idx = sorted_prefix_indices[:, h : h + 1, :cap].sort(dim=-1).values
            idx = idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_prefix = full_k[:, h : h + 1, :prefix_len, :].gather(dim=2, index=idx)
            v_prefix = full_v[:, h : h + 1, :prefix_len, :].gather(dim=2, index=idx)
            selected_k = torch.cat([k_prefix, full_k[:, h : h + 1, -obs_len:, :]], dim=2)
            selected_v = torch.cat([v_prefix, full_v[:, h : h + 1, -obs_len:, :]], dim=2)
        else:
            selected_k = full_k[:, h : h + 1, -obs_len:, :]
            selected_v = full_v[:, h : h + 1, -obs_len:, :]

        length = selected_k.shape[2]
        out_k[:, h : h + 1, :length, :] = selected_k
        out_v[:, h : h + 1, :length, :] = selected_v
        if length < max_len:
            pad_k = selected_k[:, :, -1:, :].expand(bsz, 1, max_len - length, head_dim)
            pad_v = selected_v[:, :, -1:, :].expand(bsz, 1, max_len - length, head_dim)
            out_k[:, h : h + 1, length:, :] = pad_k
            out_v[:, h : h + 1, length:, :] = pad_v

    return out_k, out_v, head_lens[0].detach().cpu().tolist()

def _build_flatten_head_metadata(
    head_lens,
    num_key_value_groups,
    max_seqlen_k=None,
    klen_sum=None,
):
    head_lens = head_lens.to(dtype=torch.int32)
    cu_klen = torch.zeros(
        head_lens.shape[0] + 1,
        device=head_lens.device,
        dtype=torch.int32,
    )
    cu_klen[1:] = torch.cumsum(head_lens, dim=0)
    return {
        "head_lens": head_lens,
        "cu_klen": cu_klen,
        # One new KV token is appended to every KV head during each decode step.
        "cu_offset": torch.arange(
            0,
            head_lens.shape[0] + 1,
            device=head_lens.device,
            dtype=torch.int32,
        ),
        # GQA varlen treats each KV head as one sequence. The query tensor shape is
        # [num_kv_heads, num_key_value_groups, head_dim], so cu_qlen advances by 1.
        "cu_qlen": torch.arange(
            0,
            head_lens.shape[0] + 1,
            device=head_lens.device,
            dtype=torch.int32,
        ),
        "max_seqlen_k": (
            int(max_seqlen_k)
            if max_seqlen_k is not None
            else (int(head_lens.max().item()) if head_lens.numel() else 0)
        ),
        "klen_sum": (
            int(klen_sum)
            if klen_sum is not None
            else int(head_lens.sum().item())
        ),
        "num_key_value_groups": int(num_key_value_groups),
    }

def pack_uniform_heads_flatten(full_k, full_v, num_key_value_groups):
    """
    Flatten [B, H_kv, S, D] KV into AdaKV/SnapKV varlen layout.

    The output groups tokens by KV head:
        head0 token0..S-1, head1 token0..S-1, ...
    """
    bsz, num_heads, seq_len, head_dim = full_k.shape
    if bsz != 1:
        raise NotImplementedError("TideKV head-adaptive varlen currently supports batch_size=1")

    flat_k = full_k.reshape(num_heads, seq_len, head_dim).reshape(-1, head_dim).contiguous()
    flat_v = full_v.reshape(num_heads, seq_len, head_dim).reshape(-1, head_dim).contiguous()
    head_lens = torch.full(
        (num_heads,),
        seq_len,
        device=full_k.device,
        dtype=torch.int32,
    )
    metadata = _build_flatten_head_metadata(
        head_lens,
        num_key_value_groups,
        max_seqlen_k=seq_len,
        klen_sum=num_heads * seq_len,
    )
    return flat_k, flat_v, metadata

def pack_adaptive_heads_flatten(
    full_k,
    full_v,
    sorted_prefix_indices,
    head_keep,
    obs_len,
    num_key_value_groups,
):
    """
    Build a true per-head flattened varlen cache from adaptive keep counts.

    Unlike ``pack_adaptive_heads_padded``, this does not create fake padded KV
    entries. Decode must use flash-attn varlen with the returned metadata.
    """
    bsz, num_heads, _, head_dim = full_k.shape
    if bsz != 1:
        raise NotImplementedError("TideKV head-adaptive varlen currently supports batch_size=1")

    prefix_len = full_k.shape[2] - obs_len
    flat_ks = []
    flat_vs = []
    lens = []
    head_keep_values = head_keep[0].detach().cpu().tolist()

    for h in range(num_heads):
        cap = int(head_keep_values[h])
        if cap > 0:
            idx = sorted_prefix_indices[:, h : h + 1, :cap].sort(dim=-1).values
            idx = idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_prefix = full_k[:, h : h + 1, :prefix_len, :].gather(dim=2, index=idx)
            v_prefix = full_v[:, h : h + 1, :prefix_len, :].gather(dim=2, index=idx)
            selected_k = torch.cat([k_prefix, full_k[:, h : h + 1, -obs_len:, :]], dim=2)
            selected_v = torch.cat([v_prefix, full_v[:, h : h + 1, -obs_len:, :]], dim=2)
        else:
            selected_k = full_k[:, h : h + 1, -obs_len:, :]
            selected_v = full_v[:, h : h + 1, -obs_len:, :]

        lens.append(selected_k.shape[2])
        flat_ks.append(selected_k.reshape(-1, head_dim))
        flat_vs.append(selected_v.reshape(-1, head_dim))

    flat_k = torch.cat(flat_ks, dim=0).contiguous()
    flat_v = torch.cat(flat_vs, dim=0).contiguous()
    head_lens = torch.tensor(lens, device=full_k.device, dtype=torch.int32)
    metadata = _build_flatten_head_metadata(
        head_lens,
        num_key_value_groups,
        max_seqlen_k=max(lens) if lens else 0,
        klen_sum=sum(lens),
    )
    return flat_k, flat_v, metadata

# Snap-A and Pyramid-A select once per KV group and retain compact KV heads.
# The same grouped attention scores are used by their paired TideKV selectors.
class SnapKVClusterA(_SnapKVConfig):
    """SnapKV with TideKV-style GQA-group-consensus selection (compact KV).

    Inputs: ``key_states``/``value_states`` are KV-head-compact ``[B, Hkv, S, D]``;
    ``query_states`` is the full ``[B, Hq, S, D]`` (NOT mean-aggregated), so each
    head's attention distribution is preserved before the group sum.
    """

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        bsz, num_kv_heads, q_len, head_dim = key_states.shape
        groups = num_key_value_groups

        # prefill detection identical to SnapKVCluster (budget measured in tokens)
        if q_len < self.max_capacity_prompt:
            return key_states, value_states

        window = self.window_size
        # per-query-head observation-window attention against the (repeated) compact keys
        key_q = repeat_kv_heads(key_states, groups) if groups > 1 else key_states  # [B, Hq, S, D]
        attn_weights = torch.matmul(
            query_states[..., -window:, :], key_q.transpose(2, 3)
        ) / math.sqrt(head_dim)
        # same intra-window causal mask as SnapKV
        mask = torch.full(
            (window, window), torch.finfo(attn_weights.dtype).min, device=attn_weights.device
        )
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        attn_weights[:, :, -window:, -window:] += mask[None, None, :, :]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # sum over the observation-window query positions, prefix columns only -> [B, Hq, prefix]
        attn_weights_sum = attn_weights[:, :, -window:, :-window].sum(dim=-2)
        # === the only change vs SnapKV: GQA group consensus ===
        # sum the per-head attention across the query heads of each KV group -> [B, Hkv, prefix]
        prefix_len = attn_weights_sum.shape[-1]
        attn_weights_sum = attn_weights_sum.view(bsz, num_kv_heads, groups, prefix_len).sum(dim=2)
        if self.pooling == 'avgpool':
            attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)
        elif self.pooling == 'maxpool':
            attn_cache = F.max_pool1d(attn_weights_sum, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)
        else:
            raise ValueError('Pooling method not supported')
        indices = attn_cache.topk(self.max_capacity_prompt - window, dim=-1).indices  # [B, Hkv, keep]
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        # gather from the COMPACT (KV-head) cache -> compact compressed cache
        k_past_compress = key_states[:, :, :-window, :].gather(dim=2, index=indices)
        v_past_compress = value_states[:, :, :-window, :].gather(dim=2, index=indices)
        k_cur = key_states[:, :, -window:, :]
        v_cur = value_states[:, :, -window:, :]
        key_states = torch.cat([k_past_compress, k_cur], dim=2)
        value_states = torch.cat([v_past_compress, v_cur], dim=2)
        return key_states, value_states

class PyramidKVClusterA(_PyramidKVBase):
    """PyramidKV with TideKV-style GQA-group-consensus selection (compact KV).

    The observation-window attention is pooled across the query heads of each KV
    group (group sum) to produce one consensus Top-K per KV head. When
    ``source_gqa_ratio > 1``, its per-layer capacity exactly follows the compact
    layer schedule used by TideKV-Pyramid.
    """

    def __init__(self, *args, source_gqa_ratio=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.source_gqa_ratio = int(source_gqa_ratio)
        if self.source_gqa_ratio < 1:
            raise ValueError("source_gqa_ratio must be >= 1")
        if self.max_capacity_prompt % self.source_gqa_ratio:
            raise ValueError(
                f"max_capacity_prompt={self.max_capacity_prompt} must be divisible by "
                f"source_gqa_ratio={self.source_gqa_ratio}"
            )
        reference_capacity = self.max_capacity_prompt // self.source_gqa_ratio
        if reference_capacity <= self.window_size:
            raise ValueError(
                f"Pyramid reference capacity={reference_capacity} must exceed "
                f"window_size={self.window_size}"
            )

    def _aligned_total_capacity(self, q_len):
        """Return this layer's TideKV-Pyramid-matched compact capacity."""
        ratio = self.source_gqa_ratio
        reference_capacity = self.max_capacity_prompt // ratio
        base_prefix = reference_capacity - self.window_size
        min_prefix = base_prefix // self.beta
        max_prefix = base_prefix * 2 - min_prefix
        step = (max_prefix - min_prefix) // max(self.num_hidden_layers - 1, 1)
        source_prefix = max_prefix - self.layer_idx * step
        return min(int(q_len), (source_prefix + self.window_size) * ratio)

    def _obs_window_attn_cache_grouped(self, key_states, query_states, groups):
        bsz, num_kv_heads, q_len, head_dim = key_states.shape
        obs_win = min(self.window_size, q_len)
        key_q = repeat_kv_heads(key_states, groups) if groups > 1 else key_states
        attn_weights = torch.matmul(
            query_states[..., -obs_win:, :], key_q.transpose(2, 3)
        ) / math.sqrt(head_dim)
        mask = torch.full(
            (obs_win, obs_win), torch.finfo(attn_weights.dtype).min, device=attn_weights.device
        )
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        attn_weights[:, :, -obs_win:, -obs_win:] += mask[None, None, :, :]
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_sum = attn_weights[:, :, -obs_win:, :-obs_win].sum(dim=-2)  # [B, Hq, prefix]
        prefix_len = attn_weights_sum.shape[-1]
        # GQA group consensus: sum across the query heads of each KV group -> [B, Hkv, prefix]
        attn_weights_sum = attn_weights_sum.view(bsz, num_kv_heads, groups, prefix_len).sum(dim=2)
        if self.pooling == "avgpool":
            return F.avg_pool1d(attn_weights_sum, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)
        if self.pooling == "maxpool":
            return F.max_pool1d(attn_weights_sum, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)
        raise ValueError(f"Unsupported pooling: {self.pooling}")

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        q_len = query_states.shape[-2]
        total_capacity = self._aligned_total_capacity(q_len)
        if q_len <= total_capacity:
            return key_states, value_states
        attn_cache = self._obs_window_attn_cache_grouped(key_states, query_states, num_key_value_groups)
        prefix_topk = total_capacity - self.window_size
        # _compress_with_topk gathers from the compact (KV-head) cache with per-KV-head indices
        return self._compress_with_topk(key_states, value_states, attn_cache, prefix_topk)

def init_snapkv_a(self):
    """Inject the SnapKV-A cluster (GQA-group-consensus selection, compact KV)."""
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 4096
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'merge'):
            self.config.merge = None
        self.kv_cluster = SnapKVClusterA(
            window_size=self.config.window_size,
            max_capacity_prompt=self.config.max_capacity_prompt,
            kernel_size=self.config.kernel_size,
            pooling=self.config.pooling,
            merge=self.config.merge,
        )

def init_pyramidkv_a(self, num_hidden_layers=None):
    """Inject the PyramidKV-A cluster (GQA-group-consensus selection, compact KV)."""
    if hasattr(self, "kv_cluster"):
        return
    if not hasattr(self.config, "window_size"):
        self.config.window_size = 128
    if not hasattr(self.config, "max_capacity_prompt"):
        self.config.max_capacity_prompt = 1024
    if not hasattr(self.config, "kernel_size"):
        self.config.kernel_size = 7
    if not hasattr(self.config, "pooling"):
        self.config.pooling = "avgpool"
    if not hasattr(self.config, "pyram_beta"):
        self.config.pyram_beta = 20
    if not hasattr(self.config, "pyramid_source_gqa_ratio"):
        self.config.pyramid_source_gqa_ratio = 1
    if not hasattr(self.config, "merge"):
        self.config.merge = None
    nlayers = num_hidden_layers
    if nlayers is None:
        nlayers = getattr(self.config, "num_hidden_layers", 32)
    self.kv_cluster = PyramidKVClusterA(
        num_hidden_layers=nlayers,
        layer_idx=getattr(self, "layer_idx", None),
        window_size=self.config.window_size,
        max_capacity_prompt=self.config.max_capacity_prompt,
        kernel_size=self.config.kernel_size,
        pooling=self.config.pooling,
        beta=self.config.pyram_beta,
        source_gqa_ratio=self.config.pyramid_source_gqa_ratio,
        merge=self.config.merge,
    )

def tidekv_compress_layer(
    layer,
    obs_hidden,
    full_k,
    full_v,
    budget,
    obs_len,
    model,
    kernel_size=7,
    pooling="avgpool",
    score_block_size=1024,
    tidekv_head_adaptive=False,
    tidekv_head_floor_alpha=0.5,
    tidekv_head_normalize=False,
    selection_mode="snapkv",
    chunk_length=20,
    preselected_chunk_indices=None,
    return_head_metadata=False,
    debug_info=None,
):

    seq_len = full_k.shape[2]
    if seq_len <= budget:
        if return_head_metadata:
            attn = layer.self_attn
            groups = attn.num_heads // full_k.shape[1]
            return pack_uniform_heads_flatten(
                full_k,
                full_v,
                num_key_value_groups=groups,
            )
        return full_k, full_v

    bsz = full_k.shape[0]
    num_kv_heads = full_k.shape[1]
    head_dim = full_k.shape[3]

    prefix_len = max(0, seq_len - obs_len)
    num_keep = budget - obs_len
    if num_keep <= 0:
        raise ValueError(
            f"tidekv_compress_layer: budget ({budget}) must be > obs_len ({obs_len}); "
            f"got num_keep={num_keep}. Increase base_budget or reduce obs_window_size."
        )
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

    if preselected_chunk_indices is not None:
        if selection_mode != "chunkkv":
            raise ValueError("preselected_chunk_indices requires selection_mode='chunkkv'")
        selected = preselected_chunk_indices.to(device=full_k.device, dtype=torch.long)
        if selected.ndim != 2 or selected.shape != (bsz, num_keep):
            raise ValueError(
                "preselected_chunk_indices must have shape "
                f"({bsz}, {num_keep}), got {tuple(selected.shape)}"
            )
        if selected.numel() and (
            int(selected.min().item()) < 0 or int(selected.max().item()) >= prefix_len
        ):
            raise ValueError("preselected_chunk_indices contains an out-of-range prefix index")
        sorted_prefix_indices = selected.unsqueeze(1).expand(bsz, num_kv_heads, -1)
        idx_expanded = sorted_prefix_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        k_prefix = full_k[:, :, :prefix_len, :].gather(dim=2, index=idx_expanded)
        v_prefix = full_v[:, :, :prefix_len, :].gather(dim=2, index=idx_expanded)
        out_k = torch.cat([k_prefix, full_k[:, :, -obs_len:, :]], dim=2)
        out_v = torch.cat([v_prefix, full_v[:, :, -obs_len:, :]], dim=2)
        if debug_info is not None:
            debug_info["selected_indices"] = sorted_prefix_indices.detach()
            debug_info["chunk_indices_reused"] = True
        return out_k, out_v

    attn = layer.self_attn
    normed_obs = layer.input_layernorm(obs_hidden)
    q_obs = attn.q_proj(normed_obs)
    num_attn_heads = attn.num_heads
    q_obs = q_obs.view(bsz, obs_len, num_attn_heads, head_dim).transpose(1, 2)

    device = q_obs.device
    obs_position_ids = torch.arange(
        seq_len - obs_len, seq_len, device=device, dtype=torch.long
    ).unsqueeze(0).expand(bsz, -1)
    _rotary_emb = getattr(model.model, "rotary_emb", None)
    if _rotary_emb is None:  # Mistral keeps RoPE per-layer
        _rotary_emb = model.model.layers[0].self_attn.rotary_emb
    cos, sin = _rotary_emb(obs_hidden, obs_position_ids)
    cos = cos.to(device)
    sin = sin.to(device)
    cos_unsq = cos.unsqueeze(1)
    sin_unsq = sin.unsqueeze(1)
    q_obs = (q_obs * cos_unsq) + (_rotate_half(q_obs) * sin_unsq)

    groups = num_attn_heads // num_kv_heads
    scale = 1.0 / math.sqrt(head_dim)
    qf = q_obs.float()

    # Pass 1: normalize over all keys, rather than independently within each block.
    log_z = torch.full(
        (bsz, num_attn_heads, obs_len), -float("inf"), device=device, dtype=torch.float32
    )
    for k_start in range(0, seq_len, score_block_size):
        k_end = min(k_start + score_block_size, seq_len)
        k_blk = full_k[:, :, k_start:k_end, :]
        if groups > 1:
            k_blk_q = k_blk.repeat_interleave(groups, dim=1)
        else:
            k_blk_q = k_blk
        logits_blk = torch.einsum("bhod,bhkd->bhok", qf, k_blk_q.float()) * scale
        key_positions = torch.arange(k_start, k_end, device=device, dtype=torch.long)
        causal_mask = build_causal_score_mask(
            obs_position_ids[0], key_positions, dtype=logits_blk.dtype
        )
        logits_blk = logits_blk + causal_mask.unsqueeze(0).unsqueeze(0)
        block_logz = torch.logsumexp(logits_blk, dim=-1)
        log_z = torch.logaddexp(log_z, block_logz)
        del k_blk, k_blk_q, logits_blk, block_logz, key_positions, causal_mask

    top_vals = None
    top_idx = None
    full_pooled = None
    need_streaming_topk = ((not tidekv_head_adaptive)) and (
        selection_mode != "chunkkv"
    )
    if tidekv_head_adaptive or selection_mode == "chunkkv":
        full_pooled = torch.zeros(
            bsz, num_kv_heads, prefix_len, device=device, dtype=torch.float32,
        )
    # Pass 2: score with pooling halos so block boundaries do not change selection.
    for center_start in range(0, prefix_len, score_block_size):
        center_end = min(center_start + score_block_size, prefix_len)
        pad = kernel_size // 2
        raw_start = max(0, center_start - pad)
        raw_end = min(prefix_len, center_end + pad)

        k_blk = full_k[:, :, raw_start:raw_end, :]
        if groups > 1:
            k_blk_q = k_blk.repeat_interleave(groups, dim=1)
        else:
            k_blk_q = k_blk

        logits_blk = torch.einsum("bhod,bhkd->bhok", qf, k_blk_q.float()) * scale
        key_positions = torch.arange(raw_start, raw_end, device=device, dtype=torch.long)
        causal_mask = build_causal_score_mask(
            obs_position_ids[0], key_positions, dtype=logits_blk.dtype
        )
        logits_blk = logits_blk + causal_mask.unsqueeze(0).unsqueeze(0)

        attn_blk = torch.exp(logits_blk - log_z.unsqueeze(-1))

        score_q = attn_blk.sum(dim=2)
        if groups > 1:
            rl = raw_end - raw_start
            score_kv = score_q.view(bsz, num_kv_heads, groups, rl).sum(dim=2)
        else:
            score_kv = score_q

        pooled = pool_center_block(
            score_kv,
            raw_start,
            raw_end,
            center_start,
            center_end,
            prefix_len,
            kernel_size,
            pooling,
        )

        if full_pooled is not None:
            full_pooled[:, :, center_start:center_end] = pooled.float()

        if need_streaming_topk:
            block_abs_idx = torch.arange(
                center_start, center_end, device=pooled.device, dtype=torch.long
            ).view(1, 1, -1).expand(bsz, num_kv_heads, -1)
            top_vals, top_idx = update_streaming_topk(
                top_vals, top_idx, pooled, block_abs_idx, k=num_keep
            )
        del k_blk, k_blk_q, logits_blk, attn_blk, score_q, score_kv, pooled
        del key_positions, causal_mask

    out_metadata = None
    sorted_prefix_indices = None
    head_keep = None
    if seq_len <= budget or prefix_len <= num_keep:
        if return_head_metadata:
            out_k, out_v, out_metadata = pack_uniform_heads_flatten(
                full_k,
                full_v,
                num_key_value_groups=groups,
            )
        else:
            out_k, out_v = full_k, full_v
        adaptive_head_lens = None
    else:
        adaptive_head_lens = None
        if tidekv_head_adaptive:
            sorted_scores, sorted_prefix_indices = full_pooled.sort(dim=-1, descending=True)
            head_keep = allocate_adaptive_head_keep(
                full_pooled,
                prefix_keep=num_keep,
                floor_alpha=tidekv_head_floor_alpha,
                normalize=tidekv_head_normalize,
                sorted_scores=sorted_scores,
            )
            if return_head_metadata:
                out_k, out_v, out_metadata = pack_adaptive_heads_flatten(
                    full_k,
                    full_v,
                    sorted_prefix_indices,
                    head_keep,
                    obs_len,
                    num_key_value_groups=groups,
                )
                adaptive_head_lens = out_metadata["head_lens"].detach().cpu().tolist()
            else:
                out_k, out_v, adaptive_head_lens = pack_adaptive_heads_padded(
                    full_k,
                    full_v,
                    sorted_prefix_indices,
                    head_keep,
                    obs_len,
                )
        elif selection_mode == "chunkkv":
            from tidekv_selectors import select_chunk_indices

            selected = select_chunk_indices(full_pooled, num_keep, chunk_length)
            sorted_prefix_indices = selected.unsqueeze(1).expand(
                bsz, num_kv_heads, -1
            )
            idx_expanded = sorted_prefix_indices.unsqueeze(-1).expand(
                -1, -1, -1, head_dim
            )
            k_prefix = full_k[:, :, :prefix_len, :].gather(dim=2, index=idx_expanded)
            v_prefix = full_v[:, :, :prefix_len, :].gather(dim=2, index=idx_expanded)
            k_obs = full_k[:, :, -obs_len:, :]
            v_obs = full_v[:, :, -obs_len:, :]
            out_k = torch.cat([k_prefix, k_obs], dim=2)
            out_v = torch.cat([v_prefix, v_obs], dim=2)
        else:
            indices = torch.sort(top_idx, dim=-1).values
            sorted_prefix_indices = indices
            idx_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_prefix = full_k[:, :, :prefix_len, :].gather(dim=2, index=idx_expanded)
            v_prefix = full_v[:, :, :prefix_len, :].gather(dim=2, index=idx_expanded)
            k_obs = full_k[:, :, -obs_len:, :]
            v_obs = full_v[:, :, -obs_len:, :]
            out_k = torch.cat([k_prefix, k_obs], dim=2)
            out_v = torch.cat([v_prefix, v_obs], dim=2)

    if full_pooled is not None:
        del full_pooled
    if return_head_metadata:
        if debug_info is not None:
            debug_info["selected_indices"] = (
                sorted_prefix_indices.detach() if sorted_prefix_indices is not None else None
            )
            debug_info["head_keep"] = head_keep.detach() if head_keep is not None else None
            debug_info["head_metadata"] = out_metadata
        return out_k, out_v, out_metadata
    if debug_info is not None:
        debug_info["selected_indices"] = (
            sorted_prefix_indices.detach()
            if sorted_prefix_indices is not None
            else (torch.sort(top_idx, dim=-1).values.detach() if top_idx is not None else None)
        )
    return out_k, out_v

def _rotate_half(x):

    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
