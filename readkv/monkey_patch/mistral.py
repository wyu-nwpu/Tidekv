import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
from typing import Optional, Tuple

from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb, repeat_kv
from transformers.cache_utils import Cache
from transformers.utils import logging

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
from simple_utils import (
    grouped_query_attention_weights,
    init_h2o,
    init_StreamingLLM,
    init_snapkv_a,
    init_pyramidkv_a,
    init_chunkkv,
)

from .llama import _flash_attention_forward

# Conditional import for flash_attn
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    flash_attn_func = None

logger = logging.get_logger(__name__)

# H2O (Flash Attention 2)
def mistral_flash_attn2_forward_H2O(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

    init_h2o(self)
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None and not (len(past_key_value.key_cache) > self.layer_idx):
        self.kv_cluster.reset()

    if past_key_value is not None and len(past_key_value.key_cache) > self.layer_idx:
        past_k = past_key_value.key_cache[self.layer_idx]
        past_v = past_key_value.value_cache[self.layer_idx]
        current_k = torch.cat([past_k, key_states], dim=2)
        current_v = torch.cat([past_v, value_states], dim=2)
    else:
        current_k = key_states
        current_v = value_states

    total_len = current_k.shape[2]
    need_h2o = use_cache and (total_len > self.kv_cluster.max_capacity_prompt)

    # Expand KV to num_attention_heads for FlashAttention
    current_k_full = repeat_kv(current_k, self.num_key_value_groups)
    current_v_full = repeat_kv(current_v, self.num_key_value_groups)

    fa_query = query_states.transpose(1, 2)
    fa_key = current_k_full.transpose(1, 2)
    fa_value = current_v_full.transpose(1, 2)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        target_dtype = torch.float16
        fa_query = fa_query.to(target_dtype)
        fa_key = fa_key.to(target_dtype)
        fa_value = fa_value.to(target_dtype)

    # 4. FlashAttention
    attn_output = flash_attn_func(
        fa_query, fa_key, fa_value,
        dropout_p=self.attention_dropout if self.training else 0.0,
        softmax_scale=None,
        causal=True,
        return_attn_probs=False,
    )
    del current_k_full, current_v_full

    if need_h2o:
        attn_weights = grouped_query_attention_weights(
            query_states, current_k, self.num_key_value_groups, causal=True
        )

        compressed_k, compressed_v = self.kv_cluster.evict_and_compress(
            current_k, current_v, attn_weights
        )

        if past_key_value is not None:
            if len(past_key_value.key_cache) <= self.layer_idx:
                past_key_value.key_cache.append(compressed_k)
                past_key_value.value_cache.append(compressed_v)
            else:
                past_key_value.key_cache[self.layer_idx] = compressed_k
                past_key_value.value_cache[self.layer_idx] = compressed_v

    elif use_cache and past_key_value is not None:
        if len(past_key_value.key_cache) <= self.layer_idx:
            past_key_value.key_cache.append(current_k)
            past_key_value.value_cache.append(current_v)
        else:
            past_key_value.key_cache[self.layer_idx] = current_k
            past_key_value.value_cache[self.layer_idx] = current_v
    if (
        past_key_value is not None
        and self.layer_idx == 0
    ):
        past_key_value._seen_tokens += q_len

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value

# StreamingLLM (Flash Attention 2) — with position re-indexing
def mistral_flash_attn2_forward_StreamingLLM(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

    init_StreamingLLM(self)
    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin)

    if past_key_value is not None:
        if q_len > 1:
            # === Prefill ===
            if len(past_key_value.key_cache) > self.layer_idx:
                history_k = past_key_value.key_cache[self.layer_idx]
                history_v = past_key_value.value_cache[self.layer_idx]
                full_k = torch.cat([history_k, key_states], dim=-2)
                full_v = torch.cat([history_v, value_states], dim=-2)
                k_cache, v_cache = self.kv_cluster.compress_kv(full_k, full_v)
                attn_query = query_states
                attn_key = repeat_kv(k_cache, self.num_key_value_groups)
                attn_value = repeat_kv(v_cache, self.num_key_value_groups)
            else:
                full_k, full_v = key_states, value_states
                attn_query = query_states
                k_cache, v_cache = self.kv_cluster.compress_kv(full_k, full_v)
                attn_key = repeat_kv(k_cache, self.num_key_value_groups)
                attn_value = repeat_kv(v_cache, self.num_key_value_groups)

            # Cache compressed RoPE'd KV for decode consistency.
            if len(past_key_value.key_cache) <= self.layer_idx:
                past_key_value.key_cache.append(k_cache)
                past_key_value.value_cache.append(v_cache)
            else:
                past_key_value.key_cache[self.layer_idx] = k_cache
                past_key_value.value_cache[self.layer_idx] = v_cache
            if self.layer_idx == 0:
                past_key_value._seen_tokens += q_len

        else:
            # === Decode ===
            history_k = past_key_value.key_cache[self.layer_idx]
            history_v = past_key_value.value_cache[self.layer_idx]

            full_k = torch.cat([history_k, key_states], dim=-2)
            full_v = torch.cat([history_v, value_states], dim=-2)

            k_compressed, v_compressed = self.kv_cluster.compress_kv(full_k, full_v)

            past_key_value.key_cache[self.layer_idx] = k_compressed
            past_key_value.value_cache[self.layer_idx] = v_compressed
            if self.layer_idx == 0:
                past_key_value._seen_tokens += 1

            attn_query = query_states
            attn_key = repeat_kv(k_compressed, self.num_key_value_groups)
            attn_value = repeat_kv(v_compressed, self.num_key_value_groups)
    else:
        attn_query = query_states
        attn_key = key_states
        attn_value = value_states

    # Transpose for FlashAttention [batch, seq, head, dim]
    attn_query = attn_query.transpose(1, 2)
    attn_key = attn_key.transpose(1, 2)
    attn_value = attn_value.transpose(1, 2)

    input_dtype = attn_query.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        attn_query = attn_query.to(target_dtype)
        attn_key = attn_key.to(target_dtype)
        attn_value = attn_value.to(target_dtype)

    attn_output = _flash_attention_forward(
        self, attn_query, attn_key, attn_value,
        attention_mask, q_len,
        dropout=self.attention_dropout if self.training else 0.0,
    )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value

# SnapKV (Flash Attention 2)

def _mistral_flash_attn2_forward_grouped_consensus(
    self, hidden_states, attention_mask, position_ids, past_key_value,
    output_attentions, use_cache, **kwargs,
):
    """Shared Mistral SnapKV-A / PyramidKV-A forward (GQA-group-consensus, compact KV).

    Same as the Llama -A forward but with Mistral's RoPE call
    (``apply_rotary_pos_emb(q, k, cos, sin)`` without position_ids).
    """
    output_attentions = False
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        cache_has_data = (len(past_key_value.key_cache) > self.layer_idx)
        if not cache_has_data:
            self.kv_seq_len = 0
        if hasattr(self, "kv_seq_len"):
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    # SnapKV-A / PyramidKV-A store COMPACT (KV-head) cache: no pre-repeat of K/V.

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}
        if key_states.shape[-2] == kv_seq_len:  # prefill: group-consensus compress + compact store
            self.kv_seq_len = kv_seq_len
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states, query_states, value_states, attention_mask, self.num_key_value_groups
            )
            past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
        else:  # decode
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        past_key_value._seen_tokens = self.kv_seq_len

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    dropout_rate = self.attention_dropout if self.training else 0.0
    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype
        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    attn_output = _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
    )
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None, past_key_value

def mistral_flash_attn2_forward_SnapKV_A(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """SnapKV-A for Mistral (GQA-group-consensus selection, compact KV)."""
    init_snapkv_a(self)
    return _mistral_flash_attn2_forward_grouped_consensus(
        self, hidden_states, attention_mask, position_ids, past_key_value,
        output_attentions, use_cache, **kwargs,
    )

def mistral_flash_attn2_forward_PyramidKV_A(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """PyramidKV-A for Mistral (GQA-group-consensus selection, compact KV)."""
    init_pyramidkv_a(self, num_hidden_layers=self.config.num_hidden_layers)
    return _mistral_flash_attn2_forward_grouped_consensus(
        self, hidden_states, attention_mask, position_ids, past_key_value,
        output_attentions, use_cache, **kwargs,
    )


def mistral_flash_attn2_forward_ChunkKV(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """ChunkKV for Mistral (same flow as SnapKV / llama ChunkKV)."""
    init_chunkkv(self)
    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        cache_has_data = len(past_key_value.key_cache) > self.layer_idx
        if not cache_has_data:
            self.kv_seq_len = 0
        if hasattr(self, "kv_seq_len"):
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}
        if key_states.shape[-2] == kv_seq_len:
            self.kv_seq_len = kv_seq_len
            query_for_chunk = query_states
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states, query_for_chunk, value_states, attention_mask, self.num_key_value_groups
            )
            past_key_value.update(
                key_states_compress, value_states_compress, self.layer_idx, cache_kwargs
            )
        else:
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        past_key_value._seen_tokens = self.kv_seq_len

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    dropout_rate = self.attention_dropout if self.training else 0.0
    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype
        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    attn_output = _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
    )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

