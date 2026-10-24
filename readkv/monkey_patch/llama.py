import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb,repeat_kv
from transformers.cache_utils import Cache
from transformers.utils import logging
import os
import sys
import warnings
from typing import Optional, Tuple
import math

logger = logging.get_logger(__name__)

sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/..')
from simple_utils import (
    grouped_query_attention_weights,
    init_h2o,
    init_StreamingLLM,
    init_snapkv_a,
    init_pyramidkv_a,
    init_chunkkv,
)

# Conditional import for flash_attn to avoid compatibility issues
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    flash_attn_func = None
    flash_attn_varlen_func = None
    index_first_axis = None
    pad_input = None
    unpad_input = None

def _flash_attention_forward(
    self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
):
    """
    Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
    first unpad the input, then computes the attention scores and pad the final attention scores.

    Args:
        query_states (`torch.Tensor`):
            Input query states to be passed to Flash Attention API
        key_states (`torch.Tensor`):
            Input key states to be passed to Flash Attention API
        value_states (`torch.Tensor`):
            Input value states to be passed to Flash Attention API
        attention_mask (`torch.Tensor`):
            The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
            position of padding tokens and 1 for the position of non-padding tokens.
        dropout (`float`):
            Attention dropout
        softmax_scale (`float`, *optional*):
            The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
    """
    if not FLASH_ATTN_AVAILABLE:
        raise ImportError("flash_attn is not available. Please use 'sdpa' or 'eager' attention implementation.")
    if not self._flash_attn_uses_top_left_mask:
        causal = self.is_causal
    else:
        # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
        causal = self.is_causal and query_length != 1

    # Contains at least one padding token in the sequence
    if attention_mask is not None:
        batch_size = query_states.shape[0]
        query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
            query_states, key_states, value_states, attention_mask, query_length
        )

        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

        attn_output_unpad = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_k,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
        )

        attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
    else:
        attn_output = flash_attn_func(
            query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
        )

    # if self.layer_idx == 0:
    #     import pdb; pdb.set_trace()

    return attn_output

def llama_attn_forward_H2O(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()
        init_h2o(self)

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # 3. RoPE
        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
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

        # 6. Expand KV to num_attention_heads for attention computation
        current_k_full = repeat_kv(current_k, self.num_key_value_groups)
        current_v_full = repeat_kv(current_v, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, current_k_full.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            mask_len = attention_mask.shape[-1]
            if mask_len == total_len:
                attn_weights = attn_weights + attention_mask
            elif q_len == 1:

                pass
            else:

                causal_mask = torch.full(
                    (q_len, total_len), float("-inf"),
                    device=attn_weights.device, dtype=attn_weights.dtype
                )
                causal_mask = torch.triu(causal_mask, diagonal=1 + total_len - q_len)
                attn_weights = attn_weights + causal_mask[None, None, :, :]

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, current_v_full)
        del current_k_full, current_v_full

        if need_h2o:

            num_kv_heads = current_k.shape[1]
            attn_weights_for_h2o = attn_weights.view(
                bsz, num_kv_heads, self.num_key_value_groups, q_len, total_len
            ).mean(dim=2)  # (bsz, num_kv_heads, q_len, total_len)

            compressed_k, compressed_v = self.kv_cluster.evict_and_compress(
                current_k, current_v, attn_weights_for_h2o
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

        # 11. Output projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

def llama_flash_attn2_forward_StreamingLLM(
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
        query_states, key_states, cos, sin, position_ids
    )

    if past_key_value is not None:
        if q_len > 1:

            # Prefill and decode both attend to the retained sink/recent cache.
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

            # Cache stores compressed RoPE'd KV for decode consistency.
            if len(past_key_value.key_cache) <= self.layer_idx:
                past_key_value.key_cache.append(k_cache)
                past_key_value.value_cache.append(v_cache)
            else:
                past_key_value.key_cache[self.layer_idx] = k_cache
                past_key_value.value_cache[self.layer_idx] = v_cache
            if self.layer_idx == 0:
                past_key_value._seen_tokens += q_len

        else:

            history_k = past_key_value.key_cache[self.layer_idx]
            history_v = past_key_value.value_cache[self.layer_idx]

            # Concat current RoPE'd K/V with history
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
        # No past_key_value: standard forward
        attn_query = query_states
        attn_key = key_states
        attn_value = value_states

    # FlashAttention requires [batch_size, sequence_length, num_heads, head_dim]
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

def llama_flash_attn2_forward_H2O(
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

    # Reshape
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # RoPE
    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    
    # GQA: Keep KV at num_key_value_heads for cache operations to avoid 4x memory waste.
    # repeat_kv is applied AFTER cache/compression for FlashAttention.

    # [H2O] Reset accumulated attention scores at the start of each new sequence.
    # accumulated_scores persists on the cluster instance across samples; stale scores
    # cause incorrect eviction decisions on subsequent samples.
    if past_key_value is not None and not (len(past_key_value.key_cache) > self.layer_idx):
        self.kv_cluster.reset()

    if past_key_value is not None and len(past_key_value.key_cache) > self.layer_idx:
        past_k = past_key_value.key_cache[self.layer_idx]   # num_key_value_heads
        past_v = past_key_value.value_cache[self.layer_idx]
        current_k = torch.cat([past_k, key_states], dim=2)
        current_v = torch.cat([past_v, value_states], dim=2)
    else:
        current_k = key_states   # num_key_value_heads
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

    attn_output = flash_attn_func(
        fa_query,
        fa_key,
        fa_value,
        dropout_p=self.attention_dropout if self.training else 0.0,
        softmax_scale=None,
        causal=True,
        return_attn_probs=False
    )
    del current_k_full, current_v_full

    if need_h2o:
        attn_weights = grouped_query_attention_weights(
            query_states, current_k, self.num_key_value_groups, causal=True
        )

        compressed_k, compressed_v = self.kv_cluster.evict_and_compress(
            current_k, 
            current_v, 
            attn_weights
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


def _llama_flash_attn2_forward_grouped_consensus(
    self, hidden_states, attention_mask, position_ids, past_key_value,
    output_attentions, use_cache, **kwargs,
):
    """Shared SnapKV-A / PyramidKV-A forward.

    Identical KV-cache flow to the SnapKV/PyramidKV forwards EXCEPT it always stores
    a KV-head-compact cache and passes the FULL (query-head) query to the cluster, so
    the cluster does the GQA-group-consensus selection (see SnapKVClusterA /
    PyramidKVClusterA). ``self.kv_cluster`` must already be initialized by the caller.
    """
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. "
            "Please make sure use `attention_mask` instead.`"
        )
        attention_mask = kwargs.pop("padding_mask")
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
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    # SnapKV-A / PyramidKV-A store COMPACT (KV-head) cache: no pre-repeat of K/V.

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}
        if key_states.shape[-2] == kv_seq_len:  # prefill: group-consensus compress + compact store
            self.kv_seq_len = kv_seq_len
            # full query-head query is passed; the cluster pools across the GQA group
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states, query_states, value_states, attention_mask, self.num_key_value_groups
            )
            past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
        else:  # decode: append the new (compact) token
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        past_key_value._seen_tokens = self.kv_seq_len

    # repeat AFTER cache/compression for FlashAttention (prefill attends full context;
    # decode attends the compressed compact cache)
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

def llama_flash_attn2_forward_SnapKV_A(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """SnapKV-A: SnapKV with TideKV-style GQA-group-consensus selection (compact KV)."""
    init_snapkv_a(self)
    return _llama_flash_attn2_forward_grouped_consensus(
        self, hidden_states, attention_mask, position_ids, past_key_value,
        output_attentions, use_cache, **kwargs,
    )

def _llama_non_fa_attn_forward_grouped_consensus(
    self,
    hidden_states,
    attention_mask,
    position_ids,
    past_key_value,
    output_attentions,
    use_cache,
    *,
    backend,
    cache_position=None,
    position_embeddings=None,
    **kwargs,
):
    """Matched SnapKV-A cache flow for Llama eager and SDPA backends."""
    if getattr(self.config, "pretraining_tp", 1) != 1:
        raise NotImplementedError("SnapKV-A eager/SDPA requires pretraining_tp=1")
    if backend not in {"eager", "sdpa"}:
        raise ValueError(f"unsupported non-FA backend: {backend}")
    if backend == "sdpa" and output_attentions:
        backend = "eager"

    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError("SnapKV-A caching requires a layer index")
        cache_has_data = len(past_key_value.key_cache) > self.layer_idx
        if not cache_has_data:
            self.kv_seq_len = 0
        if getattr(self, "kv_seq_len", 0) != 0:
            kv_seq_len += self.kv_seq_len
        else:
            kv_seq_len += past_key_value.get_usable_length(
                key_states.shape[-2], self.layer_idx
            )

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin
    )

    if past_key_value is not None:
        cache_kwargs = {
            "sin": sin,
            "cos": cos,
            "cache_position": cache_position,
        }
        if key_states.shape[-2] == kv_seq_len:
            self.kv_seq_len = kv_seq_len
            compressed_k, compressed_v = self.kv_cluster.update_kv(
                key_states,
                query_states,
                value_states,
                attention_mask,
                self.num_key_value_groups,
            )
            past_key_value.update(
                compressed_k, compressed_v, self.layer_idx, cache_kwargs
            )
        else:
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        past_key_value._seen_tokens = self.kv_seq_len

    # Prefill attends the full K/V tensors above while the cache stores only the
    # compact selection. Decode uses the compact cache returned by update().
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    causal_mask = attention_mask
    if causal_mask is not None:
        causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

    attn_weights = None
    if backend == "sdpa":
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=causal_mask is None and q_len > 1,
        )
    else:
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)
        if causal_mask is not None:
            attn_weights = attn_weights + causal_mask
        attn_weights = F.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = F.dropout(
            attn_weights,
            p=self.attention_dropout,
            training=self.training,
        )
        attn_output = torch.matmul(attn_weights, value_states)

    expected_shape = (bsz, self.num_heads, q_len, self.head_dim)
    if attn_output.size() != expected_shape:
        raise ValueError(
            f"attention output should have shape {expected_shape}, got {attn_output.size()}"
        )
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)
    if not output_attentions:
        attn_weights = None
    return attn_output, attn_weights, past_key_value

def llama_sdpa_attn_forward_SnapKV_A(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
):
    init_snapkv_a(self)
    return _llama_non_fa_attn_forward_grouped_consensus(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_value,
        output_attentions,
        use_cache,
        backend="sdpa",
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )

def llama_eager_attn_forward_SnapKV_A(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
):
    init_snapkv_a(self)
    return _llama_non_fa_attn_forward_grouped_consensus(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_value,
        output_attentions,
        use_cache,
        backend="eager",
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )

def llama_flash_attn2_forward_PyramidKV_A(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """PyramidKV-A: PyramidKV with TideKV-style GQA-group-consensus selection (compact KV)."""
    init_pyramidkv_a(self, num_hidden_layers=self.config.num_hidden_layers)
    return _llama_flash_attn2_forward_grouped_consensus(
        self, hidden_states, attention_mask, position_ids, past_key_value,
        output_attentions, use_cache, **kwargs,
    )


def llama_flash_attn2_forward_ChunkKV(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """ChunkKV: SnapKV scores + chunk-wise KV selection (kvpress ChunkKVPress)."""
    init_chunkkv(self)
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. "
            "Please make sure use `attention_mask` instead.`"
        )
        attention_mask = kwargs.pop("padding_mask")

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
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

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

