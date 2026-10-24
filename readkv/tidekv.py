"""Layer-major chunked prefill, post-layer KV selection, and greedy decode."""

import torch
from transformers.cache_utils import DynamicCache

from adakv.cache import DynamicCacheSplitHeadFlatten
from simple_utils import pack_uniform_heads_flatten, tidekv_compress_layer
from layer_budget import (
    effective_layer_budget,
    get_layer_budget,
    resolve_prefill_max_capacity,
)
from validation import validate_tidekv_parameters

_TIDEKV_ORIGINAL_LLAMA_FLASH_ATTN2_FORWARD = None
_TIDEKV_FLASH_ATTN_VARLEN_FUNC = None

def resolve_rotary_emb(model):
    """Return the RoPE module, model-agnostically.

    Llama (transformers >=4.43) exposes a model-level ``model.model.rotary_emb``; Mistral
    keeps it per-layer at ``layers[i].self_attn.rotary_emb``. Both take ``(x, position_ids)``
    and return ``(cos, sin)``, so TideKV's prefill/compress only needs the right object.
    """
    base = model.model
    rotary = getattr(base, "rotary_emb", None)
    if rotary is not None:
        return rotary
    return base.layers[0].self_attn.rotary_emb

def update_observation_tail(obs_hidden_tail, hidden_states, obs_window_size):
    """Append current layer-input hidden states and retain the true final O tokens."""
    if obs_window_size <= 0:
        raise ValueError(f"obs_window_size must be > 0, got {obs_window_size}")
    current = hidden_states[:, -obs_window_size:, :]
    if obs_hidden_tail is None:
        return current.clone()
    return torch.cat([obs_hidden_tail, current], dim=1)[:, -obs_window_size:, :].clone()

def _metadata_to_device(metadata, device):
    for key in ("head_lens", "cu_klen", "cu_offset", "cu_qlen"):
        if metadata[key].device != device:
            metadata[key] = metadata[key].to(device)
    return metadata

def advance_tidekv_head_metadata(metadata):
    """Advance flattened-cache metadata after one token is appended to every KV head."""
    metadata["head_lens"].add_(1)
    metadata["cu_klen"].add_(metadata["cu_offset"])
    metadata["max_seqlen_k"] = int(metadata["max_seqlen_k"]) + 1
    metadata["klen_sum"] = int(metadata.get("klen_sum", 0)) + metadata["head_lens"].numel()
    return metadata

def _apply_rotary_pos_emb_compat(query_states, key_states, cos, sin, position_ids):
    if not hasattr(_apply_rotary_pos_emb_compat, "_fn"):
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

        _apply_rotary_pos_emb_compat._fn = apply_rotary_pos_emb
    apply_rotary_pos_emb = _apply_rotary_pos_emb_compat._fn

    try:
        return apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    except TypeError:
        return apply_rotary_pos_emb(query_states, key_states, cos, sin)

def _tidekv_llama_flash_attn2_varlen_forward(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
    cache_position=None,
    position_embeddings=None,
    **kwargs,
):
    global _TIDEKV_FLASH_ATTN_VARLEN_FUNC
    if _TIDEKV_FLASH_ATTN_VARLEN_FUNC is None:
        from flash_attn import flash_attn_varlen_func

        _TIDEKV_FLASH_ATTN_VARLEN_FUNC = flash_attn_varlen_func

    if past_key_value is None or not getattr(past_key_value, "tidekv_varlen", False):
        raise RuntimeError("TideKV varlen forward requires a TideKV flattened cache")

    bsz, q_len, _ = hidden_states.size()
    if bsz != 1 or q_len != 1:
        raise NotImplementedError(
            "TideKV head-adaptive varlen decode currently supports batch_size=1 and q_len=1"
        )

    layer_idx = self.layer_idx
    metadata = _metadata_to_device(past_key_value.tidekv_head_metadata[layer_idx], hidden_states.device)

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = _apply_rotary_pos_emb_compat(
        query_states, key_states, cos, sin, position_ids
    )

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

    cache_kwargs = {
        "sin": sin,
        "cos": cos,
        "cache_position": cache_position,
        "head_lens": metadata["head_lens"],
        "cu_klen": metadata["cu_klen"],
    }
    key_states, value_states = past_key_value.update(
        key_states,
        value_states,
        layer_idx,
        cache_kwargs,
    )

    advance_tidekv_head_metadata(metadata)

    groups = int(metadata.get("num_key_value_groups", self.num_key_value_groups))
    if groups != self.num_key_value_groups:
        raise ValueError(
            f"TideKV metadata num_key_value_groups={groups} does not match layer "
            f"num_key_value_groups={self.num_key_value_groups}"
        )

    query_states = query_states.view(-1, groups, self.head_dim)
    key_states = key_states.view(-1, 1, self.head_dim)
    value_states = value_states.view(-1, 1, self.head_dim)

    attn_output = _TIDEKV_FLASH_ATTN_VARLEN_FUNC(
        query_states,
        key_states,
        value_states,
        metadata["cu_qlen"],
        metadata["cu_klen"],
        1,
        int(metadata["max_seqlen_k"]),
        causal=True,
    )
    attn_output = attn_output.reshape(bsz, self.num_heads, q_len, self.head_dim)
    attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)
    return attn_output, None, past_key_value

def install_tidekv_varlen_decode_patch():
    """
    Patch LlamaFlashAttention2 decode so TideKV can consume per-head flattened KV.

    The regular HuggingFace cache path assumes every KV head has the same sequence
    length. Head-adaptive TideKV violates that assumption, so decode must use
    flash-attn varlen with per-layer head_lens/cu_klen metadata.
    """
    global _TIDEKV_ORIGINAL_LLAMA_FLASH_ATTN2_FORWARD

    from transformers.models.llama.modeling_llama import LlamaFlashAttention2

    if getattr(LlamaFlashAttention2, "_readkv_tidekv_varlen_patched", False):
        return

    _TIDEKV_ORIGINAL_LLAMA_FLASH_ATTN2_FORWARD = LlamaFlashAttention2.forward

    def patched_forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        if getattr(past_key_value, "tidekv_varlen", False):
            return _tidekv_llama_flash_attn2_varlen_forward(
                self,
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return _TIDEKV_ORIGINAL_LLAMA_FLASH_ATTN2_FORWARD(
            self,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    LlamaFlashAttention2.forward = patched_forward
    LlamaFlashAttention2._readkv_tidekv_varlen_patched = True

def greedy_decode(model, past_key_values, next_token_id, eos_token_ids,
                  max_new_tokens, current_pos):

    if isinstance(eos_token_ids, int):
        eos_token_ids = {eos_token_ids}

    device = next_token_id.device
    generated_ids = [next_token_id.squeeze(-1)]

    # RoPE positions follow the original prompt length, not the retained cache size.
    for _ in range(max_new_tokens - 1):
        if next_token_id.item() in eos_token_ids:
            break
        position_ids = torch.tensor([[current_pos]], device=device, dtype=torch.long)
        outputs = model(
            input_ids=next_token_id,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)
        past_key_values = outputs.past_key_values
        generated_ids.append(next_token_id.squeeze(-1))
        current_pos += 1

    return torch.stack(generated_ids, dim=-1)

def tidekv_prefill(model, input_ids, max_capacity, chunk_size=512,
                   window_size=128, kernel_size=7, pooling='avgpool',
                   warmup_layers=0, obs_window_size=None, score_block_size=1024,
                   tidekv_head_adaptive=False,
                   tidekv_head_floor_alpha=0.5,
                   tidekv_head_normalize=False,
                   selection_mode="snapkv",
                   chunk_length=20,
                   chunkkv_reuse_layers=1,
                   attention_mask=None,
                   empty_cache_after_prefill: bool = True):
    """Finish each full causal layer before selecting its persistent KV cache."""
    if obs_window_size is None:
        obs_window_size = window_size

    seq_len = input_ids.shape[1]
    bsz = input_ids.shape[0]
    max_capacity = resolve_prefill_max_capacity(seq_len, max_capacity)
    chunkkv_reuse_layers = int(chunkkv_reuse_layers)
    if chunkkv_reuse_layers < 1:
        raise ValueError("chunkkv_reuse_layers must be >= 1")
    if selection_mode != "chunkkv" and chunkkv_reuse_layers != 1:
        raise ValueError(
            "chunkkv_reuse_layers is only valid with selection_mode='chunkkv'"
        )

    embed_tokens = model.model.embed_tokens
    layers = model.model.layers
    final_norm = model.model.norm
    _rotary_emb = resolve_rotary_emb(model)  # Llama: model-level; Mistral: per-layer
    # Llama decoder layers accept precomputed position_embeddings; Mistral does not.
    _pass_pos_emb = getattr(model.model, "rotary_emb", None) is not None
    lm_head = model.lm_head
    num_layers = len(layers)
    _impl = getattr(model.config, "attn_implementation", None) or getattr(
        model.config, "_attn_implementation", None
    )
    use_flash_attn = (_impl or "").lower() == "flash_attention_2"
    validate_tidekv_parameters(
        model=model,
        batch_size=bsz,
        seq_len=seq_len,
        num_layers=num_layers,
        max_capacity=max_capacity,
        chunk_size=chunk_size,
        score_block_size=score_block_size,
        obs_window_size=obs_window_size,
        kernel_size=kernel_size,
        pooling=pooling,
        warmup_layers=warmup_layers,
        tidekv_head_adaptive=tidekv_head_adaptive,
        tidekv_head_floor_alpha=tidekv_head_floor_alpha,
        selection_mode=selection_mode,
        chunk_length=chunk_length,
        attention_mask=attention_mask,
    )

    chunk_starts = list(range(0, seq_len, chunk_size))
    num_chunks = len(chunk_starts)
    last_chunk_idx = num_chunks - 1

    hidden_chunks = [None] * num_chunks
    if tidekv_head_adaptive:
        final_kv_cache = DynamicCacheSplitHeadFlatten()
        final_kv_cache.tidekv_varlen = True
        final_kv_cache.tidekv_head_metadata = [None] * num_layers
    else:
        final_kv_cache = DynamicCache()
    chunk_reuse_indices = None

    with torch.no_grad():
        # Finish the full causal layer before evicting any of its prompt keys.
        for layer_idx in range(num_layers):
            layer_budget = get_layer_budget(max_capacity, layer_idx)
            if not isinstance(max_capacity, int):
                layer_budget = effective_layer_budget(layer_budget, seq_len)
            layer = layers[layer_idx]
            layer_dev = layer.input_layernorm.weight.device

            prefix_k = None
            prefix_v = None
            prefix_kv_len = 0
            obs_hidden_tail = None
            compress_debug_info = None

            for ci, start in enumerate(chunk_starts):
                end = min(start + chunk_size, seq_len)
                chunk_len = end - start

                if layer_idx == 0:
                    h_input = embed_tokens(input_ids[:, start:end].to(layer_dev))
                else:
                    h_prev = hidden_chunks[ci]
                    h_input = h_prev.to(layer_dev)

                # Selection queries must be rebuilt from layer inputs, not outputs.
                obs_hidden_tail = update_observation_tail(
                    obs_hidden_tail, h_input, obs_window_size
                )

                chunk_pos_ids = torch.arange(
                    start, end, device=layer_dev, dtype=torch.long,
                ).unsqueeze(0).expand(bsz, -1)
                position_embeddings = _rotary_emb(h_input, chunk_pos_ids)

                if use_flash_attn:
                    causal_mask = None
                else:
                    q_positions = torch.arange(start, end, device=layer_dev)
                    if prefix_kv_len > 0:
                        k_positions = torch.cat([
                            torch.arange(0, prefix_kv_len, device=layer_dev),
                            torch.arange(start, end, device=layer_dev),
                        ])
                    else:
                        k_positions = torch.arange(start, end, device=layer_dev)
                    causal_mask = (k_positions.unsqueeze(0) > q_positions.unsqueeze(1))
                    causal_mask = causal_mask.float() * torch.finfo(h_input.dtype).min
                    causal_mask = causal_mask.to(h_input.dtype).unsqueeze(0).unsqueeze(0)
                    if bsz > 1:
                        causal_mask = causal_mask.expand(bsz, 1, -1, -1)

                layer_cache = DynamicCache()
                if prefix_k is not None:
                    layer_cache.key_cache = [torch.empty(0)] * layer_idx + [prefix_k]
                    layer_cache.value_cache = [torch.empty(0)] * layer_idx + [prefix_v]
                    layer_cache._seen_tokens = prefix_kv_len
                else:
                    layer_cache.key_cache = [torch.empty(0)] * layer_idx
                    layer_cache.value_cache = [torch.empty(0)] * layer_idx

                cache_position = torch.arange(start, end, dtype=torch.long, device=layer_dev)

                layer_kwargs = dict(
                    hidden_states=h_input,
                    attention_mask=causal_mask,
                    position_ids=chunk_pos_ids,
                    past_key_value=layer_cache,
                    output_attentions=False,
                    use_cache=True,
                    cache_position=cache_position,
                )
                # Llama (>=4.43) takes precomputed position_embeddings; Mistral computes RoPE
                # internally from position_ids and rejects the kwarg. _pass_pos_emb gates it.
                if _pass_pos_emb:
                    layer_kwargs["position_embeddings"] = position_embeddings
                layer_outputs = layer(**layer_kwargs)

                h_output = layer_outputs[0]
                prefix_k = layer_cache.key_cache[layer_idx]
                prefix_v = layer_cache.value_cache[layer_idx]
                prefix_kv_len = prefix_k.shape[2]

                hidden_chunks[ci] = h_output

                del h_input, layer_outputs, layer_cache, position_embeddings, chunk_pos_ids, cache_position
                if causal_mask is not None:
                    del causal_mask

            full_k = prefix_k
            full_v = prefix_v

            obs_len = int(obs_hidden_tail.shape[1])
            layer_device = next(layer.parameters()).device
            need_compress = layer_idx >= warmup_layers and full_k.shape[2] > layer_budget

            if layer_idx < warmup_layers:
                if tidekv_head_adaptive:
                    compressed_k, compressed_v, head_metadata = pack_uniform_heads_flatten(
                        full_k,
                        full_v,
                        num_key_value_groups=layer.self_attn.num_key_value_groups,
                    )
                else:
                    compressed_k = full_k
                    compressed_v = full_v
            elif need_compress:
                reuse_chunk_indices = None
                needs_chunk_reuse_trace = (
                    selection_mode == "chunkkv"
                    and chunkkv_reuse_layers > 1
                )
                if needs_chunk_reuse_trace:
                    if layer_idx % chunkkv_reuse_layers:
                        reuse_chunk_indices = chunk_reuse_indices
                    else:
                        chunk_reuse_indices = None
                compress_debug_info = {} if needs_chunk_reuse_trace else None
                compress_out = tidekv_compress_layer(
                    layer=layer,
                    obs_hidden=obs_hidden_tail.to(layer_device),
                    full_k=full_k,
                    full_v=full_v,
                    budget=layer_budget,
                    obs_len=obs_len,
                    model=model,
                    kernel_size=kernel_size,
                    pooling=pooling,
                    score_block_size=score_block_size,
                    tidekv_head_adaptive=tidekv_head_adaptive,
                    tidekv_head_floor_alpha=tidekv_head_floor_alpha,
                    tidekv_head_normalize=tidekv_head_normalize,
                    selection_mode=selection_mode,
                    chunk_length=chunk_length,
                    preselected_chunk_indices=reuse_chunk_indices,
                    return_head_metadata=tidekv_head_adaptive,
                    debug_info=compress_debug_info,
                )
                if tidekv_head_adaptive:
                    compressed_k, compressed_v, head_metadata = compress_out
                else:
                    compressed_k, compressed_v = compress_out
                if needs_chunk_reuse_trace and reuse_chunk_indices is None:
                    selected = compress_debug_info.get("selected_indices")
                    if selected is not None:
                        chunk_reuse_indices = selected[:, 0, :].detach()
                if compressed_k is not full_k:
                    del full_k, full_v
            else:
                if tidekv_head_adaptive:
                    compressed_k, compressed_v, head_metadata = pack_uniform_heads_flatten(
                        full_k,
                        full_v,
                        num_key_value_groups=layer.self_attn.num_key_value_groups,
                    )
                else:
                    compressed_k = full_k
                    compressed_v = full_v

            while len(final_kv_cache.key_cache) <= layer_idx:
                final_kv_cache.key_cache.append(None)
                final_kv_cache.value_cache.append(None)
            final_kv_cache.key_cache[layer_idx] = compressed_k
            final_kv_cache.value_cache[layer_idx] = compressed_v
            if tidekv_head_adaptive:
                final_kv_cache.tidekv_head_metadata[layer_idx] = head_metadata
                final_kv_cache._seen_tokens = 1
            else:
                final_kv_cache._seen_tokens = compressed_k.shape[2]

            del compress_debug_info
            del prefix_k, prefix_v, obs_hidden_tail

        final_kv_cache._seen_tokens = (
            1 if tidekv_head_adaptive else final_kv_cache.key_cache[0].shape[2]
        )

        last_h = hidden_chunks[-1][:, -1:, :].to(final_norm.weight.device)
        last_h = final_norm(last_h)
        last_logits = lm_head(last_h[:, -1, :])

    del hidden_chunks, last_h
    if empty_cache_after_prefill:
        torch.cuda.empty_cache()

    return final_kv_cache, last_logits
