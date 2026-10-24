"""Unified block prompt processing generation engine."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from .cache_state import BPPCacheState, build_model_positions
from .capture import QueryCapture
from .policies.base import CompressionContext

@dataclass
class BPPGenerationOutput:
    generated_ids: torch.Tensor
    state: BPPCacheState

class BPPGenerationEngine:
    """Run one batch-size-1 greedy BPP generation request."""

    def __init__(self, model, spec, policy) -> None:
        self.model = model
        self.spec = spec
        self.policy = policy

    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor) -> tuple[BPPCacheState, torch.Tensor]:
        if input_ids.shape[0] != 1:
            raise ValueError("BPPGenerationEngine currently supports batch_size=1")

        state = BPPCacheState(
            num_layers=len(self.model.model.layers),
            model_config=getattr(self.model, "config", None),
        )
        self.policy.reset(len(self.model.model.layers))
        last_hidden = None

        for start in range(0, input_ids.shape[1], self.spec.block_size):
            end = min(start + self.spec.block_size, input_ids.shape[1])
            block = input_ids[:, start:end]
            absolute_pos, physical_cache_pos = build_model_positions(
                seen_tokens=state.seen_tokens,
                physical_cache_length=state.physical_cache_length(),
                query_length=block.shape[1],
                device=block.device,
            )
            self._assert_position_contract(state, absolute_pos, physical_cache_pos)
            capture = QueryCapture(self.model) if self.policy.requires_query_capture else nullcontext()
            with capture:
                outputs = self.model.model(
                    input_ids=block,
                    past_key_values=state.cache,
                    position_ids=absolute_pos.unsqueeze(0),
                    cache_position=physical_cache_pos,
                    use_cache=True,
                    return_dict=True,
                )
            state.cache = outputs.past_key_values
            last_hidden = outputs.last_hidden_state[:, -1, :]

            self._compress_all_layers(state, capture, absolute_pos)
            state.seen_tokens = end
            state.assert_budget(self._active_budget())

        if last_hidden is None:
            raise ValueError("input_ids must contain at least one token")
        first_token = self.model.lm_head(last_hidden).argmax(dim=-1, keepdim=True)
        return state, first_token

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_ids: set[int] | list[int] | tuple[int, ...],
    ) -> BPPGenerationOutput:
        eos_ids = set(int(token_id) for token_id in eos_token_ids)
        state, token = self.prefill(input_ids)
        generated: list[torch.Tensor] = []

        for step in range(max_new_tokens):
            generated.append(token)
            token_id = int(token.item())
            if token_id in eos_ids or step == max_new_tokens - 1:
                break
            token = self._decode_one(state, token)

        generated_ids = torch.cat(generated, dim=1)
        return BPPGenerationOutput(generated_ids=generated_ids, state=state)

    def _decode_one(self, state: BPPCacheState, token: torch.Tensor) -> torch.Tensor:
        absolute_pos, physical_cache_pos = build_model_positions(
            seen_tokens=state.seen_tokens,
            physical_cache_length=state.physical_cache_length(),
            query_length=1,
            device=token.device,
        )
        self._assert_position_contract(state, absolute_pos, physical_cache_pos)
        capture = QueryCapture(self.model) if self.policy.requires_query_capture else nullcontext()
        with capture:
            outputs = self.model.model(
                input_ids=token,
                past_key_values=state.cache,
                position_ids=absolute_pos.unsqueeze(0),
                cache_position=physical_cache_pos,
                use_cache=True,
                return_dict=True,
            )
        state.cache = outputs.past_key_values
        next_hidden = outputs.last_hidden_state[:, -1, :]
        next_token = self.model.lm_head(next_hidden).argmax(dim=-1, keepdim=True)

        self._compress_all_layers(state, capture, absolute_pos)
        state.seen_tokens += 1
        state.assert_budget(self._active_budget())
        return next_token

    def _compress_all_layers(self, state: BPPCacheState, capture, absolute_pos: torch.Tensor) -> None:
        for layer_idx, layer in enumerate(self.model.model.layers):
            cache_view = state.layer(layer_idx)
            state.append_positions(
                layer_idx=layer_idx,
                block_positions=absolute_pos,
                batch=cache_view.keys.shape[0],
                kv_heads=cache_view.keys.shape[1],
            )
            query, query_pos = self._prepare_policy_query(layer_idx, capture, absolute_pos)
            ctx = CompressionContext(
                layer_idx=layer_idx,
                keys=cache_view.keys,
                values=cache_view.values,
                positions=state.positions[layer_idx],
                budget=self._active_budget(),
                query_states=query,
                query_positions=query_pos,
                num_key_value_groups=getattr(layer.self_attn, "num_key_value_groups", 1),
            )
            result = self.policy.compress(ctx)
            state.replace_layer(layer_idx, result)

    def _prepare_policy_query(
        self,
        layer_idx: int,
        capture,
        absolute_pos: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self.policy.requires_query_capture:
            return None, None
        raw_query = capture.outputs[layer_idx]
        if raw_query is None:
            raise RuntimeError(f"missing q_proj capture for layer {layer_idx}")
        attention = self.model.model.layers[layer_idx].self_attn
        batch, seq_len, _ = raw_query.shape
        query_heads = attention.config.num_attention_heads
        head_dim = attention.head_dim
        query = raw_query.view(batch, seq_len, query_heads, head_dim).transpose(1, 2)
        position_ids = absolute_pos.unsqueeze(0)
        cos, sin = self.model.model.rotary_emb(raw_query, position_ids=position_ids)
        query, _ = apply_rotary_pos_emb(query, query, cos, sin)
        return query.contiguous(), absolute_pos

    def _active_budget(self) -> int:
        if self.spec.name == "none_bpp":
            return 10**12
        return self.spec.kv_budget

    @staticmethod
    def _assert_position_contract(
        state: BPPCacheState,
        absolute_pos: torch.Tensor,
        physical_cache_pos: torch.Tensor,
    ) -> None:
        physical_length = state.physical_cache_length()
        assert absolute_pos.shape == physical_cache_pos.shape
        assert int(absolute_pos[0]) == state.seen_tokens
        assert int(physical_cache_pos[0]) == physical_length
        if state.seen_tokens > physical_length:
            assert int(absolute_pos[0]) != int(physical_cache_pos[0])
