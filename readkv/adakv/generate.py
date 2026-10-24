"""Greedy generation for AdaKV's flattened, per-head cache."""

from __future__ import annotations

import torch

def adakv_generate(model, input_ids, max_new_tokens, eos_token_ids):
    """
    Prefill + decode via ``model()`` with patched ``prepare_inputs_for_generation``.

    AdaKV requires ``LlamaModel``/``MistralModel`` adaptive forward (last-token hidden only)
    and flatten KV cache; do not call ``model.model()`` directly.
    """
    device = input_ids.device
    prompt_len = input_ids.shape[1]
    attention_mask = torch.ones_like(input_ids, device=device)

    past_key_values = None
    cache_position = torch.arange(0, prompt_len, device=device, dtype=torch.long)

    model_inputs = model.prepare_inputs_for_generation(
        input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        cache_position=cache_position,
        use_cache=True,
    )
    outputs = model(**model_inputs)
    past_key_values = outputs.past_key_values
    next_token_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    generated_ids = [next_token_id.squeeze(-1)]
    cur_pos = prompt_len

    for _ in range(max_new_tokens - 1):
        if next_token_id.item() in eos_token_ids:
            break
        cache_position = torch.tensor([cur_pos], device=device, dtype=torch.long)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=device, dtype=attention_mask.dtype)],
            dim=-1,
        )
        model_inputs = model.prepare_inputs_for_generation(
            next_token_id,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            cache_position=cache_position,
            use_cache=True,
        )
        outputs = model(**model_inputs)
        next_token_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        past_key_values = outputs.past_key_values
        generated_ids.append(next_token_id.squeeze(-1))
        cur_pos += 1

    return torch.stack(generated_ids, dim=-1), past_key_values
