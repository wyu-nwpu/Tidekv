"""Install AdaKV class-level monkey patches (Llama + Mistral)."""

from __future__ import annotations

import transformers

from adakv.llama_hijack import (
    adaptive_LlamaModel_forward,
    adaptive_llama_flash_attn2_forward,
    prepare_inputs_for_generation_llama,
)
from adakv.mistral_hijack import (
    adaptive_MistralModel_forward,
    adaptive_mistral_flash_attn2_forward,
    prepare_inputs_for_generation_mistral,
)

def install_adakv_monkey_patch() -> None:
    """Replace model/attention/generation hooks for AdaKV only."""
    print("Installing AdaKV monkey patch (flatten cache + varlen decode)...")
    transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = (
        prepare_inputs_for_generation_llama
    )
    transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = adaptive_llama_flash_attn2_forward
    transformers.models.llama.modeling_llama.LlamaModel.forward = adaptive_LlamaModel_forward

    transformers.models.mistral.modeling_mistral.MistralForCausalLM.prepare_inputs_for_generation = (
        prepare_inputs_for_generation_mistral
    )
    transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = (
        adaptive_mistral_flash_attn2_forward
    )
    transformers.models.mistral.modeling_mistral.MistralModel.forward = adaptive_MistralModel_forward
