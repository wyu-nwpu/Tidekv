"""Install a single BPP generation entrypoint on LlamaForCausalLM."""

from __future__ import annotations

from dataclasses import asdict

from transformers.models.llama.modeling_llama import LlamaForCausalLM

from .engine import BPPGenerationEngine
from .registry import build_policy

_ACTIVE_PATCH = None
_ORIGINAL_BPP_GENERATE = getattr(LlamaForCausalLM, "bpp_generate", None)

def install_bpp_monkey_patch(spec) -> None:
    """Add ``model.bpp_generate`` and prevent mixed baselines per process."""
    global _ACTIVE_PATCH
    spec.validate()
    requested = tuple(sorted(asdict(spec).items()))
    if _ACTIVE_PATCH is not None and _ACTIVE_PATCH != requested:
        raise RuntimeError(
            "A different BPP baseline is already active. "
            "Run each baseline in an independent Python process."
        )

    def bpp_generate(self, input_ids, max_new_tokens, eos_token_ids):
        policy = build_policy(spec)
        engine = BPPGenerationEngine(self, spec, policy)
        return engine.generate(input_ids, max_new_tokens, eos_token_ids)

    LlamaForCausalLM.bpp_generate = bpp_generate
    _ACTIVE_PATCH = requested

def uninstall_bpp_monkey_patch() -> None:
    global _ACTIVE_PATCH
    if _ORIGINAL_BPP_GENERATE is None:
        if hasattr(LlamaForCausalLM, "bpp_generate"):
            delattr(LlamaForCausalLM, "bpp_generate")
    else:
        LlamaForCausalLM.bpp_generate = _ORIGINAL_BPP_GENERATE
    _ACTIVE_PATCH = None

def active_patch():
    return _ACTIVE_PATCH

