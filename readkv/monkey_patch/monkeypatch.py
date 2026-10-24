"""Install one supported attention patch per inference process."""

import transformers

from . import llama, mistral

_ACTIVE_KV_PATCH = None
_PATCH_METHODS = {"snap_a", "pyramidkv_a", "chunkkv", "adakv", "h2o", "streaming"}
_FLASH_FUNCTIONS = {
    "snap_a": "flash_attn2_forward_SnapKV_A",
    "pyramidkv_a": "flash_attn2_forward_PyramidKV_A",
    "chunkkv": "flash_attn2_forward_ChunkKV",
    "streaming": "flash_attn2_forward_StreamingLLM",
    "h2o": "flash_attn2_forward_H2O",
}


def infer_model_type_from_name(name):
    for model_type in ("llama", "mistral"):
        if model_type in (name or "").lower():
            return model_type
    return None


def validate_patch_support(cache_type, attn_implementation, model_type=None):
    if cache_type not in _PATCH_METHODS:
        return
    if model_type not in {"llama", "mistral"}:
        raise NotImplementedError(f"Unsupported model type: {model_type}")
    if attn_implementation == "flash_attention_2":
        return
    if model_type == "llama" and cache_type == "snap_a" and attn_implementation in {"sdpa", "eager"}:
        return
    if model_type == "llama" and cache_type == "h2o" and attn_implementation == "eager":
        return
    raise NotImplementedError(f"{cache_type}/{model_type} has no {attn_implementation} patch")


def install_kv_compression_monkey_patch(cache_type, attn_implementation, model_type=None):
    global _ACTIVE_KV_PATCH
    unpatched = {"full", "tidekv", "tidekv_snap", "tidekv_pyramidkv", "tidekv_adakv", "tidekv_chunkkv"}
    if cache_type not in _PATCH_METHODS | unpatched:
        raise ValueError(f"Unsupported method: {cache_type}")
    validate_patch_support(cache_type, attn_implementation, model_type)
    requested = (cache_type, attn_implementation, model_type) if cache_type in _PATCH_METHODS else None
    if _ACTIVE_KV_PATCH is not None and requested != _ACTIVE_KV_PATCH:
        raise RuntimeError("Run each attention method in a separate process")
    if requested is None or requested == _ACTIVE_KV_PATCH:
        return
    if cache_type == "adakv":
        from adakv.install import install_adakv_monkey_patch
        install_adakv_monkey_patch()
    else:
        module = getattr(transformers.models, model_type)
        modeling = getattr(module, f"modeling_{model_type}")
        functions = llama if model_type == "llama" else mistral
        prefix = "Llama" if model_type == "llama" else "Mistral"
        if attn_implementation == "flash_attention_2":
            attention_class = getattr(modeling, f"{prefix}FlashAttention2")
            forward = getattr(functions, f"{model_type}_{_FLASH_FUNCTIONS[cache_type]}")
        elif attn_implementation == "sdpa":
            attention_class = modeling.LlamaSdpaAttention
            forward = llama.llama_sdpa_attn_forward_SnapKV_A
        else:
            attention_class = modeling.LlamaAttention
            forward = llama.llama_attn_forward_H2O if cache_type == "h2o" else llama.llama_eager_attn_forward_SnapKV_A
        attention_class.forward = forward
    _ACTIVE_KV_PATCH = requested
    print(f"[attention] {cache_type}, {model_type}, {attn_implementation}")
