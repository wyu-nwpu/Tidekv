import random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from baseline_config import (
    is_tidekv_cache_type,
    resolve_baseline_config,
    tidekv_head_adaptive_for_cache_type,
    tidekv_selection_for_cache_type,
)

def str2bool(value):
    """Parse a command-line boolean value."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"boolean value must be true/false, got {value!r}")

def precision_to_dtype(precision: str) -> torch.dtype:
    mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    try:
        return mapping[precision.lower()]
    except KeyError as exc:
        raise ValueError(
            f"precision must be one of {sorted(mapping)}, got {precision!r}"
        ) from exc

def seed_everything(seed):
    random.seed(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_model_and_tokenizer(
    model_name_or_path:str,
    precision:str='fp16',
    hf_cache_dir:str=None,
    max_capacity_prompts:int=1024,
    window_sizes:int=128,
    sink_sizes:int=64,
    flash_attn:str=None,
    use_cache:bool=True,
    floor_alpha:float=0.5,
    adakv_skip:int=0,
    adakv_normalize:bool=False,
    pyram_beta:int=20,
    pyramid_source_gqa_ratio:int=1,
    chunk_length:int=20,
    cache_type:str="full",
    kernel_size:int=None,
    pooling:str=None,
    chunkkv_reuse_layers:int=2,
)-> tuple[AutoModelForCausalLM,AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=hf_cache_dir,
        padding_side="left",
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the supported inference paths")
    if torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one GPU per inference process with CUDA_VISIBLE_DEVICES")
    device_map = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        cache_dir=hf_cache_dir,
        torch_dtype=precision_to_dtype(precision),
        device_map=device_map,
        attn_implementation=flash_attn,
        use_cache=use_cache,
    )
    
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    baseline = resolve_baseline_config(
        cache_type,
        window_size=window_sizes,
        sink_size=sink_sizes,
        kernel_size=kernel_size,
        pooling=pooling,
        chunk_length=chunk_length,
        chunkkv_reuse_layers=chunkkv_reuse_layers,
    )
    if max_capacity_prompts is None:
        max_capacity_prompts = 1024
    window_sizes = baseline.window_size
    sink_sizes = baseline.sink_size
    kernel_sizes = baseline.kernel_size
    pooling = baseline.pooling
    tidekv_selection_mode = (
        tidekv_selection_for_cache_type(cache_type)
        if is_tidekv_cache_type(cache_type) else None
    )
    tidekv_head_adaptive = (
        tidekv_head_adaptive_for_cache_type(cache_type)
        if is_tidekv_cache_type(cache_type) else False
    )
    model.config.tidekv_selection_mode = tidekv_selection_mode
    model.config.tidekv_chunk_length = chunk_length
    model.config.tidekv_head_adaptive = tidekv_head_adaptive
    print(
        "[BASELINE] "
        f"method={baseline.cache_type} "
        f"window={baseline.window_size} "
        f"sink={baseline.sink_size} kernel={baseline.kernel_size} "
        f"pooling={baseline.pooling} chunk_length={baseline.chunk_length} "
        f"reuse_layers={baseline.chunkkv_reuse_layers}"
    )

    layers = len(model.model.layers)
    if not isinstance(window_sizes, list):
        window_sizes = [window_sizes] * layers
    if not isinstance(max_capacity_prompts, list):
        max_capacity_prompts = [max_capacity_prompts] * layers
    if not isinstance(kernel_sizes, list):
        kernel_sizes = [kernel_sizes] * layers

    if not isinstance(sink_sizes, list):
        sink_sizes = [sink_sizes] * layers

    for i in range(layers):
        attn_cfg = model.model.layers[i].self_attn.config
        attn_cfg.window_size = window_sizes[i]
        attn_cfg.max_capacity_prompt = max_capacity_prompts[i]
        attn_cfg.kernel_size = kernel_sizes[i]
        attn_cfg.sink_size = sink_sizes[i]
        attn_cfg.floor_alpha = floor_alpha
        attn_cfg.adakv_skip = adakv_skip
        attn_cfg.adakv_normalize = adakv_normalize
        attn_cfg.pyram_beta = pyram_beta
        attn_cfg.pyramid_source_gqa_ratio = pyramid_source_gqa_ratio
        attn_cfg.chunk_length = chunk_length
        attn_cfg.chunkkv_reuse_layers = baseline.chunkkv_reuse_layers
        attn_cfg.tidekv_selection_mode = tidekv_selection_mode
        attn_cfg.tidekv_chunk_length = chunk_length
        attn_cfg.tidekv_head_adaptive = tidekv_head_adaptive
        # AdaKV-main config names (used by readkv.adakv cluster)
        attn_cfg.base_capacity = max_capacity_prompts[i]
        attn_cfg.skip = adakv_skip
        attn_cfg.normalize = adakv_normalize
        attn_cfg.pooling = pooling
        attn_cfg.pyram_mode = False
        attn_cfg.gqa_support = attn_cfg.num_attention_heads != attn_cfg.num_key_value_heads
        attn_cfg.gqa_func = "mean"

    return model, tokenizer
