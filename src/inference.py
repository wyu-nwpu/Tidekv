"""Shared inference configuration and sample-level generation."""

import gc
import sys
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache

from baseline_config import (
    SUPPORTED_CACHE_TYPES, is_tidekv_cache_type, resolve_baseline_config,
    tidekv_head_adaptive_for_cache_type, tidekv_layer_budget_for_cache_type,
    tidekv_selection_for_cache_type, write_baseline_run_config,
)
from pred_utils import collect_eos_token_ids
from project_config import ROOT, load_config
from tidekv_budget import resolve_tidekv_max_capacity
from utils import load_model_and_tokenizer, str2bool

sys.path.insert(0, str(ROOT / "readkv"))
from monkey_patch.monkeypatch import infer_model_type_from_name, install_kv_compression_monkey_patch
from tidekv import greedy_decode, install_tidekv_varlen_decode_patch, tidekv_prefill

def positive_int(value):
    value = int(value)
    if value < 1:
        raise ValueError("Expected a positive integer")
    return value

def add_inference_arguments(parser):
    parser.add_argument("--model", default="Llama-3.1-8B-Instruct-128k",
                        choices=["Llama-3.1-8B-Instruct-128k", "Mistral-7B-Instruct-v0.2-32k"])
    parser.add_argument("--model_path", help="Local Hugging Face model directory")
    parser.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument("--cache_type", required=True, choices=SUPPORTED_CACHE_TYPES)
    parser.add_argument("--budget", "--max_capacity_prompts", dest="max_capacity_prompts",
                        type=positive_int, default=1024, metavar="BUDGET",
                        help="KV entries per head (base budget for Pyramid)")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--max_samples", type=positive_int)
    parser.add_argument("--fail_on_oom", type=str2bool, default=False)
    parser.add_argument("--window_size", "--window_sizes", dest="window_sizes", type=positive_int)
    parser.add_argument("--sink_sizes", type=int)
    parser.add_argument("--kernel_size", type=positive_int)
    parser.add_argument("--pooling", choices=["avgpool", "maxpool"])
    parser.add_argument("--attn_implementation", default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--chunk_size", type=positive_int, default=512)
    parser.add_argument("--obs_window_size", type=positive_int)
    parser.add_argument("--warmup_layers", type=int, default=0)
    parser.add_argument("--layer_budget_config", help="Optional per-layer budget JSON")
    parser.add_argument("--tidekv_selection_mode", choices=["snapkv", "chunkkv"])
    parser.add_argument("--tidekv_head_adaptive", action="store_true")
    parser.add_argument("--tidekv_head_floor_alpha", type=float, default=0.5)
    parser.add_argument("--tidekv_head_normalize", action="store_true")
    parser.add_argument("--tidekv_layer_budget", choices=["uniform", "pyramid"])
    parser.add_argument("--pyram_beta", type=positive_int, default=20)
    parser.add_argument("--floor_alpha", type=float, default=0.5)
    parser.add_argument("--adakv_skip", type=int, default=0)
    parser.add_argument("--adakv_normalize", action="store_true")
    parser.add_argument("--chunk_length", type=positive_int, default=20)
    parser.add_argument("--chunkkv_reuse_layers", type=positive_int, default=2)

def resolve_arguments(args, gqa_ratio):
    """Use the same paired defaults for both datasets and direct CLI calls."""
    method = args.cache_type
    args.tidekv_cache = is_tidekv_cache_type(method)
    cfg = resolve_baseline_config(
        method, budget=args.max_capacity_prompts, window_size=args.window_sizes, sink_size=args.sink_sizes,
        kernel_size=args.kernel_size, pooling=args.pooling,
        chunk_length=args.chunk_length, chunkkv_reuse_layers=args.chunkkv_reuse_layers,
    )
    args.window_sizes, args.sink_sizes = cfg.window_size, cfg.sink_size
    args.kernel_size, args.pooling = cfg.kernel_size, cfg.pooling
    if args.obs_window_size is None:
        args.obs_window_size = args.window_sizes
    if args.tidekv_cache:
        args.tidekv_selection_mode = args.tidekv_selection_mode or tidekv_selection_for_cache_type(method)
        args.tidekv_layer_budget = args.tidekv_layer_budget or tidekv_layer_budget_for_cache_type(method)
        args.tidekv_head_adaptive |= tidekv_head_adaptive_for_cache_type(method)
        if args.tidekv_selection_mode != "chunkkv":
            args.chunkkv_reuse_layers = 1
    args.pyramid_source_gqa_ratio = gqa_ratio
    args.tidekv_pyramid_source_gqa_ratio = gqa_ratio
    if args.warmup_layers < 0 or args.adakv_skip < 0:
        raise ValueError("Layer counts must be nonnegative")
    if args.kernel_size % 2 != 1:
        raise ValueError("kernel_size must be odd")
    if not 0 <= args.floor_alpha <= 1 or not 0 <= args.tidekv_head_floor_alpha <= 1:
        raise ValueError("Head budget floor must lie in [0, 1]")
    minimum = args.obs_window_size if args.tidekv_cache else (args.sink_sizes if method == "streaming" else args.window_sizes)
    if method != "full" and args.max_capacity_prompts <= minimum:
        raise ValueError("Cache budget must exceed the observation window or sink size")
    return cfg

class InferenceRuntime:
    def __init__(self, args):
        self.args = args
        args.model_path = str(Path(args.model_path or load_config("model2path")[args.model]).resolve())
        model_config = AutoConfig.from_pretrained(args.model_path)
        gqa_ratio = model_config.num_attention_heads // model_config.num_key_value_heads
        cfg = resolve_arguments(args, gqa_ratio)
        model_type = infer_model_type_from_name(args.model)
        if model_type != model_config.model_type:
            raise ValueError(f"Model name and model config disagree: {args.model}, {model_config.model_type}")
        if args.tidekv_head_adaptive and model_type != "llama":
            raise NotImplementedError("Adaptive-head TideKV decode is currently implemented for Llama only")
        if args.tidekv_head_adaptive and args.attn_implementation != "flash_attention_2":
            raise ValueError("Adaptive-head TideKV requires FlashAttention2 varlen decode")
        runtime_config = {key: str(value.resolve()) if isinstance(value, Path) else value
                          for key, value in vars(args).items()
                          if key not in {"save_dir", "fail_on_oom", "task", "tasks"}}
        self.capacity = resolve_tidekv_max_capacity(args, model_config.num_hidden_layers) if args.tidekv_cache else args.max_capacity_prompts
        runtime_config["effective_layer_budgets"] = self.capacity
        write_baseline_run_config(args.save_dir, cfg, budget=args.max_capacity_prompts,
                                  chunk_size=args.chunk_size,
                                  runtime_config=runtime_config)
        install_kv_compression_monkey_patch(args.cache_type, args.attn_implementation, model_type=model_type)
        if args.tidekv_cache and args.tidekv_head_adaptive:
            install_tidekv_varlen_decode_patch()
        self.model, self.tokenizer = load_model_and_tokenizer(
            args.model_path, args.precision, max_capacity_prompts=args.max_capacity_prompts,
            window_sizes=args.window_sizes, sink_sizes=args.sink_sizes,
            flash_attn=args.attn_implementation, floor_alpha=args.floor_alpha,
            adakv_skip=args.adakv_skip, adakv_normalize=args.adakv_normalize,
            pyram_beta=args.pyram_beta, pyramid_source_gqa_ratio=args.pyramid_source_gqa_ratio,
            chunk_length=args.chunk_length, cache_type=args.cache_type,
            kernel_size=args.kernel_size, pooling=args.pooling,
            chunkkv_reuse_layers=args.chunkkv_reuse_layers,
        )
        self.model.eval()
        self.eos_ids = collect_eos_token_ids(self.tokenizer, self.model)

    @torch.inference_mode()
    def generate(self, prompt, max_new_tokens):
        """Keep sample tensors in this frame so OOM recovery can release them."""
        args, model = self.args, self.model
        # Prepared Llama prompts already contain BOS and the assistant header.
        input_ids = self.tokenizer(prompt, return_tensors="pt", truncation=False,
                                   add_special_tokens=not prompt.startswith("<|begin_of_text|>")).input_ids.to(model.device)
        prompt_length = input_ids.shape[-1]
        if prompt_length + max_new_tokens > model.config.max_position_embeddings:
            raise ValueError(f"Prompt ({prompt_length}) plus generation ({max_new_tokens}) exceeds model context")
        if args.tidekv_cache:
            cache, logits = tidekv_prefill(
                model, input_ids, self.capacity, chunk_size=args.chunk_size,
                window_size=args.window_sizes, kernel_size=args.kernel_size, pooling=args.pooling,
                obs_window_size=args.obs_window_size, warmup_layers=args.warmup_layers,
                tidekv_head_adaptive=args.tidekv_head_adaptive,
                tidekv_head_floor_alpha=args.tidekv_head_floor_alpha,
                tidekv_head_normalize=args.tidekv_head_normalize,
                selection_mode=args.tidekv_selection_mode, chunk_length=args.chunk_length,
                chunkkv_reuse_layers=args.chunkkv_reuse_layers,
            )
            first_token = logits.argmax(dim=-1, keepdim=True)
            del logits
            generated = greedy_decode(model, cache, first_token, self.eos_ids, max_new_tokens, prompt_length)
        elif args.cache_type == "adakv":
            from adakv.generate import adakv_generate
            generated, cache = adakv_generate(model, input_ids, max_new_tokens, self.eos_ids)
        else:
            outputs = model.model(input_ids=input_ids, past_key_values=DynamicCache(), use_cache=True)
            first_token = model.lm_head(outputs.last_hidden_state[:, -1, :]).argmax(dim=-1, keepdim=True)
            cache = outputs.past_key_values
            del outputs
            generated = greedy_decode(model, cache, first_token, self.eos_ids, max_new_tokens, prompt_length)
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)

def release_sample_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
