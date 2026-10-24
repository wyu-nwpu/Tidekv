"""Shared defaults for compact-cache baselines and paired TideKV variants."""

from dataclasses import dataclass
import json
from pathlib import Path

PAIRS = {
    "snap_a": "tidekv_snap",
    "pyramidkv_a": "tidekv_pyramidkv",
    "adakv": "tidekv_adakv",
    "chunkkv": "tidekv_chunkkv",
}
TIDEKV_CACHE_TYPES = ("tidekv", *PAIRS.values())
SUPPORTED_CACHE_TYPES = ("full", *PAIRS, *TIDEKV_CACHE_TYPES, "h2o", "streaming")
_DEFAULTS = {
    "full": (32, 0, 5), "snap_a": (32, 0, 5), "adakv": (32, 0, 5),
    "pyramidkv_a": (128, 0, 7), "chunkkv": (64, 0, 5),
    "h2o": (128, 0, 5), "streaming": (128, 64, 5),
}


@dataclass(frozen=True)
class BaselineConfig:
    cache_type: str
    window_size: int
    sink_size: int
    kernel_size: int
    pooling: str
    chunk_length: int
    chunkkv_reuse_layers: int


def is_tidekv_cache_type(method):
    return method in TIDEKV_CACHE_TYPES


def tidekv_selection_for_cache_type(method):
    return "chunkkv" if method == "tidekv_chunkkv" else "snapkv"


def tidekv_head_adaptive_for_cache_type(method):
    return method == "tidekv_adakv"


def tidekv_layer_budget_for_cache_type(method):
    return "pyramid" if method == "tidekv_pyramidkv" else "uniform"


def resolve_baseline_config(cache_type, *, budget=None, window_size=None, sink_size=None,
                            kernel_size=None, pooling=None, chunk_length=20,
                            chunkkv_reuse_layers=2):
    if cache_type not in SUPPORTED_CACHE_TYPES:
        raise ValueError(f"Unsupported method: {cache_type}")
    pair = {variant: baseline for baseline, variant in PAIRS.items()}
    baseline = pair.get(cache_type, "snap_a" if cache_type == "tidekv" else cache_type)
    window, sink, kernel = _DEFAULTS[baseline]
    if baseline == "pyramidkv_a" and budget == 256:
        window, kernel = 8, 5
    window = window if window_size is None else window_size
    sink = sink if sink_size is None else sink_size
    kernel = kernel if kernel_size is None else kernel_size
    pooling = pooling or "avgpool"
    if window < 1 or sink < 0 or kernel < 1 or kernel % 2 != 1:
        raise ValueError("Invalid window, sink, or odd pooling kernel")
    if pooling not in {"avgpool", "maxpool"} or chunk_length < 1 or chunkkv_reuse_layers < 1:
        raise ValueError("Invalid pooling or ChunkKV settings")
    if cache_type == "h2o":
        sink = 0
    return BaselineConfig(cache_type, window, sink, kernel,
                          pooling, chunk_length, chunkkv_reuse_layers)


def write_baseline_run_config(output_dir, config, *, budget, chunk_size, runtime_config):
    payload = {
        "cache_type": config.cache_type, "budget": budget,
        "window_size": config.window_size, "kernel_size": config.kernel_size,
        "pooling": config.pooling, "chunk_size": chunk_size,
        "effective_runtime": runtime_config,
    }
    if config.cache_type in {"chunkkv", "tidekv_chunkkv"}:
        payload.update(chunk_length=config.chunk_length, reuse_layers=config.chunkkv_reuse_layers)
    if config.cache_type in {"h2o", "streaming"}:
        payload["sink_size"] = config.sink_size
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.glob("*.jsonl")):
        raise FileExistsError(f"Predictions already exist; choose a new output directory: {directory}")
    path = directory / "baseline_run_config.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload
