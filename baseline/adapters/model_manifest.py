"""Model and tokenizer file metadata for external baseline runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

_METADATA_NAMES = {
    "added_tokens.json",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
}
_WEIGHT_SUFFIXES = {".bin", ".pt", ".pth", ".safetensors"}

def capture_model_manifest(model_path: Path) -> dict[str, Any]:
    """Record metadata and weight shard names/sizes without reading file contents."""
    model_path = Path(model_path).resolve()
    if not model_path.is_dir():
        raise ValueError(f"model path does not exist: {model_path}")

    metadata_files = sorted(
        path
        for path in model_path.iterdir()
        if path.is_file()
        and (path.name in _METADATA_NAMES or path.name.endswith(".index.json"))
    )
    weight_shards = sorted(
        path
        for path in model_path.iterdir()
        if path.is_file() and path.suffix in _WEIGHT_SUFFIXES
    )
    return {
        "model_path": str(model_path),
        "metadata_files": [
            {"name": path.name, "size_bytes": path.stat().st_size} for path in metadata_files
        ],
        "weight_shards": [
            {"name": path.name, "size_bytes": path.stat().st_size}
            for path in weight_shards
        ],
    }
