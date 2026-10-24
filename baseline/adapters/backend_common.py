"""Shared run lifecycle for isolated external baseline adapters."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
from pathlib import Path
import platform
import sys
from typing import Any, Callable

import torch

from baseline.adapters.common_io import (
    append_prediction,
    atomic_write_json,
    read_inputs,
)
from baseline.adapters.model_manifest import capture_model_manifest
from baseline.adapters.protocol import InputRecord, PredictionRecord
from baseline.registry import get_backend

def parse_adapter_args(method, argv=None):
    parser = argparse.ArgumentParser(description=f"Run {method} on tokenized inputs")
    parser.add_argument("--phase", required=True, choices=("longbench", "ruler"))
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--model_name", default="Llama-3.1-8B-Instruct-128k")
    parser.add_argument("--model_path", required=True, type=Path)
    parser.add_argument("--input_manifest", required=True, type=Path)
    parser.add_argument("--dataset")
    parser.add_argument("--save_dir", required=True, type=Path)
    parser.add_argument("--max_samples", type=int)
    if method == "kivi":
        parser.add_argument("--k_bits", type=int, choices=(2, 4), default=2)
        parser.add_argument("--v_bits", type=int, choices=(2, 4), default=2)
        parser.add_argument("--group_size", type=int, default=32)
        parser.add_argument("--residual_length", type=int, default=32)
    elif method == "keydiff":
        parser.add_argument("--kv_budget", type=int, default=4096)
        parser.add_argument("--block_size", type=int, default=128)
    args = parser.parse_args(argv)
    if args.model_name != "Llama-3.1-8B-Instruct-128k":
        parser.error("External baselines support Llama-3.1-8B-Instruct only")
    if args.gpu < 0 or (args.max_samples is not None and args.max_samples < 1):
        parser.error("GPU index must be nonnegative and max_samples must be positive")
    if method == "kivi":
        if args.group_size < 1 or args.residual_length < 1 or args.residual_length % args.group_size:
            parser.error("KIVI residual_length must be a positive multiple of group_size")
        args.config_id = f"k{args.k_bits}_v{args.v_bits}_g{args.group_size}_r{args.residual_length}"
    elif method == "keydiff":
        if args.kv_budget < 1 or args.block_size < 1:
            parser.error("KeyDiff budget and block size must be positive")
        args.keydiff_anchor = "raw_mean"
        args.config_id = f"n{args.kv_budget}_b{args.block_size}_raw_mean"
    else:
        args.config_id = "exact_offload_ordered"
    args.method = method
    args.source_dir = Path(__file__).resolve().parents[2] / get_backend(method).source_path
    args.precision = "fp16" if method == "kivi" else "bf16"
    args.batch_size = 1
    return args

def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return str(value)

def _config_payload(
    args,
    *,
    model_manifest: dict[str, Any],
) -> dict[str, Any]:
    arguments = {
        key: _jsonable(value)
        for key, value in sorted(vars(args).items())
    }
    backend = get_backend(args.method)
    payload = {
        "schema_version": 4,
        "arguments": arguments,
        "source_revision": backend.source_revision,
        "model_manifest": model_manifest,
    }
    return payload

def _environment_manifest(args) -> dict[str, Any]:
    packages = {}
    for name in ("torch", "transformers", "flash-attn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cuda_available = torch.cuda.is_available()
    device_name = None
    if cuda_available and args.gpu < torch.cuda.device_count():
        device_name = torch.cuda.get_device_name(args.gpu)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "torch_cuda": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpu": args.gpu,
        "device_name": device_name,
    }

def _source_manifest(args) -> dict[str, Any]:
    backend = get_backend(args.method)
    source_dir = Path(args.source_dir).resolve()
    return {
        "method": backend.method,
        "revision": backend.source_revision,
        "kind": backend.source_kind,
        "source_dir": str(source_dir),
        "config_id": args.config_id,
    }

def _prediction_path(save_dir: Path, record: InputRecord) -> Path:
    if not record.dataset or Path(record.dataset).name != record.dataset:
        raise ValueError(f"unsafe dataset name: {record.dataset!r}")
    return save_dir / "predictions" / f"{record.dataset}.jsonl"

def _status(path: Path, **values: Any) -> None:
    atomic_write_json(path, {"schema_version": 1, **values})

def _is_cuda_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()

def _oom_prediction(record: InputRecord) -> PredictionRecord:
    return PredictionRecord(
        sample_id=record.sample_id,
        pred="",
        answers=record.answers,
        all_classes=record.all_classes,
        length=record.length,
        prompt_tokens=record.length,
        generated_tokens=0,
        finish_reason="oom",
        oom=True,
        oom_stage="inference",
    )

def run_adapter(
    args,
    load_model: Callable[[Any], tuple[Any, Any]],
    run_record: Callable[[Any, Any, InputRecord, Any], PredictionRecord],
) -> int:
    """Run every selected sample once in a fresh output directory."""
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    status_path = save_dir / "status.json"
    try:
        if any((save_dir / "predictions").glob("*.jsonl")):
            raise FileExistsError(f"Predictions already exist; choose a new output directory: {save_dir}")
        records = read_inputs(Path(args.input_manifest))
        if args.dataset not in {args.phase, "all", None}:
            records = [record for record in records if record.dataset == args.dataset]
        if args.max_samples is not None:
            records = records[: args.max_samples]
        if not records:
            raise ValueError("input manifest contains no selected records")

        model_manifest = capture_model_manifest(Path(args.model_path))
        run_config = _config_payload(
            args,
            model_manifest=model_manifest,
        )
        atomic_write_json(save_dir / "run_config.json", run_config)
        atomic_write_json(
            save_dir / "environment.json", _environment_manifest(args)
        )
        atomic_write_json(save_dir / "source_manifest.json", _source_manifest(args))
        atomic_write_json(save_dir / "model_manifest.json", model_manifest)
        atomic_write_json(
            save_dir / "input_manifest.json",
            {
                "path": str(Path(args.input_manifest).resolve()),
                "records": len(records),
                "sample_ids": [record.sample_id for record in records],
            },
        )

        _status(
            status_path,
            status="running",
            stage="model_load",
            total=len(records),
            pending=len(records),
            completed=0,
            error=None,
        )

        model, tokenizer = load_model(args)
        oom_samples = 0
        for index, item in enumerate(records, start=1):
            _status(
                status_path,
                status="running",
                stage="inference",
                total=len(records),
                pending=len(records) - index + 1,
                completed=index - 1,
                sample_id=item.sample_id,
                error=None,
            )
            sample_oom = False
            try:
                prediction = run_record(model, tokenizer, item, args)
            except RuntimeError as error:
                if not _is_cuda_oom(error):
                    raise
                prediction = _oom_prediction(item)
                sample_oom = True
                print(f"[OOM] {item.sample_id}; continuing with the next sample")
            if sample_oom:
                gc.collect()
                torch.cuda.empty_cache()
            if prediction.sample_id != item.sample_id:
                raise RuntimeError("prediction sample_id mismatch")
            append_prediction(_prediction_path(save_dir, item), prediction)
            oom_samples += int(prediction.oom)

        _status(
            status_path,
            status="complete",
            stage="complete",
            total=len(records),
            pending=0,
            completed=len(records),
            oom_samples=oom_samples,
            oom_rate=oom_samples / len(records),
            error=None,
        )
        return 0
    except torch.cuda.OutOfMemoryError as error:
        torch.cuda.empty_cache()
        _status(status_path, status="oom", stage="inference", error=str(error))
        return 3
    except Exception as error:
        _status(status_path, status="failed", stage="error", error=f"{type(error).__name__}: {error}")
        return 1
