#!/usr/bin/env python3
"""Prepare and validate the method-independent Llama RULER data tree."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
RULER_PREPARE = ROOT / "dataset" / "ruler" / "RULER-main" / "scripts" / "data" / "prepare.py"
FORMAL_RULER_LENGTHS = (4096, 8192, 16384, 32768, 65536, 131072)
RULER_TASKS = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
)
ASSISTANT_GENERATION_HEADER = "<|start_header_id|>assistant<|end_header_id|>"
MODEL_ID = "Llama-3.1-8B-Instruct-128k"

class SharedRulerDataError(ValueError):
    """Raised when a shared RULER task file is missing or malformed."""

def _task_file(root: Path, length: int, task: str) -> Path:
    return root / str(length) / task / "validation.jsonl"

def _error(length: int, task: str, path: Path, message: str) -> SharedRulerDataError:
    return SharedRulerDataError(
        f"invalid shared RULER data for length={length} task={task} path={path}: {message}"
    )

def _validate_file(path: Path, length: int, task: str, num_samples: int, model_id: str = MODEL_ID) -> int:
    if not path.is_file():
        raise _error(length, task, path, "missing validation.jsonl")

    rows = 0
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise _error(length, task, path, f"line {line_number}: invalid JSON ({error.msg})") from error
                if not isinstance(row, dict):
                    raise _error(length, task, path, f"line {line_number}: row must be a JSON object")
                if not isinstance(row.get("input"), str):
                    raise _error(length, task, path, f"line {line_number}: input must be a string")
                if model_id.startswith("Mistral"):
                    if "[INST]" not in row["input"] or "[/INST]" not in row["input"]:
                        raise _error(length, task, path, f"line {line_number}: missing Mistral instruction template")
                elif ASSISTANT_GENERATION_HEADER not in row["input"]:
                    raise _error(length, task, path, f"line {line_number}: missing assistant generation header")
                if not isinstance(row.get("index"), int) or isinstance(row["index"], bool):
                    raise _error(length, task, path, f"line {line_number}: index must be an integer")
                if not isinstance(row.get("outputs"), list) or not row["outputs"]:
                    raise _error(length, task, path, f"line {line_number}: outputs must be a nonempty list")
                if not isinstance(row.get("answer_prefix"), str):
                    raise _error(length, task, path, f"line {line_number}: answer_prefix must be a string")
                rows += 1
    except OSError as error:
        raise _error(length, task, path, str(error)) from error

    if rows != num_samples:
        raise _error(length, task, path, f"expected {num_samples} records, found {rows}")
    return rows

def validate_shared_data(
    data_dir: Path | str,
    lengths: Iterable[int],
    tasks: Iterable[str],
    num_samples: int,
    model_id: str = MODEL_ID,
) -> dict[str, int]:
    """Validate every requested task file and return its aggregate counts."""
    root = Path(data_dir)
    normalized_lengths = tuple(int(length) for length in lengths)
    normalized_tasks = tuple(tasks)
    if num_samples < 1:
        raise ValueError("num_samples must be positive")

    records = 0
    for length in normalized_lengths:
        for task in normalized_tasks:
            records += _validate_file(_task_file(root, length, task), length, task, num_samples, model_id)
    return {"files": len(normalized_lengths) * len(normalized_tasks), "records": records}

def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_path = Path(handle.name)
    os.replace(temp_path, path)

def _quarantine_invalid_file(path: Path) -> Path | None:
    """Move an invalid target aside so the upstream count-only check cannot skip it."""
    if not path.exists():
        return None
    if not path.is_file():
        raise OSError(f"cannot quarantine non-file target: {path}")
    descriptor, quarantine_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.invalid.",
        suffix=".jsonl",
    )
    os.close(descriptor)
    quarantine_path = Path(quarantine_name)
    path.replace(quarantine_path)
    return quarantine_path

def prepare_shared_data(
    data_dir: Path | str,
    tokenizer_path: Path | str,
    lengths: Sequence[int],
    tasks: Sequence[str],
    num_samples: int,
    seed: int,
    python: Path | str,
    *,
    model_id: str = MODEL_ID,
) -> dict[str, int]:
    """Regenerate only missing or invalid RULER task files, then validate all."""
    root = Path(data_dir)
    tokenizer = Path(tokenizer_path)
    normalized_lengths = tuple(int(length) for length in lengths)
    normalized_tasks = tuple(tasks)
    if not tokenizer.is_dir() or not any((tokenizer / name).is_file() for name in ("tokenizer.json", "tokenizer.model")):
        raise FileNotFoundError(f"Tokenizer assets are missing from {tokenizer}")
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    if model_id.startswith("Mistral") and any(length > 32768 for length in normalized_lengths):
        raise ValueError("Mistral RULER preparation is limited to 32768 tokens")
    root.mkdir(parents=True, exist_ok=True)

    tokenizer_files = [
        {"name": name, "size_bytes": (tokenizer / name).stat().st_size,
         "mtime_ns": (tokenizer / name).stat().st_mtime_ns}
        for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json")
        if (tokenizer / name).is_file()
    ]
    identity = {
        "model_id": model_id, "num_samples": num_samples, "seed": seed,
        "tokenizer_files": tokenizer_files,
        "model_template_type": "meta-chat" if model_id.startswith("Mistral") else "meta-llama3",
    }
    config_path = root / "prepare_config.json"
    if config_path.is_file():
        previous = json.loads(config_path.read_text())
        changed = [key for key, value in identity.items() if key in previous and previous[key] != value]
        if changed:
            raise SharedRulerDataError(
                f"Shared-data generation settings changed: {changed}. Use a different --data_dir."
            )
    config = {
        **identity, "generator_path": str(RULER_PREPARE.resolve()),
        "lengths": list(normalized_lengths), "tasks": list(normalized_tasks),
        "subset": "validation", "tokenizer_path": str(tokenizer.resolve()),
        "tokenizer_type": "hf", "status": "preparing",
    }
    _atomic_write_json(config_path, config)

    for length in normalized_lengths:
        for task in normalized_tasks:
            path = _task_file(root, length, task)
            try:
                _validate_file(path, length, task, num_samples, model_id)
            except SharedRulerDataError:
                quarantine_path = _quarantine_invalid_file(path)
                command = [
                    str(python),
                    str(RULER_PREPARE),
                    "--save_dir",
                    str(root / str(length)),
                    "--benchmark",
                    "synthetic",
                    "--task",
                    task,
                    "--tokenizer_path",
                    str(tokenizer),
                    "--tokenizer_type",
                    "hf",
                    "--max_seq_length",
                    str(length),
                    "--model_template_type",
                    "meta-chat" if model_id.startswith("Mistral") else "meta-llama3",
                    "--num_samples",
                    str(num_samples),
                    "--subset",
                    "validation",
                    "--random_seed",
                    str(seed),
                ]
                subprocess.run(command, check=True, cwd=RULER_PREPARE.parent)
                _validate_file(path, length, task, num_samples, model_id)
                if quarantine_path is not None:
                    quarantine_path.unlink()

    result = validate_shared_data(root, normalized_lengths, normalized_tasks, num_samples, model_id)
    _atomic_write_json(config_path, {**config, "status": "complete"})
    return result

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "validate"))
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--lengths", nargs="+", type=int, default=list(FORMAL_RULER_LENGTHS))
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer_path", type=Path)
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--tasks", nargs="+", choices=RULER_TASKS, default=list(RULER_TASKS))
    return parser.parse_args()

def main() -> None:
    args = _parse_args()
    if args.command == "validate":
        result = validate_shared_data(args.data_dir, args.lengths, args.tasks, args.num_samples, args.model_id)
    else:
        if args.tokenizer_path is None:
            raise SystemExit("prepare requires --tokenizer_path")
        result = prepare_shared_data(
            args.data_dir,
            args.tokenizer_path,
            args.lengths,
            args.tasks,
            args.num_samples,
            args.seed,
            Path(sys.executable),
            model_id=args.model_id,
        )
    print(json.dumps(result, sort_keys=True))

if __name__ == "__main__":
    main()
