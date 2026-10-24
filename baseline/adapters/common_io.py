"""Durable JSON and JSONL helpers for external baseline runs."""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from .protocol import InputRecord, PredictionRecord

def _json_line(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )

def _reject_non_finite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")

def _load_json(line: str) -> Any:
    return json.loads(line, parse_constant=_reject_non_finite_constant)

def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)

def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically replace *path* with a fully synced JSON document."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    try:
        with temporary_path.open("w", encoding="utf-8") as output:
            output.write(_json_line(value))
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

def read_inputs(path: Path) -> list[InputRecord]:
    """Read and validate a canonical input JSONL manifest."""
    path = Path(path)
    records: list[InputRecord] = []
    open_text = gzip.open if path.suffix == ".gz" else Path.open
    with open_text(path, "rt", encoding="utf-8") as source:
        for row_number, line in enumerate(source, start=1):
            try:
                data = _load_json(line)
                if not isinstance(data, dict):
                    raise ValueError("row must be a JSON object")
                records.append(InputRecord.from_dict(data))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid input JSONL row {row_number}") from error
    return records

def append_prediction(path: Path, record: PredictionRecord) -> None:
    """Append one fully flushed prediction record without rewriting prior rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    created = not path.exists()
    with path.open("a", encoding="utf-8") as output:
        output.write(_json_line(record.to_dict()))
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    if created:
        _fsync_directory(path.parent)

