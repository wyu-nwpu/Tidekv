"""JSONL input and prediction helpers."""

import json
from pathlib import Path

def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

def append_prediction(handle, row):
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()
