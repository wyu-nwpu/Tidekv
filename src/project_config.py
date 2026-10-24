"""Project-relative configuration shared by the command-line entry points."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load_config(name):
    with (ROOT / "config" / f"{name}.json").open(encoding="utf-8") as handle:
        values = json.load(handle)
    if name == "model2path":
        values = {
            model: str((ROOT / path).resolve()) if path.startswith("models/") else path
            for model, path in values.items()
        }
        if os.environ.get("MODEL_PATH"):
            values[os.environ.get("MODEL_NAME", "Llama-3.1-8B-Instruct-128k")] = os.environ["MODEL_PATH"]
    return values
