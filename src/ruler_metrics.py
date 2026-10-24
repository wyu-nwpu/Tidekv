"""Official RULER substring metrics and per-task evaluation."""

import csv
import importlib.util
from pathlib import Path

import yaml

from manifest import read_jsonl
from project_config import ROOT

RULER_SCRIPTS = ROOT / "dataset/ruler/RULER-main/scripts"
TASKS = (
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2",
)

def load_task_config(task):
    with (RULER_SCRIPTS / "synthetic.yaml").open() as handle:
        customized = yaml.safe_load(handle)
    spec = importlib.util.spec_from_file_location("ruler_constants", RULER_SCRIPTS / "data/synthetic/constants.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = dict(customized[task])
    config.update(module.TASKS[config["task"]])
    return config

def evaluate_task(rows, task):
    """Return successful-sample score and explicit OOM coverage."""
    config = load_task_config(task)
    valid = [row for row in rows if not row.get("oom")]
    sample_scores = []
    for row in valid:
        references = row["outputs"]
        if not references or not all(isinstance(ref, str) and ref for ref in references):
            raise ValueError(f"Invalid RULER reference in {task}")
        hits = [float(ref.lower() in row["pred"].lower()) for ref in references]
        sample_scores.append(max(hits) if config["task"] == "qa" else sum(hits) / len(hits))
    return {
        "Task": task,
        "Score": round(100 * sum(sample_scores) / len(sample_scores), 2) if sample_scores else None,
        "Total": len(rows), "Evaluated": len(valid), "OOM": len(rows) - len(valid),
        "OOMRate": round((len(rows) - len(valid)) / len(rows), 6) if rows else 0.0,
    }

def evaluate_directory(directory, tasks=None):
    directory = Path(directory)
    if tasks is None:
        tasks = [task for task in TASKS if (directory / f"{task}.jsonl").is_file()]
    if not tasks:
        raise ValueError(f"No RULER predictions found in {directory}")
    scores = [evaluate_task(read_jsonl(directory / f"{task}.jsonl"), task) for task in tasks]
    valid = [row["Score"] for row in scores if row["Score"] is not None]
    total, evaluated, oom = (sum(row[key] for row in scores) for key in ("Total", "Evaluated", "OOM"))
    average = {"Task": "AVERAGE", "Score": round(sum(valid) / len(valid), 2) if valid else None,
               "Total": total, "Evaluated": evaluated, "OOM": oom,
               "OOMRate": round(oom / total, 6) if total else 0.0}
    with (directory / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(average))
        writer.writeheader()
        writer.writerows([*scores, average])
    return scores, average
