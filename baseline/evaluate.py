#!/usr/bin/env python3
"""Evaluate canonical external-baseline predictions with repository metrics."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Callable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.adapters.common_io import atomic_write_json
from baseline.adapters.protocol import PredictionRecord

class EvaluationError(RuntimeError):
    pass

def _predictions(run_dir: Path) -> dict[str, list[PredictionRecord]]:
    grouped: dict[str, list[PredictionRecord]] = {}
    seen: set[tuple[str, str]] = set()
    for path in sorted((run_dir / "predictions").glob("*.jsonl")):
        task = path.stem
        rows = []
        with path.open(encoding="utf-8") as source:
            for row_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    record = PredictionRecord.from_dict(json.loads(line))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    raise EvaluationError(f"invalid prediction {path.name}:{row_number}") from error
                identity = (task, record.sample_id)
                if identity in seen:
                    raise EvaluationError(f"duplicate prediction identity: {record.sample_id}")
                seen.add(identity)
                rows.append(record)
        if rows:
            grouped[task] = rows
    if not grouped:
        raise EvaluationError("predictions are empty")
    return grouped

def _successful(rows: list[PredictionRecord]) -> list[PredictionRecord]:
    return [row for row in rows if not row.oom]

def _longbench_scores(grouped: dict[str, list[PredictionRecord]]) -> dict[str, float | None]:
    src = str(REPO_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from eval import compute_score, dataset2metric

    unknown = sorted(set(grouped).difference(dataset2metric))
    if unknown:
        raise EvaluationError(f"mixed or unknown LongBench tasks: {', '.join(unknown)}")
    scores: dict[str, float | None] = {}
    for task, rows in grouped.items():
        valid = _successful(rows)
        scores[task] = (
            compute_score(
                task,
                [row.pred for row in valid],
                [list(row.answers) for row in valid],
                list(valid[0].all_classes) if valid else [],
            )
            if valid
            else None
        )
    return scores

def _ruler_metrics() -> dict[str, Callable[[list[str], list[list[str]]], float]]:
    constants_path = REPO_ROOT / "dataset/ruler/RULER-main/scripts/eval/synthetic/constants.py"
    spec = importlib.util.spec_from_file_location("readkv_ruler_synthetic_constants", constants_path)
    if spec is None or spec.loader is None:
        raise EvaluationError("cannot load RULER metric module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    customized = yaml.safe_load((REPO_ROOT / "dataset/ruler/RULER-main/scripts/synthetic.yaml").read_text())
    return {name: module.TASKS[config["task"]]["metric_fn"] for name, config in customized.items()}

def _ruler_scores(grouped: dict[str, list[PredictionRecord]]) -> dict[str, float | None]:
    metrics = _ruler_metrics()
    unknown = sorted(set(grouped).difference(metrics))
    if unknown:
        raise EvaluationError(f"mixed or unknown RULER tasks: {', '.join(unknown)}")
    scores: dict[str, float | None] = {}
    for task, rows in grouped.items():
        valid = _successful(rows)
        scores[task] = (
            float(
                metrics[task](
                    [row.pred.strip() for row in valid],
                    [list(row.answers) for row in valid],
                )
            )
            if valid
            else None
        )
    return scores

def evaluate_run(run_dir: Path, benchmark: str) -> dict[str, float | int | None]:
    run_dir = Path(run_dir)
    grouped = _predictions(run_dir)
    if benchmark == "longbench":
        scores = _longbench_scores(grouped)
    elif benchmark == "ruler":
        scores = _ruler_scores(grouped)
    else:
        raise EvaluationError(f"unsupported benchmark: {benchmark}")
    scored = [score for score in scores.values() if score is not None]
    records = [row for rows in grouped.values() for row in rows]
    oom_samples = sum(row.oom for row in records)
    result: dict[str, float | int | None] = dict(sorted(scores.items()))
    result["average"] = round(sum(scored) / len(scored), 2) if scored else None
    result["num_tasks"] = len(scores)
    result["total_samples"] = len(records)
    result["evaluated_samples"] = len(records) - oom_samples
    result["oom_samples"] = oom_samples
    result["oom_rate"] = round(oom_samples / len(records), 6) if records else 0.0
    atomic_write_json(run_dir / "metrics.json", result)
    return result

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--benchmark", choices=("longbench", "ruler"), required=True)
    args = parser.parse_args()
    try:
        result = evaluate_run(args.run_dir, args.benchmark)
    except EvaluationError as error:
        print(f"evaluation failed: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
