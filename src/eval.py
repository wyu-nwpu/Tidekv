import os
import json
import argparse
from typing import List, Any, Dict

from metric import (
    qa_f1_score,
    rouge_score,
    classification_score,
    retrieval_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Evaluate jsonl predictions under save_dir")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory under results that contains jsonl files")
    return parser.parse_args(args)

def compute_score(dataset: str, predictions: List[str], answers: List[List[str]], all_classes: Any) -> float:
    metric_fn = dataset2metric.get(dataset, qa_f1_score)
    total = 0.0
    n = len(predictions)
    for pred, gts in zip(predictions, answers):
        score = 0.0
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            pred = pred.lstrip('\n').split('\n')[0]
        for gt in gts:
            score = max(score, metric_fn(pred, gt, all_classes=all_classes))
        total += score
    return round(100.0 * total / max(n, 1), 2)

def iter_jsonl_files(root: str):
    for r, _, files in os.walk(root):
        for f in files:
            if f.endswith(".jsonl"):
                yield os.path.join(r, f)

def infer_dataset_name(file_path: str) -> str:
    """Match a known task name, optionally followed by a method suffix."""
    fname = os.path.basename(file_path)
    dataset = fname.split('.')[0]  # e.g. "lcc-snap" or "repobench-p-snap"

    for known in sorted(dataset2metric.keys(), key=len, reverse=True):
        if dataset == known or dataset.startswith(known + "-"):
            return known
    raise ValueError(f"Unknown LongBench prediction file: {file_path}")

if __name__ == "__main__":
    args = parse_args()
    path = args.save_dir
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Prediction directory does not exist: {path}")

    if not os.path.isabs(path):
        path = os.path.abspath(path)

    scores: Dict[str, Any] = {}
    total_samples = 0
    evaluated_samples = 0
    oom_samples = 0
    oom_by_dataset: Dict[str, Dict[str, int]] = {}
    for file_path in iter_jsonl_files(path):
        predictions: List[str] = []
        answers: List[List[str]] = []
        all_classes = None
        file_total = 0
        file_oom = 0

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                file_total += 1
                total_samples += 1
                if data.get("oom"):
                    file_oom += 1
                    oom_samples += 1
                    continue
                predictions.append(data["pred"])
                answers.append(data["answers"])
                evaluated_samples += 1
                if all_classes is None:
                    all_classes = data.get("all_classes", None)

        dataset = infer_dataset_name(file_path)
        rel_key = os.path.relpath(file_path, path)
        print("***************")
        oom_by_dataset[rel_key] = {
            "total_samples": file_total,
            "evaluated_samples": len(predictions),
            "oom_samples": file_oom,
        }
        score_value = (
            compute_score(dataset, predictions, answers, all_classes)
            if predictions
            else None
        )
        scores[rel_key] = score_value

    scores = dict(sorted(scores.items()))
    if not scores:
        raise ValueError(f"No LongBench predictions found in {path}")
    num_datasets = len(scores)
    valid_scores = [score for score in scores.values() if score is not None]
    average = round(sum(valid_scores) / len(valid_scores), 2) if valid_scores else None

    result_payload: Dict[str, Any] = {
        **scores,
        "average": average,
        "num_datasets": num_datasets,
        "total_samples": total_samples,
        "evaluated_samples": evaluated_samples,
        "oom_samples": oom_samples,
        "oom_rate": round(oom_samples / total_samples, 6) if total_samples else 0.0,
        "oom_by_dataset": oom_by_dataset,
    }

    print(scores)
    print(f"Average over {len(valid_scores)}/{num_datasets} scored datasets: {average}")
    print(f"OOM: {oom_samples}/{total_samples} "
          f"({oom_samples / total_samples:.2%})" if total_samples else "OOM: 0/0 (0.00%)")
    output_path = os.path.join(path, "result.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, ensure_ascii=False, indent=2)
    print(f"Evaluated {num_datasets} files under: {path}")
    print("result is saved to:", output_path)
