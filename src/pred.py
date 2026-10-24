"""LongBench prediction with shared baseline and TideKV configuration."""

import argparse
from pathlib import Path
import sys

import torch
from tqdm import tqdm

from inference import InferenceRuntime, add_inference_arguments, release_sample_memory
from manifest import append_prediction, read_jsonl
from pred_utils import build_chat, post_process, post_process_code_completion
from project_config import ROOT, load_config
from utils import seed_everything

TASKS = (
    "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique",
    "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum",
    "passage_count", "passage_retrieval_en", "lcc", "repobench-p",
)
RAW_TASKS = {"trec", "triviaqa", "samsum", "lcc", "repobench-p"}
CODE_TASKS = {"lcc", "repobench-p"}

def prepare_prompt(runtime, row, task, template, max_length):
    tokenizer = runtime.tokenizer
    prompt = template.format(**row)
    ids = tokenizer(prompt, truncation=False).input_ids
    if len(ids) > max_length:
        if task in CODE_TASKS:
            prompt = tokenizer.decode(ids[-max_length:], skip_special_tokens=True)
        else:
            half = max_length // 2
            prompt = (tokenizer.decode(ids[:half], skip_special_tokens=True)
                      + tokenizer.decode(ids[-half:], skip_special_tokens=True))
    if task not in RAW_TASKS:
        prompt = build_chat(tokenizer, prompt, runtime.args.model)
    return prompt

def predict_sample(runtime, row, task, template, max_length, max_gen):
    prompt = prepare_prompt(runtime, row, task, template, max_length)
    pred = runtime.generate(prompt, max_gen)
    if task in CODE_TASKS:
        return post_process_code_completion(pred)
    pred = post_process(pred, runtime.args.model)
    return pred.split("\n")[0].strip() if task in {"trec", "triviaqa", "samsum"} else pred

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_inference_arguments(parser)
    parser.add_argument("--dataset_dir", type=Path, default=ROOT / "dataset/longbench")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--task", choices=TASKS)
    group.add_argument("--tasks", nargs="+", choices=TASKS)
    args = parser.parse_args()
    seed_everything(42)
    tasks = args.tasks or ([args.task] if args.task else TASKS)
    paths = {task: args.dataset_dir / f"{task}.jsonl" for task in tasks}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    runtime = InferenceRuntime(args)
    templates, max_generation = load_config("dataset2prompt"), load_config("dataset2maxlen")
    max_length = load_config("model2maxlen")[args.model]
    for task, path in paths.items():
        data = read_jsonl(path)
        if args.max_samples is not None:
            data = data[:args.max_samples]
        out_path = Path(args.save_dir) / f"{task}-{args.cache_type}.jsonl"
        oom = 0
        with out_path.open("w", encoding="utf-8") as handle:
            for row in tqdm(data, desc=task):
                result = {key: row[key] for key in ("answers", "all_classes", "length")}
                try:
                    result["pred"] = predict_sample(runtime, row, task, templates[task], max_length, max_generation[task])
                except RuntimeError as error:
                    if not isinstance(error, torch.cuda.OutOfMemoryError) and "out of memory" not in str(error).lower():
                        raise
                    if args.fail_on_oom:
                        raise
                    result.update(pred="", oom=True)
                    oom += 1
                    print(f"[OOM] {task}, length={row.get('length')}", file=sys.stderr)
                release_sample_memory()
                append_prediction(handle, result)
        print(f"{task}: total={len(data)}, OOM={oom}")

if __name__ == "__main__":
    main()
