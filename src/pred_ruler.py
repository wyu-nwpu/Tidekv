"""RULER prediction and scoring using shared, preformatted task data."""

import argparse
from pathlib import Path
import sys

import torch
from tqdm import tqdm

from inference import InferenceRuntime, add_inference_arguments, release_sample_memory
from manifest import append_prediction, read_jsonl
from pred_utils import post_process
from ruler_metrics import TASKS, evaluate_directory, load_task_config
from utils import seed_everything

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_inference_arguments(parser)
    parser.add_argument("--data_dir", type=Path, required=True,
                        help="One length directory containing task/validation.jsonl")
    parser.add_argument("--tasks", nargs="+", choices=TASKS)
    parser.add_argument("--subset", default="validation")
    parser.add_argument("--ruler_prompt_mode", choices=["prepared"], default="prepared")
    args = parser.parse_args()
    seed_everything(42)
    tasks = args.tasks or TASKS
    paths = {task: args.data_dir / task / f"{args.subset}.jsonl" for task in tasks}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    runtime = InferenceRuntime(args)
    for task, path in paths.items():
        data = read_jsonl(path)
        if args.max_samples is not None:
            data = data[:args.max_samples]
        out_path = Path(args.save_dir) / f"{task}.jsonl"
        max_gen = load_task_config(task)["tokens_to_generate"]
        with out_path.open("w", encoding="utf-8") as handle:
            for sample in tqdm(data, desc=task):
                result = {key: sample.get(key) for key in (
                    "index", "input", "outputs", "others", "truncation", "length")}
                result["oom"] = False
                try:
                    prompt = sample["input"] + sample.get("answer_prefix", "")
                    result["pred"] = post_process(runtime.generate(prompt, max_gen), args.model)
                except RuntimeError as error:
                    if not isinstance(error, torch.cuda.OutOfMemoryError) and "out of memory" not in str(error).lower():
                        raise
                    if args.fail_on_oom:
                        raise
                    result.update(pred="", oom=True)
                    print(f"[OOM] {task}, index={sample['index']}", file=sys.stderr)
                release_sample_memory()
                append_prediction(handle, result)
    scores, average = evaluate_directory(args.save_dir, tasks)
    for row in [*scores, average]:
        print(f"{row['Task']}: score={row['Score']}, evaluated={row['Evaluated']}/{row['Total']}, OOM={row['OOM']}")

if __name__ == "__main__":
    main()
