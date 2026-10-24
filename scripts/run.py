"""Run inference and scoring with one active method/budget per GPU."""

import argparse
import os
from pathlib import Path
import queue
import shlex
import signal
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from baseline_config import PAIRS, SUPPORTED_CACHE_TYPES

EXTERNAL = {"kivi", "headinfer", "keydiff"}
DEFAULT_METHODS = [method for pair in PAIRS.items() for method in pair]
ACTIVE = set()
LOCK = threading.Lock()
STOP = threading.Event()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Llama-3.1-8B-Instruct-128k",
                        choices=["Llama-3.1-8B-Instruct-128k", "Mistral-7B-Instruct-v0.2-32k"])
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS,
                        choices=[*SUPPORTED_CACHE_TYPES, *sorted(EXTERNAL)])
    parser.add_argument("--budgets", nargs="+", type=int, default=[256, 1024, 4096])
    parser.add_argument("--datasets", nargs="+", choices=["longbench", "ruler"], default=["longbench", "ruler"])
    parser.add_argument("--longbench_data", type=Path, default=ROOT / "dataset/longbench")
    parser.add_argument("--ruler_data", type=Path)
    parser.add_argument("--lengths", nargs="+", type=int)
    parser.add_argument("--longbench_tasks", nargs="+")
    parser.add_argument("--ruler_tasks", nargs="+")
    parser.add_argument("--ruler_num_samples", type=int, default=100)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--output", type=Path, default=ROOT / "result")
    parser.add_argument("--run_name", default="default")
    parser.add_argument("--chunk_size", type=int, default=512, help="TideKV prefill chunk size")
    parser.add_argument("--window_size", type=int)
    parser.add_argument("--kernel_size", type=int)
    parser.add_argument("--pooling", choices=["avgpool", "maxpool"])
    parser.add_argument("--chunk_length", type=int, default=20, help="ChunkKV selection chunk length")
    parser.add_argument("--reuse_layers", type=int, default=2, help="ChunkKV index reuse interval")
    parser.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="bf16",
                        help="Precision for built-in methods; external backends use fixed dtypes")
    parser.add_argument("--backend", choices=["flash_attention_2", "sdpa", "eager"], default="flash_attention_2",
                        help="Attention backend for built-in methods only")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus) or not all(gpu.isdigit() for gpu in args.gpus):
        parser.error("GPU IDs must be distinct nonnegative integers")
    if any(value < 1 for value in [*args.budgets, args.chunk_size, args.chunk_length, args.reuse_layers, args.ruler_num_samples]):
        parser.error("Budgets, sizes, and sample counts must be positive")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("max_samples must be positive")
    if Path(args.run_name).name != args.run_name or args.run_name in {".", ".."}:
        parser.error("run_name must be one directory name")
    args.methods = list(dict.fromkeys("tidekv_snap" if method == "tidekv" else method for method in args.methods))
    args.budgets = list(dict.fromkeys(args.budgets))
    args.ruler_data = args.ruler_data or ROOT / "dataset/ruler" / args.model / "data_shared"
    defaults = [4096, 8192, 16384, 32768]
    args.lengths = args.lengths or (defaults if args.model.startswith("Mistral") else [*defaults, 65536, 131072])
    if args.model.startswith("Mistral") and any(length > 32768 for length in args.lengths):
        parser.error("Specify Mistral RULER lengths at or below 32768")
    if any(length not in {4096, 8192, 16384, 32768, 65536, 131072} for length in args.lengths):
        parser.error("Unsupported RULER length")
    if args.model.startswith("Mistral") and (set(args.methods) & (EXTERNAL | {"tidekv_adakv"})):
        parser.error("The isolated backends and TideKV-AdaKV support Llama only")
    for name in ("model_path", "longbench_data", "ruler_data", "output"):
        setattr(args, name, getattr(args, name).resolve())
    return args


def run_command(command, env, log, capture=False):
    if STOP.is_set():
        raise InterruptedError("Run interrupted")
    log.write(shlex.join(command) + "\n")
    log.flush()
    with LOCK:
        if STOP.is_set():
            raise InterruptedError("Run interrupted")
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE if capture else log,
                                   stderr=log, text=True, start_new_session=True)
        ACTIVE.add(process)
    try:
        output, _ = process.communicate()
        if process.returncode:
            raise RuntimeError(f"Command exited {process.returncode}; see {log.name}")
        return output.strip().splitlines()[-1] if capture else None
    finally:
        with LOCK:
            ACTIVE.discard(process)


def builtin_commands(args, method, budget, benchmark, save_dir, length=None):
    command = [sys.executable, str(ROOT / "src" / ("pred.py" if benchmark == "longbench" else "pred_ruler.py")),
               "--model", args.model, "--model_path", str(args.model_path), "--cache_type", method,
               "--budget", str(budget or 1024), "--save_dir", str(save_dir),
               "--chunk_size", str(args.chunk_size), "--chunk_length", str(args.chunk_length),
               "--chunkkv_reuse_layers", str(args.reuse_layers), "--precision", args.precision,
               "--attn_implementation", args.backend]
    for flag, value in (("--window_size", args.window_size), ("--kernel_size", args.kernel_size),
                        ("--pooling", args.pooling), ("--max_samples", args.max_samples)):
        if value is not None:
            command += [flag, str(value)]
    if benchmark == "longbench":
        command += ["--dataset_dir", str(args.longbench_data)]
        if args.longbench_tasks:
            command += ["--tasks", *args.longbench_tasks]
        return [command, [sys.executable, str(ROOT / "src/eval.py"), "--save_dir", str(save_dir)]]
    command += ["--data_dir", str(args.ruler_data / str(length))]
    if args.ruler_tasks:
        command += ["--tasks", *args.ruler_tasks]
    return [command]


def external_commands(args, method, budget, benchmark, save_dir, length, env, log):
    prepare = [sys.executable, str(ROOT / "baseline/prepare_inputs.py"), "--benchmark", benchmark,
               "--model_path", str(args.model_path), "--output_dir", str(args.output / "input_cache"),
               "--data_dir", str(args.longbench_data if benchmark == "longbench" else args.ruler_data)]
    tasks = args.longbench_tasks if benchmark == "longbench" else args.ruler_tasks
    if tasks:
        prepare += ["--tasks", *tasks]
    if args.max_samples is not None:
        prepare += ["--max_samples", str(args.max_samples)]
    if benchmark == "ruler":
        prepare += ["--sequence_length", str(length), "--combine_tasks"]
    manifest = run_command(prepare, env, log, capture=True)
    command = [sys.executable, str(ROOT / "baseline/run.py"), "--method", method,
               "--phase", benchmark, "--gpu", "0",
               "--model_name", args.model, "--model_path", str(args.model_path),
               "--input_manifest", manifest, "--save_dir", str(save_dir)]
    if method == "keydiff":
        command += ["--kv_budget", str(budget), "--block_size", "128"]
    return [command, [sys.executable, str(ROOT / "baseline/evaluate.py"), "--run_dir", str(save_dir),
                       "--benchmark", benchmark]]


def run_unit(args, gpu, method, budget):
    label = f"{method}_{budget if budget is not None else 'native'}"
    log_dir = args.output / "logs" / args.run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    with (log_dir / f"gpu{gpu}_{label}.log").open("w") as log:
        for benchmark in args.datasets:
            for length in args.lengths if benchmark == "ruler" else [None]:
                save = args.output / benchmark / args.model / args.run_name / method
                save /= f"budget_{budget}" if budget is not None else "native"
                if args.max_samples is not None:
                    save /= f"samples_{args.max_samples}"
                if length is not None:
                    save /= str(length)
                commands = external_commands(args, method, budget, benchmark, save, length, env, log) if method in EXTERNAL else builtin_commands(args, method, budget, benchmark, save, length)
                for command in commands:
                    run_command(command, env, log)


def interrupt(signum, _frame):
    STOP.set()
    with LOCK:
        for process in ACTIVE:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def main():
    args = parse_args()
    units = [(method, budget) for method in args.methods
             for budget in ([None] if method in {"full", "kivi", "headinfer"} else args.budgets)]
    print(f"{len(units)} units; GPUs: {', '.join(args.gpus)}; datasets: {', '.join(args.datasets)}")
    for method, budget in units:
        print(f"  {method}: {budget if budget is not None else 'native (no token budget)'}")
    print(f"Output: {args.output}; run: {args.run_name}")
    if not args.execute:
        print("Dry run. Add --execute, or EXECUTE=1 with scripts/run.sh.")
        return 0
    if "ruler" in args.datasets:
        from dataset.prepare_ruler import RULER_TASKS, validate_shared_data
        print(validate_shared_data(args.ruler_data, args.lengths, args.ruler_tasks or RULER_TASKS,
                                   args.ruler_num_samples, args.model))
    work = queue.Queue()
    for unit in units:
        work.put(unit)
    failures = []

    def worker(gpu):
        while not STOP.is_set():
            try:
                method, budget = work.get_nowait()
            except queue.Empty:
                return
            print(f"[start] GPU {gpu}: {method}, budget={budget}", flush=True)
            try:
                run_unit(args, gpu, method, budget)
                print(f"[done] GPU {gpu}: {method}, budget={budget}", flush=True)
            except Exception as error:
                failures.append((method, budget, str(error)))
                print(f"[failed] GPU {gpu}: {method}: {error}", file=sys.stderr, flush=True)

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    # Each worker owns one GPU and finishes a unit before taking the next one.
    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in args.gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return 130 if STOP.is_set() else int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
