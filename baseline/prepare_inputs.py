"""Prepare shared token-ID inputs for isolated external baselines."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import gzip
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Iterator
from uuid import uuid4

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from baseline.adapters.protocol import InputRecord
from pred_utils import build_chat, collect_eos_token_ids
from ruler_metrics import TASKS as RULER_TASKS, load_task_config

MODEL_ID = "Llama-3.1-8B-Instruct-128k"
NO_CHAT_DATASETS = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}
DEFAULT_LONGBENCH_TASKS = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "result" / "input_cache"
DEFAULT_LONGBENCH_DIR = REPO_ROOT / "dataset" / "longbench"
DEFAULT_RULER_DIR = REPO_ROOT / "dataset" / "ruler" / MODEL_ID / "data_shared"
DEFAULT_RULER_SEQUENCE_LENGTH = 4096
REQUEST_CACHE_SCHEMA = 2

def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as source:
        return json.load(source)

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]

def _model_path(args: argparse.Namespace) -> Path:
    value = getattr(args, "model_path", None)
    if value is None:
        raise ValueError("--model_path is required")
    return Path(value)

def _validate_llama_model(args: argparse.Namespace) -> Path:
    model_path = _model_path(args)
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise ValueError(f"Llama model validation requires {config_path}")
    config = _read_json(config_path)
    architectures = config.get("architectures", [])
    if config.get("model_type") != "llama":
        raise ValueError(f"Only Llama model paths are supported, got {model_path}")
    if architectures and (
        not isinstance(architectures, list)
        or any(not isinstance(name, str) or "Llama" not in name for name in architectures)
    ):
        raise ValueError(f"Only Llama model paths are supported, got {model_path}")
    return model_path

def _max_position_embeddings(args: argparse.Namespace) -> int:
    config_path = _model_path(args) / "config.json"
    config = _read_json(config_path)
    value = config.get("max_position_embeddings")
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(
            f"Llama model config requires a positive max_position_embeddings: {config_path}"
        )
    return value

def _load_tokenizer(args: argparse.Namespace, tokenizer: Any | None) -> Any:
    return tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(_model_path(args))

def _tasks(args: argparse.Namespace, available: Iterable[str]) -> list[str]:
    requested = getattr(args, "tasks", None)
    if requested is None:
        return list(available)
    if isinstance(requested, str):
        requested = [requested]
    tasks = []
    for item in requested:
        for raw_task in item.split(","):
            task = raw_task.strip()
            if not task:
                raise ValueError("empty task entry")
            tasks.append(task)
    if not tasks:
        raise ValueError("At least one task must be requested")
    duplicate_tasks = []
    seen_tasks = set()
    for task in tasks:
        if task in seen_tasks and task not in duplicate_tasks:
            duplicate_tasks.append(task)
        seen_tasks.add(task)
    if duplicate_tasks:
        raise ValueError(f"duplicate requested task(s): {', '.join(duplicate_tasks)}")
    unknown = sorted(set(tasks).difference(available))
    if unknown:
        raise ValueError(f"Unsupported task(s): {', '.join(unknown)}")
    return tasks

def _max_samples(args: argparse.Namespace) -> int | None:
    value = getattr(args, "max_samples", None)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("max_samples must be a positive integer")
    return value

def _token_ids(tokenizer: Any, prompt: str) -> list[int]:
    encoded = tokenizer(prompt, truncation=False)
    input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return [int(token_id) for token_id in input_ids]

def _decode(tokenizer: Any, input_ids: list[int]) -> str:
    return tokenizer.decode(input_ids, skip_special_tokens=True)

def _truncate_longbench_prompt(
    prompt: str, tokenizer: Any, max_length: int, dataset: str
) -> str:
    tokenized_prompt = _token_ids(tokenizer, prompt)
    if len(tokenized_prompt) <= max_length:
        return prompt
    if dataset in {"lcc", "repobench-p"}:
        return _decode(tokenizer, tokenized_prompt[-max_length:])
    half = max_length // 2
    return _decode(tokenizer, tokenized_prompt[:half]) + _decode(tokenizer, tokenized_prompt[-half:])

def _final_input_ids(
    tokenizer: Any, prompt: str, *, add_special_tokens: bool = True
) -> list[int]:
    encoded = tokenizer(
        [prompt], return_tensors="pt", add_special_tokens=add_special_tokens
    )
    return [int(token_id) for token_id in encoded.input_ids[0].tolist()]

def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    return [str(answer) for answer in value]

def _records_payload(records: list[InputRecord]) -> bytes:
    return b"".join(
        (
            json.dumps(
                record.to_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            + b"\n"
        )
        for record in records
    )

def _output_root(args: argparse.Namespace) -> Path:
    return Path(getattr(args, "output_dir", DEFAULT_OUTPUT_ROOT))

def _file_identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }

def _request_cache_spec(args: argparse.Namespace) -> dict[str, Any]:
    """Describe every cheap-to-check input that affects canonical token IDs."""
    benchmark = args.benchmark
    model_path = _validate_llama_model(args)
    model_files = []
    for name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
    ):
        path = model_path / name
        if path.is_file():
            model_files.append(_file_identity(path))

    behavior_files = [
        _file_identity(Path(__file__)),
        _file_identity(SRC_ROOT / "pred_utils.py"),
    ]
    common: dict[str, Any] = {
        "schema": REQUEST_CACHE_SCHEMA,
        "benchmark": benchmark,
        "model_files": model_files,
        "behavior_files": behavior_files,
        "max_samples": _max_samples(args),
    }
    if benchmark == "longbench":
        task_names = _tasks(args, DEFAULT_LONGBENCH_TASKS)
        data_dir = Path(getattr(args, "data_dir", None) or DEFAULT_LONGBENCH_DIR)
        data_files = []
        for dataset in task_names:
            filename = f"{dataset}_e.jsonl" if getattr(args, "e", False) else f"{dataset}.jsonl"
            path = data_dir / filename
            if getattr(args, "e", False) and not path.exists():
                path = data_dir / f"{dataset}.jsonl"
            data_files.append(_file_identity(path))
        common.update({
            "tasks": task_names,
            "data_files": data_files,
            "max_prompt_tokens": _longbench_max_prompt_tokens(args),
            "extended": bool(getattr(args, "e", False)),
            "prompt_config": _file_identity(REPO_ROOT / "config" / "dataset2prompt.json"),
            "generation_config": _file_identity(REPO_ROOT / "config" / "dataset2maxlen.json"),
        })
        return common

    task_names = _tasks(args, RULER_TASKS)
    common.update({
        "tasks": task_names,
        "data_files": [_file_identity(_ruler_data_path(args, task)) for task in task_names],
        "sequence_length": int(getattr(args, "sequence_length", DEFAULT_RULER_SEQUENCE_LENGTH)),
        "subset": getattr(args, "subset", "validation"),
        "combine_tasks": bool(getattr(args, "combine_tasks", False)),
        "synthetic_config": _file_identity(
            REPO_ROOT / "dataset" / "ruler" / "RULER-main" / "scripts" / "synthetic.yaml"
        ),
        "synthetic_constants": _file_identity(
            REPO_ROOT / "dataset" / "ruler" / "RULER-main" / "scripts" / "data" / "synthetic" / "constants.py"
        ),
    })
    return common

def _request_cache_path(args: argparse.Namespace, spec: dict[str, Any]) -> Path:
    defaults = DEFAULT_LONGBENCH_TASKS if args.benchmark == "longbench" else list(RULER_TASKS)
    tasks = "all" if spec["tasks"] == defaults else "+".join(spec["tasks"])
    length = str(spec.get("sequence_length", "all"))
    samples = spec["max_samples"] if spec["max_samples"] is not None else "all"
    return (
        _output_root(args)
        / MODEL_ID
        / args.benchmark
        / "_requests"
        / length
        / f"samples_{samples}"
        / f"{tasks}.json"
    )

def _cached_outputs(index_path: Path, spec: dict[str, Any]) -> list[Path] | None:
    if not index_path.is_file():
        return None
    try:
        cache = _read_json(index_path)
        outputs = [Path(value) for value in cache["outputs"]]
        if cache.get("schema") != REQUEST_CACHE_SCHEMA or cache.get("request") != spec or not outputs:
            return None
        for output in outputs:
            if not output.is_file():
                return None
            with gzip.open(output, "rb") as source:
                source.read(1)
        return outputs
    except (EOFError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None

def _write_request_cache(index_path: Path, spec: dict[str, Any], outputs: list[Path]) -> None:
    payload = json.dumps(
        {
            "schema": REQUEST_CACHE_SCHEMA,
            "request": spec,
            "outputs": [str(path.resolve()) for path in outputs],
        },
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    ) + "\n"
    temporary_path = index_path.with_name(f".{index_path.name}.{uuid4().hex}.tmp")
    try:
        temporary_path.write_text(payload, encoding="utf-8")
        temporary_path.replace(index_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

def prepare_cached(args: argparse.Namespace, tokenizer: Any | None = None) -> list[Path]:
    """Reuse inputs when parameters and file metadata match, without content hashes."""
    spec = _request_cache_spec(args)
    index_path = _request_cache_path(args, spec)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with _publication_lock(index_path):
        if not getattr(args, "rebuild_inputs", False):
            cached = _cached_outputs(index_path, spec)
            if cached is not None:
                return cached
        if args.benchmark == "longbench":
            outputs = [prepare_longbench(args, tokenizer)]
        else:
            outputs = prepare_ruler(args, tokenizer)
        _write_request_cache(index_path, spec, outputs)
        return outputs

@contextmanager
def _publication_lock(output_path: Path) -> Iterator[None]:
    lock_path = output_path.with_name(f".{output_path.name}.lock")
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)

def _write_manifest(
    args: argparse.Namespace, benchmark: str, records: list[InputRecord]
) -> Path:
    payload = _records_payload(records)
    # A fresh version keeps running readers safe when a cached request is rebuilt.
    input_id = uuid4().hex
    output_path = _output_root(args) / MODEL_ID / benchmark / input_id / "inputs.jsonl.gz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with temporary_path.open("wb") as raw_output:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw_output, mtime=0) as compressed:
                compressed.write(payload)
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return output_path

def _longbench_max_prompt_tokens(args: argparse.Namespace) -> int:
    override = getattr(args, "max_prompt_tokens", None)
    if override is not None:
        if override <= 0:
            raise ValueError("max_prompt_tokens must be positive")
        return int(override)
    model2maxlen = _read_json(REPO_ROOT / "config" / "model2maxlen.json")
    return int(model2maxlen[MODEL_ID])

def prepare_longbench(args: argparse.Namespace, tokenizer: Any | None = None) -> Path:
    """Write one canonical LongBench token manifest."""
    _validate_llama_model(args)
    prompts = _read_json(REPO_ROOT / "config" / "dataset2prompt.json")
    max_new_tokens = _read_json(REPO_ROOT / "config" / "dataset2maxlen.json")
    task_names = _tasks(args, DEFAULT_LONGBENCH_TASKS)
    max_samples = _max_samples(args)
    tokenizer = _load_tokenizer(args, tokenizer)
    max_length = _longbench_max_prompt_tokens(args)
    data_dir = Path(getattr(args, "data_dir", None) or DEFAULT_LONGBENCH_DIR)
    eos_token_ids = sorted(collect_eos_token_ids(tokenizer))
    records: list[InputRecord] = []

    for dataset in task_names:
        filename = f"{dataset}_e.jsonl" if getattr(args, "e", False) else f"{dataset}.jsonl"
        data_path = data_dir / filename
        if getattr(args, "e", False) and not data_path.exists():
            data_path = data_dir / f"{dataset}.jsonl"
        for index, sample in enumerate(_read_jsonl(data_path)[:max_samples]):
            raw_prompt = prompts[dataset].format(**sample)
            prompt = _truncate_longbench_prompt(raw_prompt, tokenizer, max_length, dataset)
            if dataset not in NO_CHAT_DATASETS:
                prompt = build_chat(tokenizer, prompt, MODEL_ID)
            input_ids = _final_input_ids(tokenizer, prompt, add_special_tokens=not prompt.startswith("<|begin_of_text|>"))
            records.append(
                InputRecord(
                    sample_id=f"{dataset}:{index}",
                    dataset=dataset,
                    input_ids=input_ids,
                    max_new_tokens=int(max_new_tokens[dataset]),
                    eos_token_ids=eos_token_ids,
                    answers=_string_list(sample.get("answers")),
                    all_classes=_string_list(sample.get("all_classes")),
                    length=len(input_ids),
                )
            )
    return _write_manifest(args, "longbench", records)

def _ruler_data_path(args: argparse.Namespace, task: str) -> Path:
    data_dir = Path(getattr(args, "data_dir", None) or DEFAULT_RULER_DIR)
    subset = getattr(args, "subset", "validation")
    direct_path = data_dir / task / f"{subset}.jsonl"
    if direct_path.is_file():
        return direct_path
    sequence_length = getattr(args, "sequence_length", None)
    if sequence_length is None:
        sequence_length = DEFAULT_RULER_SEQUENCE_LENGTH
    return data_dir / str(sequence_length) / task / f"{subset}.jsonl"

def _is_prepared_llama_prompt(prompt: str) -> bool:
    return prompt.startswith("<|begin_of_text|><|start_header_id|>")

def prepare_ruler(args: argparse.Namespace, tokenizer: Any | None = None) -> list[Path]:
    """Write canonical RULER token manifests, optionally combining tasks."""
    _validate_llama_model(args)
    max_position_embeddings = _max_position_embeddings(args)
    tokenizer = _load_tokenizer(args, tokenizer)
    eos_token_ids = sorted(collect_eos_token_ids(tokenizer))
    task_names = _tasks(args, RULER_TASKS)
    max_samples = _max_samples(args)
    combine_tasks = getattr(args, "combine_tasks", False)
    combined_records: list[InputRecord] = []
    outputs: list[Path] = []

    for task in task_names:
        task_config = load_task_config(task)
        generation_length = int(task_config["tokens_to_generate"])
        records: list[InputRecord] = []
        for index, sample in enumerate(_read_jsonl(_ruler_data_path(args, task))[:max_samples]):
            prompt = sample["input"]
            if not _is_prepared_llama_prompt(prompt):
                prompt = build_chat(tokenizer, prompt, MODEL_ID)
            answer_prefix = sample.get("answer_prefix", "")
            if answer_prefix and not prompt.endswith(answer_prefix):
                prompt += answer_prefix
            input_ids = _final_input_ids(
                tokenizer, prompt, add_special_tokens=False
            )
            sample_id = f"{task}:{sample.get('index', index)}"
            if len(input_ids) + generation_length > max_position_embeddings:
                raise ValueError(
                    "RULER context limit exceeded: "
                    f"task={task} sample_id={sample_id} input_length={len(input_ids)} "
                    f"generation_length={generation_length} "
                    f"model_limit={max_position_embeddings}"
                )
            records.append(
                InputRecord(
                    sample_id=sample_id,
                    dataset=task,
                    input_ids=input_ids,
                    max_new_tokens=generation_length,
                    eos_token_ids=eos_token_ids,
                    answers=_string_list(sample.get("outputs")),
                    all_classes=[],
                    length=len(input_ids),
                )
            )
        if combine_tasks:
            combined_records.extend(records)
        else:
            outputs.append(_write_manifest(args, "ruler", records))
    if combine_tasks:
        outputs.append(_write_manifest(args, "ruler", combined_records))
    return outputs

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=["longbench", "ruler"])
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_prompt_tokens", type=int, default=None)
    parser.add_argument("--sequence_length", type=int, default=DEFAULT_RULER_SEQUENCE_LENGTH)
    parser.add_argument("--subset", default="validation")
    parser.add_argument("--combine_tasks", action="store_true")
    parser.add_argument(
        "--rebuild_inputs",
        action="store_true",
        help="Ignore the request index and rebuild canonical token inputs.",
    )
    parser.add_argument("--e", action="store_true")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    for output in prepare_cached(args):
        print(output)

if __name__ == "__main__":
    main()
