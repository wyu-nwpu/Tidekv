#!/usr/bin/env python3
"""Run an external baseline in its configured Python environment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.adapters.common_io import atomic_write_json
from baseline.registry import ExternalBackend, get_backend

METHODS = ("kivi", "headinfer", "keydiff")
TARGET_MODEL_NAME = "Llama-3.1-8B-Instruct-128k"
TARGET_MODEL_FIELDS = {
    "model_type": "llama",
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
}
TERMINATION_GRACE_SECONDS = 10.0


def validate_args(args):
    if args.model_name != TARGET_MODEL_NAME:
        raise ValueError(f"--model_name must equal {TARGET_MODEL_NAME}")
    if args.gpu < 0 or (args.max_samples is not None and args.max_samples < 1):
        raise ValueError("GPU index must be nonnegative and max_samples must be positive")
    kivi_fields = ("k_bits", "v_bits", "group_size", "residual_length")
    keydiff_fields = ("kv_budget", "block_size")
    selected = kivi_fields if args.method == "kivi" else keydiff_fields if args.method == "keydiff" else ()
    for field in (*kivi_fields, *keydiff_fields):
        value = getattr(args, field)
        if field not in selected and value is not None:
            raise ValueError(f"{args.method} does not accept --{field}")
        if field in selected and value < 1:
            raise ValueError(f"--{field} must be positive")
    if args.method == "kivi" and args.residual_length % args.group_size:
        raise ValueError("KIVI residual_length must be a multiple of group_size")


def effective_config_id(args, backend):
    if args.method == "kivi":
        return f"k{args.k_bits}_v{args.v_bits}_g{args.group_size}_r{args.residual_length}"
    if args.method == "keydiff":
        return f"n{args.kv_budget}_b{args.block_size}_raw_mean"
    return backend.config_id


def backend_python(args, backend):
    return str(args.backend_python) if args.backend_python else backend.python


def build_backend_command(args, backend: ExternalBackend):
    command = [
        backend_python(args, backend), str(REPO_ROOT / backend.adapter),
        "--phase", args.phase, "--gpu", str(args.gpu),
        "--model_name", args.model_name, "--model_path", str(args.model_path),
        "--input_manifest", str(args.input_manifest), "--dataset", args.dataset,
        "--save_dir", str(args.save_dir),
    ]
    if args.max_samples is not None:
        command += ["--max_samples", str(args.max_samples)]
    fields = ("k_bits", "v_bits", "group_size", "residual_length") if args.method == "kivi" else (
        ("kv_budget", "block_size") if args.method == "keydiff" else ()
    )
    for field in fields:
        command += [f"--{field}", str(getattr(args, field))]
    return command


def _validate_llama_metadata(model_path: Path) -> None:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise ValueError(f"model metadata is missing {config_path}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid model metadata {config_path}") from error
    for field, expected in TARGET_MODEL_FIELDS.items():
        if config.get(field) != expected:
            raise ValueError(
                f"Llama 3.1 8B model configuration requires {field}={expected}, got {config.get(field)!r}"
            )
    architectures = config.get("architectures")
    if architectures is not None and (
        not isinstance(architectures, list) or architectures != ["LlamaForCausalLM"]
    ):
        raise ValueError("Llama 3.1 8B requires LlamaForCausalLM")

def run_backend(command: Sequence[str], output_dir: Path) -> int:
    """Run an adapter in its own session and forward interruption to all children."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "stdout.log").open("w", encoding="utf-8") as stdout, (
        output_dir / "stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            list(command),
            cwd=REPO_ROOT,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        previous_handlers: dict[int, object] = {}
        forwarded_signals: list[int] = []

        def forward_signal(signum: int, _frame: object) -> None:
            if process.poll() is None:
                forwarded_signals.append(signum)
                try:
                    os.killpg(process.pid, signum)
                except ProcessLookupError:
                    pass

        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, forward_signal)
        deadline: float | None = None
        try:
            while process.poll() is None:
                try:
                    returncode = process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    if forwarded_signals:
                        deadline = deadline or time.monotonic() + TERMINATION_GRACE_SECONDS
                        if time.monotonic() >= deadline:
                            os.killpg(process.pid, signal.SIGKILL)
                            returncode = process.wait()
                            break
                    continue
                else:
                    break
            else:
                returncode = process.returncode
            return -forwarded_signals[-1] if forwarded_signals else returncode
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            for signum, previous in previous_handlers.items():
                signal.signal(signum, previous)

def _write_dispatcher_status(
    args: argparse.Namespace,
    config_id: str,
    *,
    stage: str,
    exit_code: int | None,
    error: str | None,
) -> None:
    atomic_write_json(
        Path(args.save_dir) / "dispatcher_status.json",
        {
            "method": args.method,
            "config_id": config_id,
            "stage": stage,
            "exit": exit_code,
            "error": error,
        },
    )

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--phase", required=True, choices=("longbench", "ruler"))
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--model_name", default=TARGET_MODEL_NAME)
    parser.add_argument("--model_path", required=True, type=Path)
    parser.add_argument("--input_manifest", required=True, type=Path)
    parser.add_argument("--dataset")
    parser.add_argument("--save_dir", required=True, type=Path)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--k_bits", type=int, choices=(2, 4))
    parser.add_argument("--v_bits", type=int, choices=(2, 4))
    parser.add_argument("--group_size", type=int)
    parser.add_argument("--residual_length", type=int)
    parser.add_argument("--kv_budget", type=int)
    parser.add_argument("--block_size", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--backend_python", type=Path)
    args = parser.parse_args(argv)
    if args.method == "kivi":
        for field, default in (("k_bits", 2), ("v_bits", 2), ("group_size", 32), ("residual_length", 32)):
            if getattr(args, field) is None:
                setattr(args, field, default)
    elif args.method == "keydiff":
        args.kv_budget = 4096 if args.kv_budget is None else args.kv_budget
        args.block_size = 128 if args.block_size is None else args.block_size
    args.dataset = args.dataset or args.phase
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    config_id = "unresolved"
    try:
        validate_args(args)
        backend = get_backend(args.method)
        config_id = effective_config_id(args, backend)
        command = build_backend_command(args, backend)
        if args.dry_run:
            print(shlex.join(command))
            return 0
        _validate_llama_metadata(args.model_path)
    except (ValueError, KeyError) as error:
        _write_dispatcher_status(args, config_id, stage="configuration", exit_code=2, error=str(error))
        print(f"error: {error}", file=sys.stderr)
        return 2
    try:
        _write_dispatcher_status(args, config_id, stage="running", exit_code=None, error=None)
        exit_code = run_backend(command, args.save_dir)
    except OSError as error:
        _write_dispatcher_status(args, config_id, stage="child", exit_code=1, error=str(error))
        print(f"error: {error}", file=sys.stderr)
        return 1
    stage = "success" if exit_code == 0 else "signal" if exit_code < 0 else "child"
    error = None if exit_code == 0 else (
        f"child terminated by signal {-exit_code}" if exit_code < 0 else f"child exited {exit_code}"
    )
    _write_dispatcher_status(args, config_id, stage=stage, exit_code=exit_code, error=error)
    return 128 + (-exit_code) if exit_code < 0 else exit_code


if __name__ == "__main__":
    raise SystemExit(main())
