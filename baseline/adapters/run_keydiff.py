#!/usr/bin/env python3
"""KeyDiff-BPP inference on shared benchmark inputs."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.adapters.backend_common import parse_adapter_args, run_adapter
from baseline.adapters.protocol import InputRecord, PredictionRecord
from baseline.keydiff.bpp.config import BaselineSpec
from baseline.keydiff.bpp.engine import BPPGenerationEngine
from baseline.keydiff.bpp.policies.keydiff import KeyDiffPolicy
from src.pred_utils import post_process, post_process_code_completion


def run_keydiff_record(model, tokenizer, record: InputRecord, args) -> PredictionRecord:
    spec = BaselineSpec(
        name="keydiff_bpp",
        kv_budget=args.kv_budget,
        block_size=args.block_size,
        sink_tokens=0,
        keydiff_anchor="raw_mean",
        score_dtype="fp32",
    )
    spec.validate()
    engine = BPPGenerationEngine(model, spec, KeyDiffPolicy(spec))
    device = next(model.parameters()).device
    input_ids = torch.tensor([record.input_ids], dtype=torch.long, device=device)
    output = engine.generate(input_ids, record.max_new_tokens, record.eos_token_ids)
    generated = output.generated_ids
    text = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
    text = (
        post_process_code_completion(text)
        if record.dataset in {"lcc", "repobench-p"}
        else post_process(text, args.model_name)
    )
    generated_tokens = int(generated.shape[1])
    final_token = int(generated[0, -1]) if generated_tokens else None
    return PredictionRecord(
        sample_id=record.sample_id,
        pred=text,
        answers=record.answers,
        all_classes=record.all_classes,
        length=record.length,
        prompt_tokens=record.length,
        generated_tokens=generated_tokens,
        finish_reason="eos" if final_token in set(record.eos_token_ids) else "length",
    )


def load_keydiff_model(args):
    if not torch.cuda.is_available():
        raise RuntimeError("KeyDiff backend requires CUDA")
    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda", args.gpu)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        # Physical cache positions use the engine's eager causal-mask path.
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return model, tokenizer


def main(argv=None) -> int:
    args = parse_adapter_args("keydiff", argv)
    return run_adapter(args, load_keydiff_model, run_keydiff_record)


if __name__ == "__main__":
    raise SystemExit(main())
