#!/usr/bin/env python3
"""HeadInfer inference with CPU-offloaded KV caches."""

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
from src.pred_utils import post_process, post_process_code_completion


@torch.inference_mode()
def run_headinfer_record(model, tokenizer, record: InputRecord, args) -> PredictionRecord:
    cache = model._readkv_offloaded_cache_class()
    device = next(model.parameters()).device
    input_ids = torch.tensor([record.input_ids], dtype=torch.long, device=device)

    output = model.model(
        input_ids=input_ids,
        past_key_values=cache,
        use_cache=True,
        return_dict=True,
    )
    next_token = model.lm_head(output.last_hidden_state[:, -1]).argmax(
        dim=-1, keepdim=True
    )
    generated = []
    eos = set(record.eos_token_ids)
    for step in range(record.max_new_tokens):
        generated.append(next_token)
        if int(next_token.item()) in eos or step == record.max_new_tokens - 1:
            break
        output = model.model(
            input_ids=next_token,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        next_token = model.lm_head(output.last_hidden_state[:, -1]).argmax(
            dim=-1, keepdim=True
        )
    generated_ids = torch.cat(generated, dim=1)
    text = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
    text = (
        post_process_code_completion(text)
        if record.dataset in {"lcc", "repobench-p"}
        else post_process(text, args.model_name)
    )
    return PredictionRecord(
        sample_id=record.sample_id,
        pred=text,
        answers=record.answers,
        all_classes=record.all_classes,
        length=record.length,
        prompt_tokens=record.length,
        generated_tokens=int(generated_ids.shape[1]),
        finish_reason="eos" if int(generated_ids[0, -1]) in eos else "length",
    )


def load_headinfer_model(args):
    source_dir = str(Path(args.source_dir).resolve())
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
    from headinfer.cache import OffloadedCache
    from headinfer.mp import mp_headinfer

    if not torch.cuda.is_available():
        raise RuntimeError("HeadInfer backend requires CUDA")
    torch.cuda.set_device(args.gpu)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
    ).to(torch.device("cuda", args.gpu))
    model.eval()
    mp_headinfer(model)
    model._readkv_offloaded_cache_class = OffloadedCache
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    return model, tokenizer


def main(argv=None) -> int:
    args = parse_adapter_args("headinfer", argv)
    return run_adapter(args, load_headinfer_model, run_headinfer_record)


if __name__ == "__main__":
    raise SystemExit(main())
