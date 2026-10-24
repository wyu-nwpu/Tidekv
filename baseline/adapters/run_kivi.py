#!/usr/bin/env python3
"""KIVI inference on shared benchmark inputs."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
from transformers import AutoTokenizer, LlamaConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.adapters.backend_common import parse_adapter_args, run_adapter
from baseline.adapters.protocol import InputRecord, PredictionRecord
from src.pred_utils import post_process, post_process_code_completion


@torch.inference_mode()
def run_kivi_record(model, tokenizer, record: InputRecord, args) -> PredictionRecord:
    device = next(model.parameters()).device
    input_ids = torch.tensor([record.input_ids], dtype=torch.long, device=device)
    # Project only the last hidden state; full-prompt logits are not needed.
    backbone = model.model(input_ids=input_ids, use_cache=True, return_dict=True)
    logits = model.lm_head(backbone.last_hidden_state[:, -1]).float()
    cache = backbone.past_key_values
    next_token = logits.argmax(dim=-1, keepdim=True)

    generated = []
    eos = set(record.eos_token_ids)
    for step in range(record.max_new_tokens):
        generated.append(next_token)
        if int(next_token.item()) in eos or step == record.max_new_tokens - 1:
            break
        backbone = model.model(
            input_ids=next_token,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = backbone.past_key_values
        next_token = model.lm_head(backbone.last_hidden_state[:, -1]).float().argmax(
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


def load_kivi_model(args):
    source_dir = str(Path(args.source_dir).resolve())
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
    from models.llama_kivi import LlamaForCausalLM_KIVI

    if not torch.cuda.is_available():
        raise RuntimeError("KIVI backend requires CUDA")
    torch.cuda.set_device(args.gpu)
    config = LlamaConfig.from_pretrained(args.model_path)
    config.k_bits = args.k_bits
    config.v_bits = args.v_bits
    config.group_size = args.group_size
    config.residual_length = args.residual_length
    config.use_flash = True
    model = LlamaForCausalLM_KIVI.from_pretrained(
        args.model_path,
        config=config,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    ).to(torch.device("cuda", args.gpu))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    return model, tokenizer


def main(argv=None) -> int:
    args = parse_adapter_args("kivi", argv)
    return run_adapter(args, load_kivi_model, run_kivi_record)


if __name__ == "__main__":
    raise SystemExit(main())
