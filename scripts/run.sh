#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
read -r -a gpus <<< "${GPUS:-0}"
read -r -a methods <<< "${METHODS:-snap_a tidekv_snap pyramidkv_a tidekv_pyramidkv adakv tidekv_adakv chunkkv tidekv_chunkkv}"
read -r -a budgets <<< "${BUDGETS:-256 1024 4096}"
read -r -a datasets <<< "${DATASETS:-longbench ruler}"
args=(--model "${MODEL_NAME}" --model_path "${MODEL_PATH}"
  --gpus "${gpus[@]}" --methods "${methods[@]}" --budgets "${budgets[@]}"
  --datasets "${datasets[@]}" --longbench_data "${LONGBENCH_DATA}" --ruler_data "${RULER_DATA}"
  --output "${OUTPUT_ROOT:-${TIDEKV_ROOT}/result}" --run_name "${RUN_NAME:-default}"
  --chunk_size "${CHUNK_SIZE:-512}" --chunk_length "${CHUNK_LENGTH:-20}"
  --reuse_layers "${CHUNKKV_REUSE_LAYERS:-2}" --precision "${PRECISION:-bf16}"
  --backend "${ATTN_IMPLEMENTATION:-flash_attention_2}" --ruler_num_samples "${RULER_NUM_SAMPLES:-100}")
[[ "${EXECUTE:-0}" == 1 ]] && args+=(--execute)
[[ -n "${MAX_SAMPLES:-}" ]] && args+=(--max_samples "${MAX_SAMPLES}")
[[ -n "${WINDOW_SIZE:-}" ]] && args+=(--window_size "${WINDOW_SIZE}")
[[ -n "${KERNEL_SIZE:-}" ]] && args+=(--kernel_size "${KERNEL_SIZE}")
[[ -n "${POOLING:-}" ]] && args+=(--pooling "${POOLING}")
if [[ -n "${RULER_LENGTHS:-}" ]]; then
  read -r -a lengths <<< "${RULER_LENGTHS}"
  args+=(--lengths "${lengths[@]}")
fi
exec "${PYTHON}" "${TIDEKV_ROOT}/scripts/run.py" "${args[@]}" "$@"
