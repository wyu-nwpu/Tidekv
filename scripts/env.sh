#!/usr/bin/env bash
# Shared paths; activate the desired Conda environment before running a script.
TIDEKV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${TIDEKV_ROOT}/config/local.env" ]]; then
  set -a
  source "${TIDEKV_ROOT}/config/local.env"
  set +a
fi
export PYTHON="${PYTHON:-python}"
export MODEL_NAME="${MODEL_NAME:-Llama-3.1-8B-Instruct-128k}"
export MODEL_PATH="${MODEL_PATH:-${TIDEKV_ROOT}/models/${MODEL_NAME%-*}}"
export LONGBENCH_DATA="${LONGBENCH_DATA:-${TIDEKV_ROOT}/dataset/longbench}"
export RULER_DATA="${RULER_DATA:-${TIDEKV_ROOT}/dataset/ruler/${MODEL_NAME}/data_shared}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
