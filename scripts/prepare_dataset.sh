#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
kind="${1:-}"
[[ $# -gt 0 ]] && shift
case "${kind}" in
  longbench)
    exec "${PYTHON}" "${TIDEKV_ROOT}/dataset/prepare_longbench.py" --output "${LONGBENCH_DATA}" "$@" ;;
  ruler-resources)
    exec "${PYTHON}" "${TIDEKV_ROOT}/dataset/prepare_ruler_resources.py" "$@" ;;
  ruler)
    defaults="4096 8192 16384 32768"
    if [[ "${MODEL_NAME}" != Mistral* ]]; then
      defaults="${defaults} 65536 131072"
    fi
    read -r -a lengths <<< "${RULER_LENGTHS:-${defaults}}"
    exec "${PYTHON}" "${TIDEKV_ROOT}/dataset/prepare_ruler.py" prepare \
      --data_dir "${RULER_DATA}" --tokenizer_path "${MODEL_PATH}" --model_id "${MODEL_NAME}" \
      --lengths "${lengths[@]}" --num_samples "${NUM_SAMPLES:-100}" --seed "${SEED:-42}" "$@" ;;
  *) echo 'Usage: bash scripts/prepare_dataset.sh {longbench|ruler-resources|ruler} [options]' >&2; exit 2 ;;
esac
