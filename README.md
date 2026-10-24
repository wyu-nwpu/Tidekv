# TideKV

Layer-major chunked prefill with post-layer KV selection. Each layer processes
the full causal prompt before its cache is compressed for decoding.

The implementation starts at `readkv/tidekv.py:tidekv_prefill`.
Inference and scoring support LongBench and RULER.

## Usage

Prepare a Python 3.10 environment with the dependencies in `requirements.txt`
and `envs/`.

```bash
export MODEL_PATH=/path/to/Llama-3.1-8B-Instruct
```

Prepare inputs with `scripts/prepare_dataset.sh`; see [datasets](dataset/README.md).

```bash
GPUS="0 1" METHODS="snap_a tidekv_snap" BUDGETS="256 1024 4096" \
DATASETS=longbench EXECUTE=1 bash scripts/run.sh
```

Each GPU runs one method/budget at a time. Use a separate `RUN_NAME` for each run.
OOM samples are skipped and reported alongside the scores.

## Methods

| Baseline | TideKV variant |
| --- | --- |
| `snap_a` | `tidekv_snap` (`tidekv`) |
| `pyramidkv_a` | `tidekv_pyramidkv` |
| `adakv` | `tidekv_adakv` |
| `chunkkv` | `tidekv_chunkkv` |

Snap-A and Pyramid-A use compact GQA group-consensus selection. Paired methods
share parameter defaults and one budget interface. Full cache, H2O,
StreamingLLM, KIVI, HeadInfer, and KeyDiff are optional baselines.

The primary model is Llama-3.1-8B-Instruct. Built-in Mistral support excludes
adaptive-head TideKV. KIVI, HeadInfer, and KeyDiff use isolated Llama environments
configured for A100.
