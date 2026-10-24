# Scripts

Run from the project root with the runtime environment active and `MODEL_PATH` set.
Optional local defaults go in `config/local.env`.

| Script | Purpose |
| --- | --- |
| `prepare_dataset.sh` | Prepare `longbench`, `ruler-resources`, or `ruler` |
| `run.sh` / `run.py` | Schedule inference and scoring across GPUs |
| `env.sh` | Shared paths and environment defaults |

```bash
GPUS="0 1" METHODS="chunkkv tidekv_chunkkv" BUDGETS="256 1024 4096" \
DATASETS=ruler RULER_LENGTHS="65536 131072" WINDOW_SIZE=32 \
RUN_NAME=chunk_w32 EXECUTE=1 bash scripts/run.sh

GPUS="0 1 2" METHODS="kivi headinfer keydiff" DATASETS=longbench \
BUDGETS=4096 EXECUTE=1 bash scripts/run.sh

METHODS=tidekv_snap BUDGETS=256 DATASETS=longbench MAX_SAMPLES=2 \
EXECUTE=1 bash scripts/run.sh --longbench_tasks qasper
```

Without `EXECUTE=1`, `run.sh` only prints the plan. Main overrides are `GPUS`,
`METHODS`, `BUDGETS`, `DATASETS`, `CHUNK_SIZE`, `WINDOW_SIZE`, `KERNEL_SIZE`,
`POOLING`, `RUN_NAME`, and `OUTPUT_ROOT`. Use a new `RUN_NAME` for each run.
`RULER_NUM_SAMPLES` checks shared input counts; `MAX_SAMPLES` limits inference.

Run `python scripts/run.py --help` for all parameters.
See [inference](../docs/inference.md) for scoring commands.
