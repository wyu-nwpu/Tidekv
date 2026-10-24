# Datasets

LongBench JSONL files belong in `longbench/`, which is ignored by Git.
Original prompt and generation settings remain in `config/`.
Inference evaluates the 16 English tasks, not LongBench-E or Chinese tasks.

```bash
bash scripts/prepare_dataset.sh longbench
bash scripts/prepare_dataset.sh ruler-resources
RULER_LENGTHS="4096 8192 16384 32768 65536 131072" NUM_SAMPLES=100 \
bash scripts/prepare_dataset.sh ruler
```

Set `MODEL_PATH` first. RULER files are saved at
`<RULER_DATA>/<length>/<task>/validation.jsonl` and reused across methods.
Use a separate `RULER_DATA` directory for another model, seed, or sample count.
Mistral supports lengths up to 32768.

Prepared RULER prompts already include the chat template. The three isolated
backends also share tokenized inputs under `result/input_cache/`.
Use `python dataset/prepare_ruler.py --help` for task and generation options.
