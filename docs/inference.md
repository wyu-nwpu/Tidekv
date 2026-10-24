# Inference

```bash
GPUS="0" METHODS="snap_a tidekv_snap" BUDGETS=1024 \
DATASETS=longbench EXECUTE=1 bash scripts/run.sh
```

The runner schedules one active method/budget per GPU and scores automatically.
Set `DATASETS=ruler` and `RULER_LENGTHS` for RULER. Inputs must be prepared first.

Outputs use `result/<dataset>/<model>/<run>/<method>/budget_<B>/`.
Full, KIVI, and HeadInfer use `native/`; RULER adds a length directory.

Use a new `RUN_NAME` for each run. Existing predictions are not overwritten.
OOM samples are recorded and excluded from quality scores, with OOM counts and
rates reported separately.

To score existing predictions without loading a model:

```bash
python src/eval.py --save_dir /path/to/longbench/run
python src/eval_ruler.py --save_dir /path/to/ruler/length
python baseline/evaluate.py --run_dir /path/to/external/run --benchmark ruler
```

[Script examples](../scripts/README.md)
