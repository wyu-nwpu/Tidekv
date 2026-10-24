# Outputs

```text
longbench/<model>/<run>/<method>/budget_<B>/
ruler/<model>/<run>/<method>/budget_<B>/<length>/
input_cache/       Shared tokenized inputs
logs/<run>/       Inference logs
```

Full, KIVI, and HeadInfer use `native/`. Limited runs add `samples_<N>/`.
Only placeholders are distributed; generated files are ignored by Git.
Predictions are JSONL. Scores are saved as `result.json` for LongBench,
`summary.csv` for RULER, or `metrics.json` for isolated backends.
