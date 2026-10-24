# Isolated Baselines

| Method | Sources | Default configuration |
| --- | --- | --- |
| KIVI | `kivi/models`, `kivi/quant` | 2-bit K/V, group 32, residual 32, FP16 |
| HeadInfer | `headinfer/headinfer` | Ordered exact cache offload, BF16 |
| KeyDiff | `keydiff/bpp` | Block 128, `raw_mean`, BF16, eager attention |

`adapters/` connects these sources to shared inputs, OOM handling, and
scoring. `manifest.yaml` records sources; `registry.py` resolves environments.

```bash
METHODS="kivi headinfer keydiff" BUDGETS=4096 DATASETS=longbench \
EXECUTE=1 bash scripts/run.sh
```

These backends target Llama-3.1-8B-Instruct on A100. KIVI and HeadInfer
do not use a token-retention budget. Their attention backends are independent
of the built-in `ATTN_IMPLEMENTATION` setting.

Provide the backend environments separately; dependency versions are in `envs/`.
Generation limits come from each tokenized input record. KIVI uses FP16;
HeadInfer and KeyDiff use BF16, with batch size fixed at one.

