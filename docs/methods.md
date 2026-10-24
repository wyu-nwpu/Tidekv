# Methods

TideKV processes every prompt chunk through one layer before moving to the next
layer. Attention retains the complete causal prefix within that layer. Afterward,
an observation-window selector compresses the layer's persistent KV cache.
Decoding uses the compressed caches and the original absolute positions.

The variants use SnapKV selection, pyramidal layer budgets, adaptive head
allocation, or ChunkKV chunk selection. Paired baselines share their parameter
defaults; equal parameters do not guarantee identical selected tokens or scores.
`CHUNK_SIZE` controls TideKV prefill; `CHUNK_LENGTH` controls ChunkKV selection.

Snap-A/Pyramid-A are compact GQA adaptations. Built-in baselines use their own
prefill paths, without a generic online chunking wrapper. KIVI, HeadInfer, and
KeyDiff keep isolated implementations.

The prefill and decode loops are in `readkv/tidekv.py`. Selection and compression
are in `readkv/simple_utils.py` and `readkv/tidekv_selectors.py`.
Paired defaults are defined in `src/baseline_config.py`.
