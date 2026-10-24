# Runtime Requirements

Provide Linux, Python 3.10, CUDA, and the runtime dependencies. Environment setup
is managed outside this project. Reference versions are in `requirements.txt`
and `envs/`.

| Runtime | PyTorch | Transformers | FlashAttention |
| --- | --- | --- | --- |
| Main | 2.3.1+cu118 | 4.44.2 | 2.7.4.post1 |
| KIVI | 2.4.1, CUDA 12.4 | 4.43.1 | 2.6.3 |
| HeadInfer | 2.4.1, CUDA 12.4 | 4.45.2 | 2.6.3 |
| KeyDiff | 2.4.1, CUDA 12.4 | 4.44.2 | Not used |

KIVI also requires the `kivi_gemv` extension from `baseline/kivi/quant`.
Set `READKV_KIVI_PYTHON`, `READKV_HEADINFER_PYTHON`, or `READKV_KEYDIFF_PYTHON`
to the Python executable of an existing backend environment.
