# Third-Party Sources

Vendored sources retain their upstream notices. Local adapters provide shared
inputs, greedy generation, OOM handling, and benchmark scoring.

| Component | Location | Source identification |
| --- | --- | --- |
| KIVI | `baseline/kivi` | Revision `876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6`; MIT license retained in `LICENSE` |
| HeadInfer | `baseline/headinfer` | Revision `b57e449f3c0709739d812134649fdae7d1ca6a9f`; upstream package metadata declares MIT |
| KeyDiff engine | `baseline/keydiff/bpp` | Source commit `4f59c6d7b3a2ea0ab214d089e11dc1a64f73dd71` recorded in `manifest.yaml` |
| RULER | `dataset/ruler/RULER-main` | NVIDIA RULER generator and metric definitions; upstream license retained |
| AdaKV | `readkv/adakv` | Flattened adaptive-head cache integration |

The HeadInfer archive declares MIT in its package metadata but omits a standalone
license text. The imported KeyDiff source has no clear license grant. Confirm
redistribution permissions and complete the license notices before a public
release.

Model checkpoints and benchmark data have independent licenses and are excluded
from source control, including local dataset copies.
