"""Configuration objects for BPP baselines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

BaselineName = Literal["keydiff_bpp", "snap_bpp", "sink_bpp", "none_bpp"]

@dataclass(frozen=True)
class BaselineSpec:
    """Runtime knobs shared by all BPP policies."""

    name: BaselineName
    kv_budget: int = 4096
    block_size: int = 128
    sink_tokens: int = 4
    snap_observation_window: int = 32
    snap_recent_tokens: int = 32
    snap_kernel_size: int = 7
    snap_pooling: Literal["avg"] = "avg"
    keydiff_anchor: Literal["raw_mean", "normalized_mean"] = "raw_mean"
    score_dtype: Literal["fp32"] = "fp32"

    def validate(self) -> None:
        if self.name not in {"keydiff_bpp", "snap_bpp", "sink_bpp", "none_bpp"}:
            raise ValueError(f"unsupported baseline: {self.name}")
        if self.kv_budget <= 0 or self.block_size <= 0:
            raise ValueError("kv_budget and block_size must be positive")
        if not 0 <= self.sink_tokens < self.kv_budget:
            raise ValueError("sink_tokens must be in [0, kv_budget)")
        if self.name == "snap_bpp":
            if self.snap_recent_tokens >= self.kv_budget:
                raise ValueError("Snap recent window must be smaller than budget")
            if self.snap_kernel_size <= 0 or self.snap_kernel_size % 2 == 0:
                raise ValueError("Snap kernel size must be a positive odd number")
            if self.snap_pooling != "avg":
                raise ValueError("only avg pooling is implemented for Snap-BPP")
        if self.score_dtype != "fp32":
            raise ValueError("only fp32 scoring is implemented")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

