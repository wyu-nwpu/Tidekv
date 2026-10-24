"""Policy registry."""

from __future__ import annotations

from .policies.base import NoCompressionPolicy
from .policies.keydiff import KeyDiffPolicy
from .policies.sink_bpp import SinkBPPPolicy
from .policies.snap_bpp import SnapBPPPolicy

POLICY_REGISTRY = {
    "keydiff_bpp": KeyDiffPolicy,
    "snap_bpp": SnapBPPPolicy,
    "sink_bpp": SinkBPPPolicy,
    "none_bpp": NoCompressionPolicy,
}

def build_policy(spec):
    spec.validate()
    try:
        return POLICY_REGISTRY[spec.name](spec)
    except KeyError as exc:
        raise ValueError(f"unknown policy: {spec.name}") from exc

