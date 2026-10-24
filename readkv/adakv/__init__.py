"""AdaKV integration aligned with AdaKV-main (flatten KV cache + varlen decode)."""

from adakv.install import install_adakv_monkey_patch
from adakv.generate import adakv_generate

__all__ = ["install_adakv_monkey_patch", "adakv_generate"]
