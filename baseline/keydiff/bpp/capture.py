"""Query projection capture for Snap-BPP."""

from __future__ import annotations

class QueryCapture:
    """Capture q_proj outputs for each Llama decoder layer."""

    def __init__(self, model) -> None:
        self.model = model
        self.outputs = [None] * len(model.model.layers)
        self.handles = []

    def __enter__(self):
        for layer_idx, layer in enumerate(self.model.model.layers):

            def hook(_module, _inputs, output, idx=layer_idx):
                self.outputs[idx] = output.detach()

            self.handles.append(layer.self_attn.q_proj.register_forward_hook(hook))
        return self

    def __exit__(self, *_exc) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

