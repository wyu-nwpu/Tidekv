"""Input and prediction records shared by external baseline adapters."""

from dataclasses import dataclass, field, fields
from typing import Any, Mapping


def _validate_token_ids(ids, label):
    if not isinstance(ids, (list, tuple)) or any(type(token) is not int or token < 0 for token in ids):
        raise ValueError(f"{label} must contain nonnegative integer token IDs")


@dataclass
class InputRecord:
    sample_id: str
    dataset: str
    input_ids: list[int]
    max_new_tokens: int
    eos_token_ids: list[int]
    answers: list[str]
    all_classes: list[str]
    length: int

    def __post_init__(self):
        if not self.sample_id or not self.dataset:
            raise ValueError("sample_id and dataset must be nonempty")
        _validate_token_ids(self.input_ids, "input_ids")
        _validate_token_ids(self.eos_token_ids, "eos_token_ids")
        if not self.input_ids or self.length != len(self.input_ids):
            raise ValueError("length must equal the nonempty input_ids length")
        if type(self.max_new_tokens) is not int or self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")

    def to_dict(self):
        return vars(self).copy()

    @classmethod
    def from_dict(cls, data):
        return cls(**{item.name: data[item.name] for item in fields(cls)})


@dataclass
class PredictionRecord:
    sample_id: str
    pred: str = ""
    answers: list[str] = field(default_factory=list)
    all_classes: list[str] = field(default_factory=list)
    length: int = 0
    prompt_tokens: int = 0
    generated_tokens: int = 0
    finish_reason: str = ""
    oom: bool = False
    oom_stage: str | None = None

    def __post_init__(self):
        if not self.sample_id or not isinstance(self.pred, str):
            raise ValueError("A prediction needs a sample_id and string pred")
        if not isinstance(self.oom, bool):
            raise ValueError("oom must be a boolean")

    def to_dict(self):
        return vars(self).copy()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]):
        for key in ("sample_id", "pred", "answers"):
            if key not in data:
                raise ValueError(f"missing prediction field: {key}")
        values = {item.name: data[item.name] for item in fields(cls) if item.name in data}
        # Archived rows may store OOM inside the former resource payload.
        values.setdefault("oom", bool((data.get("resources") or {}).get("oom", False)))
        return cls(**values)
