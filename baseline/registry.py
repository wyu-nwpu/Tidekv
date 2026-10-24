"""Backend sources and Python environments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import sys
from typing import Any

import yaml

@dataclass(frozen=True)
class ExternalBackend:
    method: str
    source_kind: str
    source_revision: str
    source_path: str
    python: str
    adapter: str
    config_id: str

def _required_string(
    mapping: dict[str, Any], key: str, *, label: str | None = None
) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing {label or key}")
    return value

def _load_backend(record: Any) -> ExternalBackend:
    if not isinstance(record, dict):
        raise ValueError("backend record must be a mapping")

    source = record.get("source")
    if not isinstance(source, dict):
        raise ValueError("missing source")

    return ExternalBackend(
        method=_required_string(record, "method"),
        source_kind=_required_string(source, "kind", label="source.kind"),
        source_revision=_required_string(source, "revision", label="source.revision"),
        source_path=_required_string(source, "path", label="source.path"),
        python=_required_string(record, "python"),
        adapter=_required_string(record, "adapter"),
        config_id=_required_string(record, "config_id"),
    )

def load_registry(path: Path) -> dict[str, ExternalBackend]:
    """Load a complete registry and reject malformed or duplicate backends."""
    path = Path(path).resolve()
    try:
        data = yaml.safe_load(path.read_text())
    except OSError as error:
        raise ValueError(f"cannot read registry {path}") from error
    except yaml.YAMLError as error:
        raise ValueError(f"invalid YAML in registry {path}") from error

    if not isinstance(data, dict) or not isinstance(data.get("backends"), list):
        raise ValueError("registry must contain a backends list")

    registry: dict[str, ExternalBackend] = {}
    for record in data["backends"]:
        backend = _load_backend(record)
        if backend.method in registry:
            raise ValueError(f"duplicate method: {backend.method}")
        source_path = Path(backend.source_path)
        if source_path.is_absolute():
            candidates = (source_path,)
        else:
            candidates = (path.parent.parent / source_path, path.parent / source_path)
        if not any(candidate.is_dir() for candidate in candidates):
            raise ValueError(f"source.path does not exist: {backend.source_path}")
        registry[backend.method] = backend
    return registry

def get_backend(method: str) -> ExternalBackend:
    """Return a backend from this package's pinned manifest."""
    from dataclasses import replace

    backend = load_registry(Path(__file__).with_name("manifest.yaml"))[method]
    if backend.python.startswith("conda:"):
        name = backend.python.removeprefix("conda:")
        conda_exe = os.environ.get("CONDA_EXE")
        if conda_exe:
            base = Path(conda_exe).resolve().parents[1]
        else:
            executable = Path(sys.executable).resolve()
            base = executable.parents[3] if executable.parents[2].name == "envs" else executable.parents[1]
        interpreter = os.environ.get(f"READKV_{method.upper()}_PYTHON", str(base / "envs" / name / "bin/python"))
        backend = replace(backend, python=interpreter)
    return backend
