"""Tiny YAML config loader with ``${ENV}`` expansion.

Kept dependency-light (PyYAML only) so the orchestration layer installs without
the GPU stack. It loads filter thresholds, trajectory parameters, and model/tool
locations from ``configs/``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_ENV_RE = re.compile(r"\$\{([A-Z0-9_]+)\}")

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), m.group(0)), value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def load_config(name: str) -> dict[str, Any]:
    """Load ``configs/<name>.yaml`` (or an absolute path) with env expansion."""
    path = Path(name)
    if not path.is_absolute() and path.suffix == "":
        path = CONFIG_DIR / f"{name}.yaml"
    elif not path.is_absolute():
        path = CONFIG_DIR / path
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return _expand(data)
