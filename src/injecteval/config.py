"""YAML config loading + model-preset resolution."""

from __future__ import annotations

import os
from typing import Dict, Optional

import yaml

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_presets(path: Optional[str] = None) -> Dict:
    path = path or os.path.join(_HERE, "config", "models.yaml")
    if not os.path.exists(path):
        return {}
    return (load_yaml(path) or {}).get("presets", {})


def resolve_preset(name: str, presets: Optional[Dict] = None) -> Dict:
    presets = presets if presets is not None else load_presets()
    if name not in presets:
        raise ValueError(f"Unknown preset {name!r}. Known: {list(presets)}")
    return dict(presets[name])
