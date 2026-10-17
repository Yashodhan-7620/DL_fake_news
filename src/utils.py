"""Shared helpers: config loading, seeding, paths, JSON logging."""
from __future__ import annotations

import json
import os
import platform
import random
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
LABEL_NAMES = {0: "Fake", 1: "Real"}      # label 0 = fake, 1 = real (everywhere)
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(cfg: dict, key: str) -> Path:
    """Resolve a cfg['paths'][key] / cfg['data'][key] directory relative to project root."""
    section = cfg["paths"] if key in cfg["paths"] else cfg["data"]
    p = Path(section[key])
    return p if p.is_absolute() else ROOT / p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # torch is optional for the baseline-only path
        pass


def save_json(obj, path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: str | os.PathLike):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Not JSON serialisable: {type(o)}")


# def environment_info() -> dict:
#     """Library versions are recorded with every run (FR-10 reproducibility)."""
#     info = {"python": sys.version.split()[0], "platform": platform.platform()}
#     for mod in ("numpy", "pandas", "sklearn", "torch", "transformers", "streamlit"):
#         try:
#             info[mod] = __import__(mod).__version__
#         except Exception:
#             info[mod] = "not installed"
#     return info
def environment_info() -> dict:
    """Library versions are recorded with every run (FR-10 reproducibility)."""
    from importlib import metadata
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    for dist in ("numpy", "pandas", "scikit-learn", "torch", "transformers", "streamlit"):
        try:
            info[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            info[dist] = "not installed"
    return info