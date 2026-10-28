"""Where things live. Env overrides exist so tests can work in a temp folder."""

from __future__ import annotations

import os
from pathlib import Path


def _find_root() -> Path:
    if env := os.environ.get("THEDU_ROOT"):
        return Path(env)
    for p in Path(__file__).resolve().parents:
        if (p / "pyproject.toml").exists():
            return p
    return Path.cwd()


ROOT = _find_root()
DATA_DIR = ROOT / "data"
QRELS_DIR = DATA_DIR / "qrels"
CONFIGS_DIR = ROOT / "configs"
SPIKE_DIR = ROOT / "runs" / "spike"


def runs_dir() -> Path:
    return Path(os.environ.get("THEDU_RUNS_DIR", ROOT / "runs"))
