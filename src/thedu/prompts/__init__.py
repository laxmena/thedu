"""Prompt text lives in .txt files next to this module. Config files name them."""

from __future__ import annotations

from pathlib import Path

_DIR = Path(__file__).parent


def load_prompt(name: str) -> str:
    return (_DIR / f"{name}.txt").read_text().rstrip("\n")
