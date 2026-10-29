"""The only door to test answers.

Only `thedu.parity` and `thedu.test_eval` may import this module. A test greps the
source to enforce that. Nothing else (dev runs, planner, ranker, notebooks) may import it.
"""

from __future__ import annotations

from thedu.data.loader import DATASETS, ensure_downloaded, read_qrels
from thedu.paths import QRELS_DIR


def load_test_answers(name: str) -> dict[str, dict[str, int]]:
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}")
    ensure_downloaded(name)
    return read_qrels(QRELS_DIR / name / "test.tsv")
