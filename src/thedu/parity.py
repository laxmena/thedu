"""`thedu parity`: frozen baselines vs published scores, on the test splits.

One of only two modules allowed to import thedu.data.test_answers. It runs A0 (BM25) and A4
(embeddings) with the default retriever settings, scores them on test, and compares nDCG@10 with
tests/expected.yaml. Passing means every |ours - published| <= tolerance. Each run is saved like any
other, with kind "parity" in meta.json and a parity.json beside metrics.json.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from thedu.config import Bm25Config, DenseConfig, RunConfig
from thedu.data import read_docs, read_queries
from thedu.data.test_answers import load_test_answers
from thedu.paths import ROOT

DATASETS = ("scifact", "nfcorpus")
EXPECTED = ROOT / "tests" / "expected.yaml"


@dataclass
class Row:
    dataset: str
    setup: str
    ours: float
    published: float
    run_hash: str

    @property
    def diff(self) -> float:
        return self.ours - self.published


def parity_config(dataset: str) -> RunConfig:
    return RunConfig(dataset=dataset, split="test", setups=["A0", "A4"], bm25=Bm25Config(), dense=DenseConfig())


def _load_expected() -> dict:
    """tests/expected.yaml, refused unless it is frozen and unchanged."""
    from thedu.cli import MANIFEST, read_manifest

    rel = str(EXPECTED.relative_to(ROOT))
    stored = read_manifest() if MANIFEST.exists() else {}
    digest = hashlib.sha256(EXPECTED.read_bytes()).hexdigest()
    if stored.get(rel) != digest:
        raise RuntimeError(f"{rel} must be frozen and unchanged before parity runs (`thedu freeze --add {rel}`).")
    return yaml.safe_load(EXPECTED.read_text())


def run_parity(restart: bool = False, root: Path | None = None) -> tuple[bool, list[Row]]:
    from thedu.pipeline import execute

    expected = _load_expected()
    tol = expected["tolerance_ndcg10"]
    rows: list[Row] = []
    for ds in DATASETS:
        answers = load_test_answers(ds)  # the one place test answers are read
        all_queries = read_queries(ds)
        queries = {q: all_queries[q] for q in sorted(answers) if q in all_queries}
        if len(queries) != len(answers):
            raise RuntimeError(f"{ds}: {len(answers) - len(queries)} test questions have no text")
        cfg = parity_config(ds)
        run, m = execute(cfg, read_docs(ds), queries, answers, restart=restart, root=root, kind="parity")
        mine = []
        for setup in ("A0", "A4"):
            row = Row(ds, setup, m["setups"][setup]["ndcg10"]["mean"], expected["scores"][ds][setup]["ndcg10"], run.path.name)
            rows.append(row)
            mine.append({"setup": setup, "ours": row.ours, "published": row.published, "diff": row.diff,
                         "passed": abs(row.diff) <= tol, "source_url": expected["scores"][ds][setup]["source_url"]})
        (run.path / "parity.json").write_text(json.dumps({"tolerance": tol, "n_questions": len(queries), "results": mine}, indent=2))
    return all(abs(r.diff) <= tol for r in rows), rows
