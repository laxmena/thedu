"""`thedu test-eval`: the final run on the test splits. Once, frozen, no tuning afterwards (JOURNAL.md, 2026-10-31).

One of only two modules allowed to import thedu.data.test_answers. It runs the frozen configs/test_<dataset>.yaml
(refused unless the file is in the freeze manifest and unchanged), scores on test, and saves the run like any other
with kind "test" and a testeval.json beside metrics.json (load average, question count, parity check for the
datasets that have a published score). A finished test run is never repeated.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import yaml

from thedu.config import RunConfig, load_config
from thedu.data import read_docs, read_queries
from thedu.data.test_answers import load_test_answers
from thedu.paths import ROOT, runs_dir

ORDER = ("nfcorpus", "scifact", "fiqa")


class TestEvalRefused(RuntimeError):
    __test__ = False


def frozen_config(dataset: str) -> RunConfig:
    from thedu.cli import MANIFEST, read_manifest

    path = ROOT / "configs" / f"test_{dataset}.yaml"
    rel = str(path.relative_to(ROOT))
    stored = read_manifest() if MANIFEST.exists() else {}
    if stored.get(rel) != hashlib.sha256(path.read_bytes()).hexdigest():
        raise TestEvalRefused(f"{rel} must be frozen and unchanged before the test run (`thedu freeze --add {rel}`).")
    cfg = load_config(path)
    if cfg.split != "test" or cfg.dataset != dataset:
        raise TestEvalRefused(f"{rel} is not the test config of {dataset}")
    return cfg


def _parity(dataset: str, metrics: dict) -> list[dict]:
    """A0 and A4 against the published scores of tests/expected.yaml (SciFact and NFCorpus only)."""
    expected = yaml.safe_load((ROOT / "tests" / "expected.yaml").read_text())
    pub = expected["scores"].get(dataset)
    if not pub:
        return []
    tol = expected["tolerance_ndcg10"]
    out = []
    for s in ("A0", "A4"):
        ours = metrics["setups"][s]["ndcg10"]["mean"]
        out.append({"setup": s, "ours": ours, "published": pub[s]["ndcg10"], "diff": ours - pub[s]["ndcg10"],
                    "passed": abs(ours - pub[s]["ndcg10"]) <= tol})
    return out


def run_test_eval(dataset: str, dry: bool = False, root: Path | None = None) -> dict:
    from thedu.pipeline import execute, make_gateway

    cfg = frozen_config(dataset)
    answers = load_test_answers(dataset)  # the one place the test answers are read
    all_queries = read_queries(dataset)
    queries = {q: all_queries[q] for q in sorted(answers) if q in all_queries}
    if len(queries) != len(answers):
        raise TestEvalRefused(f"{dataset}: {len(answers) - len(queries)} test questions have no text")
    _, rates = make_gateway(cfg, offline=True)
    run_hash = cfg.hash(rates)
    info = {"dataset": dataset, "run_hash": run_hash, "n_questions": len(queries), "rewrite_calls_at_most": len(queries)}
    done = (root or runs_dir()) / run_hash / "testeval.json"
    if done.exists():
        raise TestEvalRefused(f"the test run of {dataset} is finished (runs/{run_hash}); it is never repeated")
    if dry:
        return info
    t0, load0 = time.time(), os.getloadavg()
    run, metrics = execute(cfg, read_docs(dataset), queries, answers, root=root, kind="test")
    info |= {"load_start": load0, "load_end": os.getloadavg(), "minutes": (time.time() - t0) / 60, "parity": _parity(dataset, metrics)}
    (run.path / "testeval.json").write_text(json.dumps(info, indent=2))
    return info
