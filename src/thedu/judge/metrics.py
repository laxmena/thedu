"""The judge: pure scoring functions. Frozen once M0 is done; do not edit.

nDCG@10 uses the rank order of the run (gain = relevance grade, log2 discount).
Recall@100 is the share of relevant documents found in the top 100.
Ranges are 95% percentile bootstrap ranges over questions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

Hit = tuple[str, float]
Run = dict[str, list[Hit]]
Qrels = dict[str, dict[str, int]]

N_BOOT = 2000
BOOT_SEED = 0


def ndcg_at_k(ranked: list[str], rels: dict[str, int], k: int = 10) -> float:
    dcg = sum(rels.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted((g for g in rels.values() if g > 0), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def recall_at_k(ranked: list[str], rels: dict[str, int], k: int = 100) -> float:
    relevant = {d for d, g in rels.items() if g > 0}
    if not relevant:
        return 0.0
    return len(relevant & set(ranked[:k])) / len(relevant)


def question_scores(hits: list[Hit], rels: dict[str, int]) -> dict[str, float]:
    ranked = [d for d, _ in hits]
    return {"ndcg10": ndcg_at_k(ranked, rels, 10), "recall100": recall_at_k(ranked, rels, 100)}


@dataclass(frozen=True)
class Interval:
    mean: float
    lo: float
    hi: float


@dataclass(frozen=True)
class Paired:
    """Setup B minus setup A on the same questions."""

    mean: float
    lo: float
    hi: float
    wins: int
    losses: int


@dataclass(frozen=True)
class Metrics:
    ndcg10: Interval
    recall100: Interval
    per_question: dict[str, dict[str, float]]


def bootstrap_interval(values: np.ndarray | list[float]) -> Interval:
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        raise ValueError("no questions to score")
    idx = np.random.default_rng(BOOT_SEED).integers(0, len(v), size=(N_BOOT, len(v)))
    means = v[idx].mean(axis=1)
    return Interval(float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def paired_difference(a: np.ndarray | list[float], b: np.ndarray | list[float]) -> Paired:
    """Paired bootstrap of b - a, question by question. Inputs must be in the same question order."""
    a_, b_ = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a_.shape != b_.shape:
        raise ValueError("paired comparison needs the same questions in the same order")
    d = b_ - a_
    iv = bootstrap_interval(d)
    return Paired(iv.mean, iv.lo, iv.hi, int((d > 0).sum()), int((d < 0).sum()))


def score(run: Run, qrels: Qrels) -> Metrics:
    """Score every question in `run`. A question without qrels is an error, not a zero."""
    per_q = {qid: question_scores(hits, qrels[qid]) for qid, hits in run.items()}
    qids = list(per_q)
    return Metrics(
        ndcg10=bootstrap_interval([per_q[q]["ndcg10"] for q in qids]),
        recall100=bootstrap_interval([per_q[q]["recall100"] for q in qids]),
        per_question=per_q,
    )
