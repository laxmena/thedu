"""Where do the relevant documents sit in each setup's top 100, and what would a perfect reranker reach?

Reads finished runs and the dev answers; retrieves and calls nothing."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from thedu.judge import bootstrap_interval, ndcg_at_k, paired_difference

Ranking = dict[str, list[tuple[str, float]]]


def read_trec(path: Path) -> Ranking:
    out: Ranking = defaultdict(list)
    for line in path.read_text().splitlines():
        qid, _, doc, _, score, _ = line.split()
        out[qid].append((doc, float(score)))
    return dict(out)


def placement(ranked: list[str], rels: dict[str, int]) -> str:
    """Where the best-placed relevant document is: top10, ranks11-100, or missing."""
    for i, d in enumerate(ranked[:100]):
        if rels.get(d, 0) > 0:
            return "top10" if i < 10 else "ranks11-100"
    return "missing"


def perfect_rerank(ranked: list[str], rels: dict[str, int]) -> list[str]:
    """The same candidates ordered by the answer key (highest grade first); the order inside a grade is kept."""
    return sorted(ranked[:100], key=lambda d: -rels.get(d, 0))


def diagnose(rankings: dict[str, Ranking], qrels: dict[str, dict[str, int]], qids: list[str]) -> dict:
    out: dict = {"n_questions": len(qids), "setups": {}, "paired_ceiling_vs_a4": {}}
    actual: dict[str, np.ndarray] = {}
    ceiling: dict[str, np.ndarray] = {}
    for s, run in rankings.items():
        place = {"top10": 0, "ranks11-100": 0, "missing": 0}
        ceil_all, ceil_matched, act = [], [], []
        for q in qids:
            ranked = [d for d, _ in run.get(q, [])]
            matched = [d for d, sc in run.get(q, []) if sc > 0]  # BM25 pads its list with score-zero filler
            place[placement(ranked, qrels[q])] += 1
            act.append(ndcg_at_k(ranked, qrels[q]))
            ceil_all.append(ndcg_at_k(perfect_rerank(ranked, qrels[q]), qrels[q]))
            ceil_matched.append(ndcg_at_k(perfect_rerank(matched, qrels[q]), qrels[q]))
        actual[s], ceiling[s] = np.array(act), np.array(ceil_all)
        iv = bootstrap_interval(ceiling[s])
        out["setups"][s] = {
            "share": {k: v / len(qids) for k, v in place.items()}, "count": place,
            "ndcg10": float(np.mean(act)),
            "perfect_rerank_ndcg10": {"mean": iv.mean, "lo": iv.lo, "hi": iv.hi},
            # Candidates with a score of zero matched nothing; a ceiling that excludes them is the honest one for BM25.
            "perfect_rerank_ndcg10_excluding_zero_score": float(np.mean(ceil_matched)),
        }
    if "A4" in actual:
        for s in ceiling:
            if s != "A4":
                p = paired_difference(actual["A4"], ceiling[s])
                out["paired_ceiling_vs_a4"][s] = {"diff": p.mean, "lo": p.lo, "hi": p.hi}
    return out
