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


def depth_table(rankings: dict[str, Ranking], qrels: dict[str, dict[str, int]], qids: list[str], depths=(50, 100)) -> dict:
    """Would deeper candidates help a reranker? Relevant documents at ranks 51-100 and the perfect-rerank ceiling
    when only the top `depth` is reordered (ranks below keep their order). Score-zero filler is not a candidate."""
    out: dict = {}
    for s, run in rankings.items():
        hit = {q: [d for d, sc in run.get(q, []) if sc > 0][:100] for q in qids}
        deep = [sum(1 for d in hit[q][50:] if qrels[q].get(d, 0) > 0) for q in qids]
        shallow = [sum(1 for d in hit[q][:50] if qrels[q].get(d, 0) > 0) for q in qids]
        total = [len(qrels[q]) for q in qids]
        only_deep = [q for q, a, b in zip(qids, shallow, deep) if a == 0 and b > 0]
        row = {"relevant_at_51_100": int(sum(deep)), "per_question_mean": float(np.mean(deep)),
               "share_questions_with_any_at_51_100": float(np.mean([x > 0 for x in deep])),
               "share_questions_relevant_only_at_51_100": len(only_deep) / len(qids),
               "relevant_in_top50": int(sum(shallow)), "relevant_total": int(sum(total))}
        for k in depths:
            vals = [ndcg_at_k(sorted(hit[q][:k], key=lambda d: -qrels[q].get(d, 0)) + hit[q][k:], qrels[q]) for q in qids]
            row[f"ceiling_top{k}"] = float(np.mean(vals))
        out[s] = row
    return out


def recall_at(rankings: dict[str, Ranking], qrels: dict[str, dict[str, int]], qids: list[str], k: int = 50) -> dict:
    """Recall@k of each setup's list (score-zero filler cannot hold a relevant document, so it is left as is)."""
    from thedu.judge import recall_at_k

    out = {}
    for s, run in rankings.items():
        vals = np.array([recall_at_k([d for d, _ in run.get(q, [])], qrels[q], k) for q in qids])
        iv = bootstrap_interval(vals)
        out[s] = {"mean": iv.mean, "lo": iv.lo, "hi": iv.hi, "values": vals}
    for s in out:
        if s != "A4":
            p = paired_difference(out[s]["values"], out["A4"]["values"])
            out[s]["a4_minus_this"] = {"diff": p.mean, "lo": p.lo, "hi": p.hi}
    for v in out.values():
        del v["values"]
    return out


def ceiling_top(hits: list[tuple[str, float]], rels: dict[str, int], depth: int = 50) -> float:
    """nDCG@10 if the top `depth` candidates (score-zero filler dropped) were ordered by the answer key."""
    cands = [d for d, s in hits if s > 0]
    return ndcg_at_k(sorted(cands[:depth], key=lambda d: -rels.get(d, 0)) + cands[depth:], rels)
