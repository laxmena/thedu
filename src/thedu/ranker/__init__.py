"""Ranker part: rank(lists, question) -> list[(doc_id, score)].

Reciprocal Rank Fusion merges ranked lists using each document's rank in each list, not its score:
a document at rank r in a list gets 1 / (k + r), and the contributions add up. Equal weights, k from config.
"""

from __future__ import annotations

import math

Hit = tuple[str, float]


def drop_zero_scores(hits: list[Hit]) -> list[Hit]:
    """BM25 pads its list with score-zero documents that matched nothing. Rank-based fusion would give
    that arbitrary filler real credit, so lists from BM25 are cleaned before they are fused."""
    return [h for h in hits if h[1] > 0]


class RrfRanker:
    def __init__(self, k: int = 60, depth: int = 100):
        self.k, self.depth = k, depth

    def rank(self, lists: list[list[Hit]], question: str) -> list[Hit]:
        """Fuse the top `depth` of each list. A single list passes through unchanged.

        Ties are ordered by document id, largest first (the retrievers' rule), and the sum uses
        math.fsum, so the result does not depend on the order the lists arrive in.
        """
        if len(lists) == 1:
            return lists[0]
        parts: dict[str, list[float]] = {}
        for hits in lists:
            for rank, (doc_id, _) in enumerate(hits[: self.depth], start=1):
                parts.setdefault(doc_id, []).append(1.0 / (self.k + rank))
        by_id = sorted(parts, reverse=True)  # larger id first; the sort below is stable
        fused = sorted(((d, math.fsum(parts[d])) for d in by_id), key=lambda h: -h[1])
        return fused[: self.depth]
