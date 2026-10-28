"""Ranker part: rank(lists, question) -> list[(doc_id, score)].

M0 has one query per plan, so there is one list and nothing to merge. Merging and
reranking arrive with the hybrid work.
"""

from __future__ import annotations

Hit = tuple[str, float]


def rank(lists: list[list[Hit]], question: str) -> list[Hit]:
    if len(lists) != 1:
        raise NotImplementedError("merging several result lists is not built yet")
    return lists[0]
