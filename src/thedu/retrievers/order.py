"""One tie rule for every retriever, so a ranking never depends on k.

Documents are ordered by score, highest first. Equal scores are ordered by document id,
largest id first (string comparison), which is trec_eval's convention. The whole collection
is ranked, then cut to k, so the top 10 is the same whether 10 or 100 results are kept.
"""

from __future__ import annotations

import numpy as np


def tie_ranks(doc_ids: list[str]) -> np.ndarray:
    """Position of each document when ids are sorted largest first (0 = largest id)."""
    order = sorted(range(len(doc_ids)), key=lambda i: doc_ids[i], reverse=True)
    ranks = np.empty(len(doc_ids), dtype=np.int64)
    ranks[order] = np.arange(len(doc_ids))
    return ranks


def top_k(scores: np.ndarray, ranks: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k best documents: score descending, then id descending."""
    return np.lexsort((ranks, -scores))[:k]
