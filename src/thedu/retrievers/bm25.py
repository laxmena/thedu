"""BM25 with bm25s, set up like Anserini's flat baseline (k1=0.9, b=0.4)."""

from __future__ import annotations

import bm25s
import numpy as np
import Stemmer

from thedu.retrievers.order import tie_ranks, top_k


class Bm25Retriever:
    def __init__(self, doc_ids: list[str], texts: list[str], k1: float = 0.9, b: float = 0.4):
        self.doc_ids = doc_ids
        self._stemmer = Stemmer.Stemmer("english")
        tokens = bm25s.tokenize(texts, stopwords="en", stemmer=self._stemmer, show_progress=False)
        self._index = bm25s.BM25(k1=k1, b=b)
        self._index.index(tokens, show_progress=False)
        self._ranks = tie_ranks(doc_ids)

    def search(self, text: str, k: int) -> list[tuple[str, float]]:
        tokens = bm25s.tokenize([text], stopwords="en", stemmer=self._stemmer, return_ids=False, show_progress=False)[0]
        scores = self._index.get_scores(tokens) if tokens else np.zeros(len(self.doc_ids))
        return [(self.doc_ids[i], float(scores[i])) for i in top_k(scores, self._ranks, k)]
