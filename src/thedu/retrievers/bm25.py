"""BM25 with bm25s, set up like Anserini's flat baseline (k1=0.9, b=0.4)."""

from __future__ import annotations

import bm25s
import Stemmer

STEMMER = Stemmer.Stemmer("english")


class Bm25Retriever:
    def __init__(self, doc_ids: list[str], texts: list[str], k1: float = 0.9, b: float = 0.4):
        self.doc_ids = doc_ids
        tokens = bm25s.tokenize(texts, stopwords="en", stemmer=STEMMER, show_progress=False)
        self._index = bm25s.BM25(k1=k1, b=b)
        self._index.index(tokens, show_progress=False)

    def search(self, text: str, k: int) -> list[tuple[str, float]]:
        tokens = bm25s.tokenize([text], stopwords="en", stemmer=STEMMER, show_progress=False)
        results, scores = self._index.retrieve(tokens, k=min(k, len(self.doc_ids)), show_progress=False)
        return [(self.doc_ids[i], float(s)) for i, s in zip(results[0], scores[0])]
