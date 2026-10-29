"""BM25 with bm25s, set up like Anserini's flat baseline (k1=0.9, b=0.4)."""

from __future__ import annotations

import bm25s
import numpy as np
import Stemmer

from thedu.retrievers.order import tie_ranks, top_k

# Python's \w does not match Indic vowel signs and the virama (marks, not letters), so the default token pattern
# cuts Tamil words into pieces. A token here is a run of word characters, characters of the Indic blocks
# (Devanagari to Sinhala, which covers the Hindi and Sanskrit words that appear inside Tamil text) and joiners.
TAMIL_TOKEN = r"(?u)[\w\u0900-\u0DFF\u200C\u200D]+"

# language -> (PyStemmer algorithm, stopword list for bm25s, token pattern); Tamil has no stopword list
LANGUAGES = {
    "english": ("english", "en", r"(?u)\b\w\w+\b"),
    "tamil": ("tamil", [], TAMIL_TOKEN),
}


class Bm25Retriever:
    def __init__(self, doc_ids: list[str], texts: list[str], k1: float = 0.9, b: float = 0.4, language: str = "english"):
        self.doc_ids = doc_ids
        algo, self._stop, self._pattern = LANGUAGES[language]
        self._stemmer = Stemmer.Stemmer(algo)
        tokens = bm25s.tokenize(texts, stopwords=self._stop, stemmer=self._stemmer, token_pattern=self._pattern, show_progress=False)
        self._index = bm25s.BM25(k1=k1, b=b)
        self._index.index(tokens, show_progress=False)
        self._ranks = tie_ranks(doc_ids)

    def tokens(self, text: str) -> list[str]:
        """The index terms of a text (stemmed), as the search sees them."""
        return bm25s.tokenize([text], stopwords=self._stop, stemmer=self._stemmer, token_pattern=self._pattern,
                              return_ids=False, show_progress=False)[0]

    def search(self, text: str, k: int) -> list[tuple[str, float]]:
        tokens = self.tokens(text)
        scores = self._index.get_scores(tokens) if tokens else np.zeros(len(self.doc_ids))
        return [(self.doc_ids[i], float(scores[i])) for i in top_k(scores, self._ranks, k)]
