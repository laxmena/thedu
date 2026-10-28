"""Per-question features. "Terms" are what BM25 sees: stopwords removed, English-stemmed.

Pre-search features can be computed before any retrieval answer is known (BM25 scores come from the
first search, which a real system runs anyway). Post-hoc features use the answer key and explain; they
must never feed a router.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import bm25s
import numpy as np
import Stemmer

_STEM = Stemmer.Stemmer("english")

PRE_SEARCH = ("n_terms", "mean_idf", "max_idf", "mean_idf_norm", "max_idf_norm", "vocab_share", "has_code",
              "top1_score", "gap", "gap_ratio")
POST_HOC = ("overlap_best", "overlap_union")
# Features whose scale depends on the collection (BM25 scores and IDF), so a rule's threshold cannot move
# between datasets. Rules use the scale-free versions: *_norm, gap_ratio, counts, shares and flags.
SCALE_FREE = ("n_terms", "mean_idf_norm", "max_idf_norm", "vocab_share", "has_code", "gap_ratio")


def terms(text: str) -> list[str]:
    return bm25s.tokenize([text], stopwords="en", stemmer=_STEM, return_ids=False, show_progress=False)[0]


class CollectionStats:
    """Document frequencies of the collection, with the Lucene BM25 idf."""

    def __init__(self, docs: dict[str, str]):
        tokens = bm25s.tokenize(list(docs.values()), stopwords="en", stemmer=_STEM, return_ids=False, show_progress=False)
        self.n = len(docs)
        self.df: Counter[str] = Counter()
        for t in tokens:
            self.df.update(set(t))
        self.max_idf = self.idf_from_df(1)

    def idf_from_df(self, df: int) -> float:
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def idf(self, term: str) -> float:
        return self.idf_from_df(self.df[term])


def has_code(question: str) -> bool:
    """Digits anywhere, or a token with two or more capital letters (acronyms, gene names, mRNA, IL-6)."""
    return bool(re.search(r"\d", question)) or any(sum(c.isupper() for c in tok) >= 2 for tok in re.findall(r"\w+", question))


def pre_search_features(question: str, stats: CollectionStats, top_scores: list[float]) -> dict[str, float]:
    t = terms(question)
    seen = [x for x in t if stats.df[x] > 0]  # IDF is over in-vocabulary terms; the share covers the rest
    idfs = [stats.idf(x) for x in seen]
    s1 = top_scores[0] if top_scores else 0.0
    s2 = top_scores[1] if len(top_scores) > 1 else 0.0
    nan = float("nan")
    return {
        "n_terms": float(len(t)),
        "mean_idf": float(np.mean(idfs)) if idfs else nan,
        "max_idf": max(idfs) if idfs else nan,
        "mean_idf_norm": float(np.mean(idfs)) / stats.max_idf if idfs else nan,
        "max_idf_norm": max(idfs) / stats.max_idf if idfs else nan,
        "vocab_share": len(seen) / len(t) if t else 0.0,
        "has_code": float(has_code(question)),
        "top1_score": s1,
        "gap": s1 - s2,
        "gap_ratio": (s1 - s2) / s1 if s1 > 0 else 0.0,
    }


def post_hoc_features(question: str, relevant_texts: list[str]) -> dict[str, float]:
    """Share of the question's terms found in the best-matching relevant document, and in any of them."""
    q = set(terms(question))
    if not q or not relevant_texts:
        return {"overlap_best": float("nan"), "overlap_union": float("nan")}
    per_doc = [q & set(terms(d)) for d in relevant_texts]
    return {"overlap_best": max(len(x) for x in per_doc) / len(q),
            "overlap_union": len(set().union(*per_doc)) / len(q)}
