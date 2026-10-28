"""Keyword-index variants for the sweep (JOURNAL.md, 2026-10-28). Not used by the pipeline's A0.

One change at a time against the current index (Snowball stemmer, Lucene stopwords, k1 0.9, b 0.4). Same tie rule as
every retriever. The index is plain BM25 with no model at index time.
"""

from __future__ import annotations

from dataclasses import dataclass

import bm25s
import numpy as np
import scipy.sparse as sp
import Stemmer

from thedu.retrievers.order import tie_ranks, top_k


@dataclass(frozen=True)
class Variant:
    stemmer: str = "snowball"  # snowball (current) | porter | none
    k1: float = 0.9
    b: float = 0.4
    bigrams: bool = False  # word bigrams added to the index and the query
    chars: str | None = None  # None | "only" | "plus": character 3 to 5-grams inside words, alone or with the words
    title_weight: float | None = None  # None: title and body are one field; w: body score + w x title score

    def label(self) -> str:
        parts = []
        if self.stemmer != "snowball":
            parts.append(f"stemmer={self.stemmer}")
        if (self.k1, self.b) != (0.9, 0.4):
            parts.append(f"k1={self.k1},b={self.b}")
        if self.bigrams:
            parts.append("bigrams")
        if self.chars:
            parts.append(f"chars-{self.chars}")
        if self.title_weight is not None:
            parts.append(f"title-w{self.title_weight}")
        return "+".join(parts) or "current"


def _stemmer(name: str):
    return {"snowball": Stemmer.Stemmer("english"), "porter": Stemmer.Stemmer("porter"), "none": None}[name]


def words(texts: list[str], stemmer: str) -> list[list[str]]:
    return bm25s.tokenize(texts, stopwords="en", stemmer=_stemmer(stemmer), return_ids=False, show_progress=False)


def char_grams(texts: list[str]) -> list[list[str]]:
    """Character 3 to 5-grams inside each word (stopwords removed, not stemmed, word edges marked)."""
    out = []
    for toks in words(texts, "none"):
        grams = []
        for w in toks:
            padded = f"#{w}#"
            grams += [f"c:{padded[i:i + n]}" for n in (3, 4, 5) for i in range(len(padded) - n + 1)]
        out.append(grams)
    return out


def analyze(texts: list[str], v: Variant) -> list[list[str]]:
    if v.chars == "only":
        return char_grams(texts)
    toks = words(texts, v.stemmer)
    if v.bigrams:
        toks = [t + [f"{a}_{b}" for a, b in zip(t, t[1:])] for t in toks]
    if v.chars == "plus":
        toks = [t + c for t, c in zip(toks, char_grams(texts))]
    return toks


def _index(tokens: list[list[str]], k1: float, b: float) -> bm25s.BM25:
    idx = bm25s.BM25(k1=k1, b=b)
    idx.index(tokens, show_progress=False)
    return idx


class SparseBm25:
    """Lucene-style BM25 over sparse term counts, built chunk by chunk. Character n-gram indexes have tens of millions
    of postings; this keeps them as arrays instead of Python lists. Used only by the character variants."""

    def __init__(self, token_chunks, n_docs: int, k1: float, b: float):
        self.vocab: dict[str, int] = {}
        rows, cols, vals = [], [], []
        dl = np.zeros(n_docs)
        d = 0
        for chunk in token_chunks:
            for toks in chunk:
                counts: dict[int, int] = {}
                for t in toks:
                    i = self.vocab.setdefault(t, len(self.vocab))
                    counts[i] = counts.get(i, 0) + 1
                rows.append(np.full(len(counts), d, dtype=np.int32))
                cols.append(np.fromiter(counts.keys(), dtype=np.int32, count=len(counts)))
                vals.append(np.fromiter(counts.values(), dtype=np.float32, count=len(counts)))
                dl[d] = len(toks)
                d += 1
        tf = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                           shape=(n_docs, len(self.vocab)))
        df = np.diff(tf.tocsc().indptr)
        idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        norm = (k1 * (1 - b + b * dl / max(dl.mean(), 1e-9))).astype(np.float32)
        row_of = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
        tf.data = idf[tf.indices] * tf.data / (tf.data + norm[row_of])
        self._w = tf.tocsc()
        self.n_docs = n_docs

    def get_scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(self.n_docs, dtype=np.float32)
        for t in query:
            if (i := self.vocab.get(t)) is not None:
                lo, hi = self._w.indptr[i], self._w.indptr[i + 1]
                out[self._w.indices[lo:hi]] += self._w.data[lo:hi]
        return out


class KeywordIndex:
    def __init__(self, doc_ids: list[str], texts: list[str], v: Variant, titles: list[str] | None = None):
        """`texts` is the whole document (title and body) unless the variant scores the title separately, in which
        case `texts` is the body and `titles` the titles."""
        self.doc_ids, self.v = doc_ids, v
        self._ranks = tie_ranks(doc_ids)
        if v.chars:
            chunks = (analyze(texts[i:i + 1000], v) for i in range(0, len(texts), 1000))
            self._body = SparseBm25(chunks, len(texts), v.k1, v.b)
        else:
            self._body = _index(analyze(texts, v), v.k1, v.b)
        self._title = None
        if v.title_weight is not None:
            assert titles is not None
            self._title = _index(analyze(titles, v), v.k1, v.b)

    def scores(self, text: str) -> np.ndarray:
        q = analyze([text], self.v)[0]
        s = self._body.get_scores(q) if q else np.zeros(len(self.doc_ids))
        if self._title is not None and q:
            s = s + self.v.title_weight * self._title.get_scores(q)
        return s

    def search(self, text: str, k: int) -> list[tuple[str, float]]:
        s = self.scores(text)
        return [(self.doc_ids[i], float(s[i])) for i in top_k(s, self._ranks, k)]
