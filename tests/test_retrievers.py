import numpy as np
import pytest

from thedu import data
from thedu.retrievers import Bm25Retriever, DenseRetriever


def check_contract(hits, k):
    assert len(hits) == k
    assert all(isinstance(d, str) and isinstance(s, float) for d, s in hits)
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)
    assert len({d for d, _ in hits}) == k


@pytest.fixture(scope="module")
def scifact():
    docs, queries, _ = data.load("scifact", "dev")
    return docs, queries


def test_bm25_returns_k_sorted_pairs(scifact):
    docs, queries = scifact
    r = Bm25Retriever(list(docs), list(docs.values()))
    for q in list(queries.values())[:5]:
        for k in (1, 10, 100):
            check_contract(r.search(q, k), k)


def test_bm25_top10_is_the_same_at_k10_and_k100(scifact):
    docs, queries = scifact
    r = Bm25Retriever(list(docs), list(docs.values()))
    for q in list(queries.values())[:100]:
        assert r.search(q, 10) == r.search(q, 100)[:10]


def test_dense_returns_k_sorted_pairs():
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(300, 16))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    r = DenseRetriever([f"d{i}" for i in range(300)], vecs, lambda qs: vecs[: len(qs)])
    for k in (1, 10, 100):
        check_contract(r.search("anything", k), k)
    assert r.search("anything", 3)[0][0] == "d0"  # the query vector is document 0's own
