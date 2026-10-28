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


def test_ties_are_ordered_by_doc_id_descending_like_trec_eval():
    from thedu.retrievers.order import tie_ranks, top_k

    ids = ["a", "c", "b", "d"]
    scores = np.array([1.0, 1.0, 1.0, 0.5])
    assert [ids[i] for i in top_k(scores, tie_ranks(ids), 4)] == ["c", "b", "a", "d"]
    assert [ids[i] for i in top_k(scores, tie_ranks(ids), 2)] == ["c", "b"]


def test_bm25_ranking_does_not_depend_on_k_even_with_ties():
    docs, queries, _ = data.load("nfcorpus", "dev")  # many questions match nothing, so many ties
    r = Bm25Retriever(list(docs), list(docs.values()))
    tied = 0
    for q in list(queries.values())[:150] + ["zzzqqq", "the and of"]:
        full = r.search(q, 100)
        assert r.search(q, 10) == full[:10] and r.search(q, 37) == full[:37]
        tied += len({s for _, s in full[:11]}) < 11
    assert tied > 0  # the sample really contains ties
