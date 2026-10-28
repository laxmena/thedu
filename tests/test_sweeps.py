import math

import pytest

from thedu.retrievers import Bm25Retriever
from thedu.retrievers.variants import KeywordIndex, Variant, SparseBm25
from thedu.sweeps import candidate_terms, index_variants, rm3_weights

DOCS = {"d1": "Glycotoxins damage kidney cells", "d2": "Kidney disease and diet", "d3": "Diet of the seaweed dulse",
        "d4": "Seaweed salad recipes with sesame", "d5": "Heart pulse and rate monitoring"}


def test_current_variant_is_the_pipelines_a0():
    ids, texts = list(DOCS), list(DOCS.values())
    a, b = Bm25Retriever(ids, texts), KeywordIndex(ids, texts, Variant())
    for q in ("kidney diet", "seaweed", "nothing matches xyz"):
        assert a.search(q, 5) == b.search(q, 5)


def test_variants_are_the_preregistered_list():
    labels = [v.label() for v in index_variants()]
    assert len(labels) == 23 and len(set(labels)) == 23 and "current" not in labels
    assert {"stemmer=porter", "stemmer=none", "bigrams", "chars-only", "chars-plus", "title-w2.0"} <= set(labels)


def test_every_variant_runs_and_finds_the_obvious_document():
    ids, texts = list(DOCS), list(DOCS.values())
    titles = ["note"] * len(texts)  # a neutral title
    for v in index_variants():
        idx = KeywordIndex(ids, texts, v, titles)
        assert idx.search("seaweed dulse", 1)[0][0] == "d3", v.label()


def test_sparse_bm25_matches_bm25s_scores():
    import bm25s
    import numpy as np
    import Stemmer

    texts = list(DOCS.values())
    tok = bm25s.tokenize(texts, stopwords="en", stemmer=Stemmer.Stemmer("english"), return_ids=False, show_progress=False)
    sparse = SparseBm25([tok], len(tok), 0.9, 0.4)
    ref = bm25s.BM25(k1=0.9, b=0.4)
    ref.index(tok, show_progress=False)
    q = ["kidney", "diet"]
    assert np.allclose(sparse.get_scores(q), ref.get_scores(q), atol=1e-5)


def test_candidate_terms_rules():
    top = ["the heart beats quickly with high pulse rates", "a high pulse and quick heart beats", "zzz lonely word"]
    assert candidate_terms("Why is the heart rate high", top) == ["beats", "pulse"]  # in two documents, not in the question


def test_rm3_weights_sum_to_one_and_split_by_orig_weight():
    docs = {"a": "kidney kidney disease treatment", "b": "kidney dialysis treatment"}
    w = rm3_weights("kidney disease", [("a", 2.0), ("b", 1.0), ("c", 0.0)], docs)
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["kidney"] > w["dialysi"] > 0
    assert rm3_weights("x", [("a", 0.0)], docs) is None


def test_m5_codenames_and_candidate_cleanup():
    from thedu.m5.entities import clean, codenames, merge

    names = codenames(50, {"banana"})
    assert len(set(names)) == 50 and all(6 <= len(n) <= 8 and n.isalpha() for n in names)
    assert names == codenames(50, {"banana"})  # same seed, same names
    assert clean("The UK") == "UK" and clean("In Canada") == "Canada"
    out = merge([{"phrase": "The UK", "dev_questions": 3, "docs": 10}, {"phrase": "UK", "dev_questions": 4, "docs": 7},
                 {"phrase": "Don't", "dev_questions": 9, "docs": 9}])
    assert [g["phrase"] for g in out] == ["UK"] and out[0]["dev_questions"] == 4
