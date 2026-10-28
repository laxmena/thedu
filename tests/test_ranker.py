import itertools

import pytest

from thedu.pipeline import fuse
from thedu.ranker import RrfRanker, drop_zero_scores


def hits(*ids):
    return [(d, 1.0 / (i + 1)) for i, d in enumerate(ids)]  # the scores are ignored by RRF; only ranks count


def test_rrf_matches_a_hand_computed_example():
    # k=60. a: 1/61 + 1/63   b: 1/62 + 1/61   c: 1/63   d: 1/62   ->  b > a > d > c
    out = RrfRanker(k=60).rank([hits("a", "b", "c"), hits("b", "d", "a")], "q")
    assert [d for d, _ in out] == ["b", "a", "d", "c"]
    expected = {"a": 1 / 61 + 1 / 63, "b": 1 / 62 + 1 / 61, "c": 1 / 63, "d": 1 / 62}
    for d, s in out:
        assert s == pytest.approx(expected[d], rel=1e-12)


def test_rrf_uses_ranks_not_scores():
    big = [("a", 1000.0), ("b", 999.0)]
    tiny = [("b", 0.002), ("a", 0.001)]
    out = RrfRanker().rank([big, tiny], "q")
    assert out[0][1] == out[1][1]  # a and b are symmetric, whatever the scores were


def test_rrf_is_deterministic_and_ignores_list_order():
    lists = [hits("a", "b", "c", "d"), hits("d", "c", "x"), hits("b", "x", "a")]
    r = RrfRanker()
    first = r.rank(lists, "q")
    assert r.rank(lists, "q") == first
    for perm in itertools.permutations(lists):
        assert r.rank(list(perm), "q") == first


def test_rrf_ties_go_to_the_larger_doc_id():
    out = RrfRanker().rank([hits("x"), hits("y")], "q")  # both have score 1/61
    assert [d for d, _ in out] == ["y", "x"]


def test_rrf_contract_sorted_pairs_at_most_depth():
    lists = [hits(*[f"a{i}" for i in range(150)]), hits(*[f"b{i}" for i in range(150)])]
    out = RrfRanker(depth=100).rank(lists, "q")
    assert len(out) == 100
    assert [s for _, s in out] == sorted((s for _, s in out), reverse=True)
    assert all(isinstance(d, str) and isinstance(s, float) for d, s in out)
    assert not any(d in {"a100", "b100"} for d, _ in out)  # ranks beyond the depth are not fused


def test_a_single_list_passes_through_unchanged():
    one = [("a", 3.0), ("b", 2.0)]
    assert RrfRanker().rank([one], "q") is one


def test_zero_score_results_are_dropped():
    assert drop_zero_scores([("a", 2.0), ("z", 0.0), ("b", 0.5)]) == [("a", 2.0), ("b", 0.5)]


def test_zero_score_bm25_filler_gets_no_credit_in_fusion():
    bm25 = [("a", 2.0), ("b", 1.0), ("z1", 0.0), ("z2", 0.0)]  # z1, z2: matched nothing
    dense = [("d", 0.9), ("z1", 0.1)]  # an embedding list is never filtered
    out = dict(fuse(RrfRanker(), {"A0": bm25, "A4": dense}, "q"))
    assert "z2" not in out  # filler in the BM25 list: no credit, not even a place
    assert out["z1"] == pytest.approx(1 / 62)  # only its embedding rank counts, not BM25 rank 3
    assert out["a"] == pytest.approx(1 / 61) and out["b"] == pytest.approx(1 / 62)
