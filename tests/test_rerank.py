import pytest

from thedu.config import RunConfig, load_config
from thedu.paths import ROOT
from thedu.ranker.rerank import CrossEncoderReranker

DOCS = {f"d{i}": f"document {i}" for i in range(8)}
SCORES = {"d0": 0.1, "d1": 0.9, "d2": 0.5, "d3": 0.5, "d4": 0.7, "d5": 0.3, "d6": 0.2, "d7": 0.8}


def make(depth=4):
    return CrossEncoderReranker(DOCS, depth=depth, max_length=20,
                                scorer=lambda pairs: [SCORES[p[1].split()[1].join(["d", ""])] if False else SCORES["d" + p[1].split()[1]] for p in pairs],
                                count_tokens=lambda t: len(t.split()))


def hits():
    return [(f"d{i}", 1.0 - i / 100) for i in range(8)]


def test_rerank_only_reorders_candidates():
    out, tokens = make(4).rerank(hits(), "what is it")
    assert [d for d, _ in out[:4]] == ["d1", "d2", "d3", "d0"]  # d2 and d3 tie: original order kept
    assert [d for d, _ in out[4:]] == ["d4", "d5", "d6", "d7"]  # below the candidates: untouched
    assert sorted(d for d, _ in out) == sorted(d for d, _ in hits())
    assert tokens == 4 * (3 + 2)  # question tokens x documents + document tokens


def test_rerank_drops_zero_score_filler_and_passes_through_empty():
    hs = [("d0", 2.0), ("d1", 1.0), ("d2", 0.0), ("d3", 0.0)]
    out, _ = make(10).rerank(hs, "q")
    assert [d for d, _ in out] == ["d1", "d0"]  # filler matched nothing; it is not a candidate and is not kept
    assert make().rerank([("d0", 0.0)], "q") == ([("d0", 0.0)], 0)


def test_rank_contract_takes_one_list():
    with pytest.raises(ValueError):
        make().rank([hits(), hits()], "q")
    assert len(make().rank([hits()], "q")) == 8


def test_old_run_hashes_are_unchanged_by_the_reranker_config():
    rates = {"input": 0.12, "output": 0.24}
    assert load_config(ROOT / "configs/m3_scifact.yaml").hash(rates) == "4a0ad1dd65"  # unchanged by adding the reranker config (the id 2a4eb7b8d0 predates llm.provider)
    assert RunConfig(dataset="fiqa", setups=["A0", "A3R"]).hash(rates) != RunConfig(dataset="fiqa", setups=["A0", "A3"]).hash(rates)
