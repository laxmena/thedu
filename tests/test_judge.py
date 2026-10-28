import json
from pathlib import Path

import numpy as np
import pytest

from thedu.judge import bootstrap_interval, paired_difference, question_scores, score

GOLDEN = Path(__file__).parent / "golden"


def read_run(path):
    run = {}
    for ln in path.read_text().splitlines():
        qid, _, doc, _rank, s, _tag = ln.split(" ")
        run.setdefault(qid, []).append((doc, float(s)))
    return run


def read_qrels(path):
    qrels = {}
    for ln in path.read_text().splitlines():
        qid, _, doc, g = ln.split("\t")
        qrels.setdefault(qid, {})[doc] = int(g)
    return qrels


@pytest.mark.parametrize("name", ["scifact", "nfcorpus"])  # nfcorpus has graded relevance (1 and 2)
def test_judge_matches_pytrec_eval_exactly(name):
    run, qrels = read_run(GOLDEN / f"{name}_dev50.run"), read_qrels(GOLDEN / f"{name}_dev50.qrels")
    stored = json.loads((GOLDEN / f"{name}_dev50.pytrec.json").read_text())
    m = score(run, qrels)
    assert set(m.per_question) == set(stored)
    for qid, ref in stored.items():
        assert m.per_question[qid]["ndcg10"] == pytest.approx(ref["ndcg10"], abs=1e-12)
        assert m.per_question[qid]["recall100"] == pytest.approx(ref["recall100"], abs=1e-12)


@pytest.mark.parametrize("name", ["scifact", "nfcorpus"])
def test_judge_matches_live_pytrec_eval(name):
    import pytrec_eval

    run, qrels = read_run(GOLDEN / f"{name}_dev50.run"), read_qrels(GOLDEN / f"{name}_dev50.qrels")
    # Strictly decreasing scores, so pytrec_eval has no ties to break.
    ev = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_10", "recall_100"}).evaluate(
        {q: {d: -float(i) for i, (d, _) in enumerate(h)} for q, h in run.items()})
    m = score(run, qrels)
    for q, v in ev.items():
        assert m.per_question[q]["ndcg10"] == pytest.approx(v["ndcg_cut_10"], abs=1e-12)
        assert m.per_question[q]["recall100"] == pytest.approx(v["recall_100"], abs=1e-12)


def test_known_values():
    rels = {"a": 1, "b": 2}
    s = question_scores([("x", 1.0), ("a", 0.9), ("b", 0.8)], rels)
    dcg = 1 / np.log2(3) + 2 / np.log2(4)
    idcg = 2 / np.log2(2) + 1 / np.log2(3)
    assert s["ndcg10"] == pytest.approx(dcg / idcg)
    assert s["recall100"] == 1.0
    assert question_scores([("x", 1.0)], rels) == {"ndcg10": 0.0, "recall100": 0.0}


def test_bootstrap_is_seeded_and_brackets_the_mean():
    v = np.random.default_rng(1).random(200)
    a, b = bootstrap_interval(v), bootstrap_interval(v)
    assert a == b and a.lo < a.mean < a.hi


def test_paired_difference_counts_and_sign():
    a = np.array([0.1, 0.5, 0.5, 0.9])
    b = np.array([0.2, 0.5, 0.4, 1.0])
    p = paired_difference(a, b)
    assert (p.wins, p.losses) == (2, 1) and p.mean == pytest.approx(0.025)
    with pytest.raises(ValueError):
        paired_difference(a, b[:3])


def test_question_without_qrels_is_an_error_not_a_zero():
    with pytest.raises(KeyError):
        score({"q1": [("d", 1.0)]}, {})
