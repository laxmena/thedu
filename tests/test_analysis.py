import math

import numpy as np
import pytest

from thedu.analysis.features import CollectionStats, has_code, post_hoc_features, pre_search_features, terms
from thedu.analysis.labels import label
from thedu.analysis.stats import (apply_stump, bucket, cross_validated_gain, fit_stump, group_effect,
                                  tercile_thresholds)


def test_labels():
    assert label(0.5, 0.3) == "rewrite-wins" and label(0.4, 0.3) == "rewrite-wins"  # exactly 0.1 counts
    assert label(0.3, 0.45) == "embed-wins" and label(0.3, 0.4) == "embed-wins"
    assert label(0.5, 0.45) == "tie" and label(0.2, 0.2) == "tie"
    assert label(0.0, 0.0) == "both-fail"
    assert label(0.1, 0.0) == "rewrite-wins" and label(0.05, 0.0) == "tie"


DOCS = {"d1": "Glycotoxins damage kidney cells", "d2": "Kidney disease and diet", "d3": "Diet of the seaweed dulse",
        "d4": "Seaweed kidney diet"}


def test_idf_and_vocabulary_features_by_hand():
    st = CollectionStats(DOCS)
    sea = terms("seaweed")[0]  # whatever the stemmer makes of it
    assert st.n == 4 and st.df["kidney"] == 3 and st.df["diet"] == 3 and st.df[sea] == 2
    f = pre_search_features("Seaweed kidney zzzxx", st, [4.0, 1.0])
    idf_sea, idf_kid = math.log(1 + (4 - 2 + 0.5) / 2.5), math.log(1 + (4 - 3 + 0.5) / 3.5)
    assert f["n_terms"] == 3 and f["vocab_share"] == pytest.approx(2 / 3)
    assert f["mean_idf"] == pytest.approx((idf_sea + idf_kid) / 2) and f["max_idf"] == pytest.approx(idf_sea)
    assert f["mean_idf_norm"] == pytest.approx(f["mean_idf"] / st.max_idf)
    assert (f["top1_score"], f["gap"], f["gap_ratio"]) == (4.0, 3.0, 0.75)
    empty = pre_search_features("the of", st, [])
    assert empty["n_terms"] == 0 and np.isnan(empty["mean_idf"]) and empty["gap_ratio"] == 0.0


def test_has_code():
    assert has_code("p53 mutations") and has_code("IL-6 levels") and has_code("mRNA decay") and has_code("in 2010")
    assert not has_code("Glycotoxins and kidney") and not has_code("Dulse")


def test_post_hoc_overlap():
    f = post_hoc_features("seaweed kidney diet", ["Seaweed kidney diet study", "Kidney only"])
    assert f["overlap_best"] == 1.0 and f["overlap_union"] == 1.0
    f = post_hoc_features("seaweed kidney diet", ["kidney", "diet"])
    assert f["overlap_best"] == pytest.approx(1 / 3) and f["overlap_union"] == pytest.approx(2 / 3)
    assert np.isnan(post_hoc_features("seaweed", [])["overlap_best"])


def test_terms_are_stemmed_and_stopworded():
    assert terms("The kidneys of the diets") == ["kidney", "diet"]


def test_buckets():
    v = np.array([1.0, 2, 3, 4, 5, 6, np.nan])
    lo, hi = tercile_thresholds(v)
    assert [bucket(x, False, (lo, hi)) for x in v] == ["low", "low", "mid", "mid", "high", "high", "none"]
    assert bucket(1.0, True, (0, 0)) == "yes" and bucket(0.0, True, (0, 0)) == "no"


def test_group_effect_sign_and_range():
    rng = np.random.default_rng(0)
    mask = rng.random(400) < 0.3
    d = rng.normal(0, 0.1, 400) + np.where(mask, 0.2, 0.0)
    e = group_effect(d, mask)
    assert e["effect"] == pytest.approx(0.2, abs=0.05) and e["lo"] > 0 and e["share"] == pytest.approx(mask.mean())
    null = group_effect(rng.normal(0, 0.1, 400), mask)
    assert null["lo"] < 0 < null["hi"]


def test_stump_finds_the_planted_rule_and_cross_validation_is_honest():
    rng = np.random.default_rng(1)
    x = rng.random(500)
    gain = np.where(x < 0.3, 0.2, -0.1) + rng.normal(0, 0.05, 500)
    feats = {"x": x, "noise": rng.random(500)}
    stump = fit_stump(feats, gain, ["x", "noise"])
    assert stump["feature"] == "x" and stump["op"] == "<=" and 0.25 <= stump["threshold"] <= 0.35
    assert cross_validated_gain(feats, gain, ["x", "noise"]) > 0.03
    pure_noise = rng.normal(0, 0.1, 500)  # no real rule: out-of-sample gain should not be clearly positive
    assert cross_validated_gain(feats, pure_noise, ["x", "noise"]) < 0.01
    assert apply_stump({"feature": None}, feats, 500).sum() == 0


def test_confirm_refuses_rules_that_are_not_frozen():
    from thedu.analysis.steps import load_frozen_rules

    with pytest.raises(RuntimeError, match="frozen"):
        load_frozen_rules("configs/champion.yaml")  # a real file, but not in the frozen manifest
    assert load_frozen_rules("configs/m3b_rules.yaml")["rules"][0]["name"] == "R1"


def test_exploration_is_scifact_only():
    from thedu.analysis.steps import explore

    with pytest.raises(ValueError, match="SciFact only"):
        explore("anything", dataset="nfcorpus")


def test_router_config_is_frozen_and_says_what_was_pre_registered():
    from thedu.analysis.steps import load_frozen_rules

    r = load_frozen_rules("configs/router.yaml")
    assert (r["feature"], r["op"], r["threshold"], r["free"], r["paid"]) == ("mean_idf_norm", ">=", 0.448, "A4", "A5")
    assert r["criterion"]["paired_vs_free_lower_bound_above"] == 0.0


def test_diagnosis_placement_and_perfect_rerank():
    from thedu.analysis.diagnosis import diagnose, perfect_rerank, placement

    rels = {"r1": 1, "r2": 2}
    ranked = [f"x{i}" for i in range(12)] + ["r1"]
    assert placement(ranked, rels) == "ranks11-100"
    assert placement(["r2"] + ranked, rels) == "top10" and placement(["x"], rels) == "missing"
    assert perfect_rerank(["a", "r1", "r2"], rels) == ["r2", "r1", "a"]
    runs = {"A0": {"q": [("a", 1.0), ("r1", 0.5), ("r2", 0.0)]}, "A4": {"q": [("r2", 1.0), ("r1", 0.9)]}}
    out = diagnose(runs, {"q": rels}, ["q"])
    assert out["setups"]["A4"]["ndcg10"] == pytest.approx(1.0) and out["setups"]["A0"]["perfect_rerank_ndcg10"]["mean"] == pytest.approx(1.0)
    assert out["setups"]["A0"]["perfect_rerank_ndcg10_excluding_zero_score"] < 1.0  # r2 sat in zero-score filler
