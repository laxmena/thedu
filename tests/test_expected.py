import yaml

from thedu.paths import ROOT


def test_expected_scores_are_complete_and_sourced():
    e = yaml.safe_load((ROOT / "tests/expected.yaml").read_text())
    assert e["tolerance_ndcg10"] == 0.010
    for ds in ("scifact", "nfcorpus"):
        for setup in ("A0", "A4"):
            s = e["scores"][ds][setup]
            assert 0 < s["ndcg10"] < 1
            assert s["source_url"].startswith("https://") and s["fetched"] and s["setup"]
