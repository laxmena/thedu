"""With cached rewrites, `thedu dev` reproduces its pinned dev numbers (A0 BM25 and A3 grounded rewrite).

Pins are at k=100 with ties ordered by doc id (trec_eval's rule). The spike's own figures
(SciFact A0 0.6945 / A3 0.6874; NFCorpus A0 0.285 / A3 0.325) were taken at k=10 with bm25s's
arbitrary tie order; SciFact is identical, NFCorpus differs in the 3rd decimal. See JOURNAL.md.
"""

import pytest

from thedu.config import load_config
from thedu.paths import ROOT
from thedu.pipeline import run_dev


def test_scifact_a0_and_a3_reproduce_on_all_809_dev_questions():
    cfg = load_config(ROOT / "tests/configs/spike_a0_a3.yaml")
    _, m = run_dev(cfg, offline=True)  # offline: a cache miss would fail the test, not call the model
    assert m["n_questions"] == 809
    assert m["setups"]["A0"]["ndcg10"]["mean"] == pytest.approx(0.6945, abs=5e-5)
    assert m["setups"]["A3"]["ndcg10"]["mean"] == pytest.approx(0.6874, abs=5e-5)
    assert m["setups"]["A3"]["fallbacks"] == 0


def test_nfcorpus_a0_and_a3_reproduce_on_all_324_dev_questions():
    cfg = load_config(ROOT / "tests/configs/spike_a0_a3_nfcorpus.yaml")
    _, m = run_dev(cfg, offline=True)
    assert m["n_questions"] == 324
    assert m["setups"]["A0"]["ndcg10"]["mean"] == pytest.approx(0.2842, abs=5e-5)
    assert m["setups"]["A3"]["ndcg10"]["mean"] == pytest.approx(0.3221, abs=5e-5)
    assert m["setups"]["A3"]["fallbacks"] == 0
