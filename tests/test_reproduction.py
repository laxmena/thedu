"""With cached rewrites, `thedu dev` on the spike's settings reproduces the spike's SciFact numbers."""

import pytest

from thedu.config import load_config
from thedu.paths import ROOT
from thedu.pipeline import run_dev


def test_spike_a0_and_a3_reproduce_on_all_809_dev_questions():
    cfg = load_config(ROOT / "tests/configs/spike_a0_a3.yaml")
    _, m = run_dev(cfg, offline=True)  # offline: a cache miss would fail the test, not call the model
    assert m["n_questions"] == 809
    assert m["setups"]["A0"]["ndcg10"]["mean"] == pytest.approx(0.6945, abs=5e-5)
    assert m["setups"]["A3"]["ndcg10"]["mean"] == pytest.approx(0.6874, abs=5e-5)
    assert m["setups"]["A3"]["fallbacks"] == 0
