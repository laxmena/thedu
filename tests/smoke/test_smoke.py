import pytest

from thedu.config import load_config
from thedu.paths import ROOT
from thedu.pipeline import run_dev

pytestmark = pytest.mark.smoke


def test_spike_a4_reproduces_on_all_809_dev_questions():
    cfg = load_config(ROOT / "tests/configs/spike_a4.yaml")
    _, m = run_dev(cfg, offline=True)
    assert m["setups"]["A4"]["ndcg10"]["mean"] == pytest.approx(0.7376, abs=5e-5)


def test_twenty_questions_end_to_end():
    cfg = load_config(ROOT / "configs/champion.yaml").model_copy(update={"n": 20})
    _, m = run_dev(cfg, offline=True)
    assert m["n_questions"] == 20 and set(m["setups"]) == {"A0", "A4", "A3"}
    for s in m["setups"].values():
        assert 0 < s["ndcg10"]["mean"] <= 1 and 0 < s["recall100"]["mean"] <= 1
