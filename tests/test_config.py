from thedu.config import RunConfig, load_config
from thedu.paths import ROOT

RATES = {"input": 0.12, "output": 0.24}


def cfg(**kw):
    return RunConfig(dataset="scifact", setups=["A0", "A3"], **kw)


def test_same_config_same_hash():
    assert cfg().hash(RATES) == cfg().hash(RATES)
    assert load_config(ROOT / "configs/champion.yaml").hash(RATES) == load_config(ROOT / "configs/champion.yaml").hash(RATES)


def test_hash_changes_with_settings_prompts_and_rates():
    base = cfg().hash(RATES)
    assert cfg(seed=8).hash(RATES) != base
    assert cfg(planner={"query_repeat": 2}).hash(RATES) != base
    assert cfg(llm={"temperature": 0.5}).hash(RATES) != base
    assert cfg().hash({"input": 0.2, "output": 0.24}) != base
    assert RunConfig(dataset="scifact", setups=["A0", "A2"]).hash(RATES) != RunConfig(
        dataset="scifact", setups=["A0", "A3"]).hash(RATES)


def test_parallelism_does_not_change_the_hash():
    assert cfg(llm={"workers": 4}).hash(RATES) == cfg(llm={"workers": 16}).hash(RATES)


def test_prompt_text_is_part_of_the_hash(monkeypatch):
    import thedu.config as c

    base = cfg().hash(RATES)
    monkeypatch.setattr(c, "load_prompt", lambda name: "a different prompt")
    assert cfg().hash(RATES) != base


def test_new_configs_use_the_task_instruction_and_spike_configs_stay_pinned():
    from thedu.config import TASK_INSTRUCTIONS, resolve_instruction

    assert cfg().dense.query_instruction == "task"
    assert resolve_instruction("scifact", "task") == TASK_INSTRUCTIONS["scifact"]
    assert resolve_instruction("scifact", "model-default") is None
    assert resolve_instruction("scifact", "custom words") == "custom words"
    assert load_config(ROOT / "tests/configs/spike_a4.yaml").dense.query_instruction == "model-default"
    assert load_config(ROOT / "configs/champion.yaml").dense.query_instruction == "model-default"
    import pytest

    assert resolve_instruction("fiqa", "task").startswith("Given a financial question")
    with pytest.raises(KeyError):
        resolve_instruction("brightx", "task")  # a dataset without a verified instruction stops instead of guessing
