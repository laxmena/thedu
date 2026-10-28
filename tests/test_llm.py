import json
from types import SimpleNamespace

import pytest

from thedu.config import LLMConfig
from thedu.llm import Gateway, GatewayError
from thedu.pricing import Rate

RATE = Rate(input_per_mtok_usd=1.0, output_per_mtok_usd=2.0)


class FakeClient:
    def __init__(self):
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.calls += 1
        self.last = kw
        msg = SimpleNamespace(content="<think>x</think> hello  world")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                               usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10))


def gw(tmp_path, **kw):
    return Gateway(LLMConfig(), RATE, tmp_path / "llm.jsonl", client=FakeClient(), **kw)


def test_call_records_tokens_cost_and_strips_thinking(tmp_path):
    g = gw(tmp_path)
    r = g.call("p")
    assert r.text == "hello  world" and (r.tokens_in, r.tokens_out) == (100, 10)
    assert r.cost_usd == pytest.approx((100 * 1.0 + 10 * 2.0) / 1e6)
    assert g.client.last["seed"] == 7 and g.client.last["temperature"] == 0.0  # from config


def test_response_is_on_disk_the_moment_it_arrives_and_reused(tmp_path):
    g = gw(tmp_path)
    g.call("p")
    lines = (tmp_path / "llm.jsonl").read_text().splitlines()  # no close(), no flush by us
    assert len(lines) == 1 and json.loads(lines[0])["text"] == "hello  world"
    g2 = gw(tmp_path)  # a new process after a crash
    r = g2.call("p")
    assert r.cached and g2.client.calls == 0


def test_a_torn_last_line_loses_only_that_entry(tmp_path):
    g = gw(tmp_path)
    g.call("a"), g.call("b")
    with open(tmp_path / "llm.jsonl", "a") as f:
        f.write('{"key": "tor')
    g2 = gw(tmp_path)
    assert g2.call("a").cached and g2.call("b").cached


def test_offline_cache_miss_is_an_error(tmp_path):
    with pytest.raises(GatewayError):
        gw(tmp_path, offline=True).call("never seen")


def test_budget_stops_new_calls(tmp_path):
    g = gw(tmp_path, budget_usd=0.0)
    with pytest.raises(GatewayError, match="budget"):
        g.call("p")


def test_spike_cache_is_read_by_legacy_key(tmp_path):
    legacy = tmp_path / "rewrites.json"
    legacy.write_text(json.dumps({"qwen/qwen3-14b::the question": "old terms"}))
    g = gw(tmp_path, legacy_path=legacy, offline=True)
    r = g.call("any prompt", legacy_key="the question")
    assert r.text == "old terms" and r.cached and r.cost_usd is None
