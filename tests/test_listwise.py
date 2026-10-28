from thedu.config import ListwiseConfig, LLMConfig, RunConfig
from thedu.llm import Gateway
from thedu.pricing import Rate
from thedu.ranker.listwise import ListwiseReranker, parse_order

DOCS = {f"d{i}": " ".join(f"w{i}x{j}" for j in range(150)) for i in range(6)}


class FakeClient:
    """Stands in for the provider: records the request and answers with fixed text."""

    def __init__(self, text):
        self.text, self.calls = text, []
        self.chat = self
        self.completions = self

    def create(self, **kw):
        self.calls.append(kw)
        msg = type("M", (), {"content": self.text})()
        usage = type("U", (), {"prompt_tokens": 100, "completion_tokens": 10, "completion_tokens_details": None})()
        return type("R", (), {"choices": [type("C", (), {"message": msg})()], "usage": usage, "model_extra": {"provider": "DeepInfra"}})()


def gateway(tmp_path, text, **cfg):
    client = FakeClient(text)
    return Gateway(LLMConfig(**cfg), Rate(input_per_mtok_usd=0.45, output_per_mtok_usd=3.0), cache_path=tmp_path / "c.jsonl", client=client), client


def test_parse_order_counts_skips_repeats_and_invalid():
    order, st = parse_order("3 > 1 > 3 > 9 > 2", 4)
    assert order == [2, 0, 1, 3]  # the missing candidate (4) is appended
    assert (st.missing, st.repeated, st.invalid, st.failed) == (1, 1, 1, False) and st.fallback
    order, st = parse_order("[2] > [1] > [3]", 3)
    assert order == [1, 0, 2] and not st.fallback
    order, st = parse_order("I cannot do that", 3)
    assert order == [0, 1, 2] and st.failed and st.missing == 3


def test_listwise_only_reorders_and_cuts_passages(tmp_path):
    gw, client = gateway(tmp_path, "3 > 1 > 2")
    rr = ListwiseReranker(DOCS, gw, depth=3, words=10)
    hits = [("d0", 5.0), ("d1", 4.0), ("d2", 3.0), ("d3", 2.0), ("d4", 0.0)]
    out, usage, st = rr.rerank(hits, "what?")
    assert [d for d, _ in out] == ["d2", "d0", "d1", "d3"]  # reordered top 3, rank 4 below, filler gone
    assert not st.fallback and usage.calls == 1
    prompt = client.calls[-1]["messages"][0]["content"]
    assert "[3] w2x0" in prompt and "w2x10" not in prompt and prompt.count("...") == 3  # cut to 10 words


def test_reasoning_option_is_sent_and_part_of_the_cache_key(tmp_path):
    gw, client = gateway(tmp_path, "1", reasoning=False)
    gw.call("p")
    assert client.calls[-1]["extra_body"]["reasoning"] == {"enabled": False}
    gw2, _ = gateway(tmp_path, "1")  # same cache file, reasoning not set: a different key
    assert gw2.call("p").cached is False


def test_hash_unchanged_unless_listwise_or_reasoning_set():
    base = RunConfig(dataset="fiqa", setups=["A0", "A3"])
    rates = {"input": 0.12, "output": 0.24}
    assert RunConfig(dataset="fiqa", setups=["A0", "A3"], llm=LLMConfig(reasoning=None)).hash(rates) == base.hash(rates)
    lw = RunConfig(dataset="fiqa", setups=["A0", "A0L"], listwise=ListwiseConfig(llm=LLMConfig(model="qwen/qwen3.5-397b-a17b", reasoning=False)))
    assert lw.hash(rates) != RunConfig(dataset="fiqa", setups=["A0", "A0R"]).hash(rates)


def test_pipeline_runs_the_listwise_setup_end_to_end(tmp_path, monkeypatch):
    from thedu import pipeline
    from thedu.config import RunConfig

    docs = {f"d{i}": f"alpha beta topic{i} " + "filler " * 5 for i in range(8)}
    docs["d3"] = "alpha beta gamma the answer is here"
    queries = {"q1": "alpha beta gamma", "q2": "alpha beta topic5"}
    qrels = {"q1": {"d3": 1}, "q2": {"d5": 1}}
    fake = FakeClient("2 > 1")  # puts the second candidate first, skips the rest

    def fake_gateway(cfg, offline=False, llm=None):
        gw = Gateway(llm or cfg.llm, Rate(input_per_mtok_usd=0.45, output_per_mtok_usd=3.0), cache_path=tmp_path / "c.jsonl", client=fake)
        return gw, {"input": 0.12, "output": 0.24}

    monkeypatch.setattr(pipeline, "make_gateway", fake_gateway)
    cfg = RunConfig(dataset="scifact", setups=["A0", "A0L"], listwise=ListwiseConfig(llm=LLMConfig(model="qwen/qwen3.5-397b-a17b", reasoning=False)),
                    rerank={"depth": 5})
    run, m = pipeline.execute(cfg, docs, queries, qrels, root=tmp_path / "runs")
    f = m["setups"]["A0L"]["listwise_fallbacks"]
    assert f["questions"] == 2 and f["questions_with_fallback"] == 2 and f["ids_missing"] >= 2  # "2 > 1" skips the rest
    assert m["setups"]["A0L"]["llm_calls"] == 2 and m["setups"]["A0"]["llm_calls"] == 0
    assert "A0L vs A0" in m["paired"] and m["paired"]["A0L vs A0"]["role"] == "secondary"
