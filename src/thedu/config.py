"""Run configs. Same config (and same prompts and rates) gives the same hash."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from thedu.prompts import load_prompt

Setup = Literal["A0", "A2", "A3", "A4", "H0", "A5", "A0R", "A3R", "A4R", "A0L", "A3L", "A4L"]
# H0 = RRF(A0, A4); A5 = RRF(A3, A4); xR = the top of x reordered by a cross-encoder; xL = by an LLM (listwise)


# The instruction MTEB gives Qwen3-Embedding for each dataset's queries. SciFact and NFCorpus: sources in
# tests/expected.yaml. FiQA: mteb/tasks/retrieval/eng/fi_qa2018_retrieval.py (mteb main, read 2026-10-27).
TASK_INSTRUCTIONS = {
    "scifact": "Given a scientific claim, retrieve documents that support or refute the claim",
    "nfcorpus": "Given a question, retrieve relevant documents that best answer the question",
    "fiqa": "Given a financial question, retrieve user replies that best answer the question",
}


def resolve_instruction(dataset: str, setting: str) -> str | None:
    """The instruction text to use, or None for the model's own default query prompt."""
    if setting == "model-default":
        return None
    if setting == "task":
        if dataset not in TASK_INSTRUCTIONS:
            raise KeyError(f"no verified MTEB task instruction for {dataset!r}; add it to TASK_INSTRUCTIONS or set one in the config")
        return TASK_INSTRUCTIONS[dataset]
    return setting


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Bm25Config(_Frozen):
    k1: float = 0.9
    b: float = 0.4


class DenseConfig(_Frozen):
    model: str = "Qwen/Qwen3-Embedding-0.6B"
    max_seq_length: int = 512
    # Text put in front of each query as "Instruct: <text>\nQuery:".
    #   "task"          the dataset's MTEB task instruction (TASK_INSTRUCTIONS); the default for new configs
    #   "model-default" the model's own generic query prompt (what the spike used)
    #   anything else   used as the instruction text
    query_instruction: str = "task"


class ProviderConfig(_Frozen):
    """OpenRouter routing: which provider may serve the call. With allow_fallbacks False, a provider that is down
    makes the call fail loudly instead of being served by a different one."""

    order: list[str]
    allow_fallbacks: bool = False


class LLMConfig(_Frozen):
    model: str = "qwen/qwen3-14b"
    base_url: str = "https://openrouter.ai/api/v1"
    key_env: str = "OPENROUTER_API_KEY"
    temperature: float = 0.0
    seed: int = 7
    max_tokens: int = 300
    workers: int = 16  # parallel calls; does not change results, so not part of the hash
    cache_file: str = "runs/cache/llm.jsonl"  # where responses are kept; not part of the hash
    provider: ProviderConfig | None = None  # None: whoever OpenRouter picks (what the spike cache used)
    reasoning: bool | None = None  # None: not sent; False: ask OpenRouter to switch the model's thinking off


class PlannerConfig(_Frozen):
    blind_prompt: Literal["blind_v1", "blind_v2"] = "blind_v1"
    grounded_prompt: Literal["grounded_v1"] = "grounded_v1"
    query_repeat: int = Field(3, ge=1)  # the question is repeated so added words can't outweigh it
    feedback_k: int = Field(5, ge=1)  # results the grounded planner reads
    snippet_words: int = Field(60, ge=1)
    use_spike_cache: bool = False  # reuse the spike's rewrites (runs/spike/rewrites.json)
    max_fallback_rate: float = 0.05  # above this the run refuses to report


class RankerConfig(_Frozen):
    rrf_k: int = Field(60, ge=1)  # the standard value; tuned later, not in M3
    depth: int = Field(100, ge=1)  # top results of each list that are fused, and kept


class RerankConfig(_Frozen):
    model: str = "BAAI/bge-reranker-v2-m3"  # zero-shot, not tuned
    depth: int = Field(50, ge=1)  # candidates reordered; ranks below stay in their original order
    max_length: int = Field(512, ge=1)  # query + document tokens per pair


class ListwiseConfig(_Frozen):
    llm: LLMConfig  # the judge model (not the rewriter)
    words: int = Field(100, ge=1)  # each candidate is cut to this many words
    prompt: Literal["listwise_v1"] = "listwise_v1"


class RunConfig(_Frozen):
    dataset: Literal["scifact", "nfcorpus", "fiqa"]
    split: Literal["dev", "test"] = "dev"  # `thedu dev` refuses "test"; only `thedu parity` runs it
    n: int | None = None  # None = every dev question
    seed: int = 7  # question sampling
    k: int = Field(100, ge=10)  # results kept per question (Recall@100 needs 100)
    setups: list[Setup]
    bm25: Bm25Config = Bm25Config()
    dense: DenseConfig = DenseConfig()
    llm: LLMConfig = LLMConfig()
    planner: PlannerConfig = PlannerConfig()
    ranker: RankerConfig = RankerConfig()
    rerank: RerankConfig = RerankConfig()  # candidate depth (and the cross-encoder for xR); left out of the hash if unused
    listwise: ListwiseConfig | None = None  # required by the xL setups
    pricing: str = "configs/pricing.yaml"

    def prompt_texts(self) -> dict[str, str]:
        names = []
        if "A2" in self.setups:
            names.append(self.planner.blind_prompt)
        if {"A3", "A3R"} & set(self.setups):
            names.append(self.planner.grounded_prompt)
        return {n: load_prompt(n) for n in names}

    def uses_rerank(self) -> bool:
        return any(x.endswith(("R", "L")) for x in self.setups)

    def uses_listwise(self) -> bool:
        return any(x.endswith("L") for x in self.setups)

    def hash(self, rates: dict[str, float] | None = None) -> str:
        llm_ex = {"workers", "cache_file"} | ({"reasoning"} if self.llm.reasoning is None else set())
        exclude: dict = {"llm": llm_ex, "pricing": True}
        if not self.uses_rerank():
            exclude["rerank"] = True  # keeps the hashes of runs made before reranking existed
        if self.listwise is None:
            exclude["listwise"] = True
        else:
            exclude["listwise"] = {"llm": {"workers", "cache_file"} | ({"reasoning"} if self.listwise.llm.reasoning is None else set())}
        payload = {
            "config": self.model_dump(exclude=exclude),
            "prompts": {**self.prompt_texts(), **({self.listwise.prompt: load_prompt(self.listwise.prompt)} if self.listwise else {})},
            "rates": rates or {},
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()[:10]


def load_config(path: str | Path) -> RunConfig:
    return RunConfig.model_validate(yaml.safe_load(Path(path).read_text()))
