"""Run configs. Same config (and same prompts and rates) gives the same hash."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from thedu.prompts import load_prompt

Setup = Literal["A0", "A2", "A3", "A4"]


# The instruction MTEB gives Qwen3-Embedding for each dataset's queries (sources in tests/expected.yaml).
# FiQA is absent on purpose: its instruction has not been checked against MTEB yet.
TASK_INSTRUCTIONS = {
    "scifact": "Given a scientific claim, retrieve documents that support or refute the claim",
    "nfcorpus": "Given a question, retrieve relevant documents that best answer the question",
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


class LLMConfig(_Frozen):
    model: str = "qwen/qwen3-14b"
    base_url: str = "https://openrouter.ai/api/v1"
    key_env: str = "OPENROUTER_API_KEY"
    temperature: float = 0.0
    seed: int = 7
    max_tokens: int = 300
    workers: int = 16  # parallel calls; does not change results, so not part of the hash


class PlannerConfig(_Frozen):
    blind_prompt: Literal["blind_v1", "blind_v2"] = "blind_v1"
    grounded_prompt: Literal["grounded_v1"] = "grounded_v1"
    query_repeat: int = Field(3, ge=1)  # the question is repeated so added words can't outweigh it
    feedback_k: int = Field(5, ge=1)  # results the grounded planner reads
    snippet_words: int = Field(60, ge=1)
    use_spike_cache: bool = False  # reuse the spike's rewrites (runs/spike/rewrites.json)
    max_fallback_rate: float = 0.05  # above this the run refuses to report


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
    pricing: str = "configs/pricing.yaml"

    def prompt_texts(self) -> dict[str, str]:
        names = []
        if "A2" in self.setups:
            names.append(self.planner.blind_prompt)
        if "A3" in self.setups:
            names.append(self.planner.grounded_prompt)
        return {n: load_prompt(n) for n in names}

    def hash(self, rates: dict[str, float] | None = None) -> str:
        payload = {
            "config": self.model_dump(exclude={"llm": {"workers"}, "pricing": True}),
            "prompts": self.prompt_texts(),
            "rates": rates or {},
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()[:10]


def load_config(path: str | Path) -> RunConfig:
    return RunConfig.model_validate(yaml.safe_load(Path(path).read_text()))
