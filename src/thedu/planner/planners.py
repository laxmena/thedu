"""A2 (blind) and A3 (grounded) rewriters behind the planner contract."""

from __future__ import annotations

from typing import Protocol

from thedu.config import PlannerConfig
from thedu.llm import Gateway, Usage
from thedu.planner.plan import Plan
from thedu.prompts import load_prompt

Hit = tuple[str, float]


class Planner(Protocol):
    def plan(self, question: str, hits: list[Hit] | None = None) -> Plan: ...


def _plan_from_terms(question: str, cfg: PlannerConfig, gateway: Gateway, prompt: str, legacy_key: str) -> Plan:
    """Ask for search terms; retry once if the answer is empty; else fall back to the raw question."""
    usage = Usage()
    for attempt in (0, 1):
        r = gateway.call(prompt, attempt=attempt, legacy_key=legacy_key if cfg.use_spike_cache else None)
        usage.add(r)
        if r.failed:
            break  # the gateway already retried the provider; do not ask again
        if terms := " ".join(r.text.split()):
            # The question repeated N times, then the terms: the added words can't outweigh the question.
            return Plan(queries=[" ".join([question] * cfg.query_repeat + [terms])], usage=usage)
    return Plan(queries=[question], usage=usage, fallback=True)


class BlindPlanner:
    """A2: the LLM sees only the question."""

    def __init__(self, gateway: Gateway, cfg: PlannerConfig):
        self.gateway, self.cfg = gateway, cfg
        self.template = load_prompt(cfg.blind_prompt)

    def plan(self, question: str, hits: list[Hit] | None = None) -> Plan:
        # The spike cached v1 under the bare question and v2 under "v2::<question>".
        legacy = question if self.cfg.blind_prompt == "blind_v1" else f"v2::{question}"
        return _plan_from_terms(question, self.cfg, self.gateway, self.template.format(q=question), legacy)


class GroundedPlanner:
    """A3: the LLM reads the top results of a first search and writes terms from the relevant ones."""

    def __init__(self, gateway: Gateway, cfg: PlannerConfig, docs: dict[str, str], dataset: str):
        self.gateway, self.cfg, self.docs, self.dataset = gateway, cfg, docs, dataset
        self.template = load_prompt(cfg.grounded_prompt)

    def _snippet(self, text: str) -> str:
        words = text.split()
        n = self.cfg.snippet_words
        return " ".join(words[:n]) + (" ..." if len(words) > n else "")

    def _prompt(self, question: str, hits: list[Hit]) -> tuple[str, str]:
        shown = [d for d, _ in hits[: self.cfg.feedback_k]]
        snippets = "\n".join(f"[{i + 1}] {self._snippet(self.docs[d])}" for i, d in enumerate(shown))
        # The spike's key for round 1 (kept so its cached rewrites are reused).
        legacy = f"a3::{self.dataset}::x{self.cfg.query_repeat}::r1::{'|'.join(shown)}::{question}"
        return self.template.format(snippets=snippets, q=question), legacy

    def prompt_for(self, question: str, hits: list[Hit]) -> str:
        return self._prompt(question, hits)[0]

    def plan(self, question: str, hits: list[Hit] | None = None) -> Plan:
        if hits is None:
            raise ValueError("the grounded planner needs the first search's results")
        prompt, legacy = self._prompt(question, hits)
        return _plan_from_terms(question, self.cfg, self.gateway, prompt, legacy)
