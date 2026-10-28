"""LLM listwise reranking: one prompt holds the question and the candidates; the model returns their order.

It only reorders its candidates. Numbers the model skips are appended in their original order; repeats and numbers outside
the list are ignored. Each of these is counted (per question here, per dataset in the report).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from thedu.llm import Gateway, Usage
from thedu.prompts import load_prompt

Hit = tuple[str, float]


@dataclass
class ListwiseStats:
    missing: int = 0  # candidates the model left out (appended in their original order)
    repeated: int = 0  # numbers the model gave more than once
    invalid: int = 0  # numbers outside 1..n
    failed: bool = False  # no usable answer at all: the original order is kept

    @property
    def fallback(self) -> bool:
        return bool(self.missing or self.repeated or self.invalid or self.failed)


def parse_order(text: str, n: int) -> tuple[list[int], ListwiseStats]:
    """Zero-based candidate order from the model's answer."""
    stats, order, seen = ListwiseStats(), [], set()
    for tok in re.findall(r"\d+", text):
        i = int(tok)
        if not 1 <= i <= n:
            stats.invalid += 1
        elif i in seen:
            stats.repeated += 1
        else:
            seen.add(i)
            order.append(i - 1)
    stats.failed = not order
    stats.missing = n - len(order)
    order += [i for i in range(n) if i not in {o for o in order}]
    return order, stats


class ListwiseReranker:
    def __init__(self, docs: dict[str, str], gateway: Gateway, depth: int = 50, words: int = 100,
                 prompt: str = "listwise_v1"):
        self.docs, self.gateway, self.depth, self.words = docs, gateway, depth, words
        self.template = load_prompt(prompt)

    def _cut(self, text: str) -> str:
        w = text.split()
        return " ".join(w[: self.words]) + (" ..." if len(w) > self.words else "")

    def rerank(self, hits: list[Hit], question: str) -> tuple[list[Hit], Usage, ListwiseStats]:
        cands = [h for h in hits if h[1] > 0][: self.depth]  # score-zero filler is not a candidate
        if not cands:
            return hits, Usage(), ListwiseStats()
        passages = "\n".join(f"[{i + 1}] {self._cut(self.docs[d])}" for i, (d, _) in enumerate(cands))
        r = self.gateway.call(self.template.format(n=len(cands), q=question, passages=passages))
        usage = Usage()
        usage.add(r)
        order, stats = parse_order("" if r.failed else r.text, len(cands))
        head = [cands[i] for i in order]
        seen = {d for d, _ in head}
        return head + [h for h in hits if h[0] not in seen and h[1] > 0], usage, stats
