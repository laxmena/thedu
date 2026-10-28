"""Cost and latency. Frozen once M0 is done; do not edit.

Cost is always priced at a reference hosted rate, even for local runs.
Latency is only ever relative to a base setup, never absolute.
"""

from __future__ import annotations


def call_cost(tokens_in: int, tokens_out: int, in_per_mtok: float, out_per_mtok: float) -> float:
    return (tokens_in * in_per_mtok + tokens_out * out_per_mtok) / 1_000_000


def cost_per_search(total_cost: float, n_questions: int) -> float:
    return total_cost / n_questions if n_questions else 0.0


def relative_latency(mean_ms: dict[str, float | None], base: str = "A0") -> dict[str, float | None]:
    """Each setup's mean latency divided by the base setup's. None where a latency is unknown."""
    b = mean_ms.get(base)
    return {s: (None if v is None or not b else v / b) for s, v in mean_ms.items()}
