"""Turn per-question rows into metrics.json (via the judge)."""

from __future__ import annotations

import numpy as np

from thedu.judge import bootstrap_interval, cost_per_search, paired_difference, relative_latency


def summarize(rows: dict[tuple[str, str], dict], setups: list[str], qids: list[str]) -> dict:
    def col(setup: str, key: str) -> np.ndarray:
        return np.array([rows[(setup, q)][key] for q in qids], dtype=float)

    out: dict = {"n_questions": len(qids), "setups": {}, "paired_vs_A0": {}}
    mean_ms: dict[str, float | None] = {}
    for s in setups:
        nd, rc = bootstrap_interval(col(s, "ndcg10")), bootstrap_interval(col(s, "recall100"))
        rs = [rows[(s, q)] for q in qids]
        uncosted = sum(r["uncosted_calls"] for r in rs)
        untimed = sum(r["untimed_calls"] for r in rs)
        total_cost = sum(r["cost_usd"] for r in rs)
        mean_ms[s] = None if untimed else float(np.mean([r["retrieval_ms"] + r["llm_ms"] for r in rs]))
        out["setups"][s] = {
            "ndcg10": {"mean": nd.mean, "lo": nd.lo, "hi": nd.hi},
            "recall100": {"mean": rc.mean, "lo": rc.lo, "hi": rc.hi},
            "llm_calls": sum(r["llm_calls"] for r in rs),
            "tokens_in": sum(r["tokens_in"] for r in rs),
            "tokens_out": sum(r["tokens_out"] for r in rs),
            # null when any call has no recorded tokens (spike cache) or the setup has no reference price (embeddings)
            "cost_per_search_usd": None if uncosted or s == "A4" else cost_per_search(total_cost, len(qids)),
            "uncosted_calls": uncosted,
            "fallbacks": sum(r["fallback"] for r in rs),
        }
    rel = relative_latency(mean_ms)
    for s in setups:
        out["setups"][s]["latency_relative_to_A0"] = rel.get(s)
    if "A0" in setups:
        for s in setups:
            if s != "A0":
                p = paired_difference(col("A0", "ndcg10"), col(s, "ndcg10"))
                out["paired_vs_A0"][s] = {"ndcg10_diff": p.mean, "lo": p.lo, "hi": p.hi, "wins": p.wins, "losses": p.losses}
    return out
