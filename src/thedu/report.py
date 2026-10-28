"""Turn per-question rows into metrics.json (via the judge)."""

from __future__ import annotations

import numpy as np

from thedu.judge import bootstrap_interval, cost_per_search, paired_difference, relative_latency

# (base, other): "other vs base" is other minus base, question by question.
PAIRS = [("H0", "A5"), ("A4", "A5"), ("A4", "H0"), ("A3", "A5")]
USES_EMBEDDINGS = {"A4", "H0", "A5"}  # embedding search has no reference price yet
UNFUSED_BM25 = {"A0", "A2", "A3"}  # their top 100 can hold zero-score filler; fused lists have it removed


def _part_ms(part: dict, probe_ms: float | None) -> float | None:
    """One component's latency: search time plus LLM time (a fresh-call median stands in for unrecorded calls)."""
    llm = part["llm_ms"]
    if part["untimed"]:
        if probe_ms is None:
            return None
        llm += probe_ms * part["untimed"]
    return part["retrieval_ms"] + llm


def _latencies(rows: list[dict], probe_ms: float | None) -> tuple[float | None, float | None]:
    """Mean per-question latency if components run in parallel (max) and one after another (sum)."""
    par, seq = [], []
    for r in rows:
        ms = [_part_ms(p, probe_ms) for p in r["parts"].values()]
        if any(m is None for m in ms):
            return None, None
        par.append(max(ms) + r["rank_ms"])
        seq.append(sum(ms) + r["rank_ms"])
    return float(np.mean(par)), float(np.mean(seq))


def summarize(rows: dict[tuple[str, str], dict], setups: list[str], qids: list[str], dataset: str,
              probe_ms: float | None = None) -> dict:
    def col(setup: str, key: str) -> np.ndarray:
        return np.array([rows[(setup, q)][key] for q in qids], dtype=float)

    out: dict = {"n_questions": len(qids), "dataset": dataset, "llm_latency_probe_ms": probe_ms,
                 "setups": {}, "paired": {}, "extras": {}}
    par_ms: dict[str, float | None] = {}
    seq_ms: dict[str, float | None] = {}
    for s in setups:
        nd, rc = bootstrap_interval(col(s, "ndcg10")), bootstrap_interval(col(s, "recall100"))
        rs = [rows[(s, q)] for q in qids]
        uncosted = sum(r["uncosted_calls"] for r in rs)
        estimated = sum(r["estimated_calls"] for r in rs)
        llm_cost = None if uncosted else cost_per_search(sum(r["cost_usd"] for r in rs), len(qids))
        par_ms[s], seq_ms[s] = _latencies(rs, probe_ms)
        out["setups"][s] = {
            "ndcg10": {"mean": nd.mean, "lo": nd.lo, "hi": nd.hi},
            "recall100": {"mean": rc.mean, "lo": rc.lo, "hi": rc.hi},
            "recall100_is_soft": dataset == "nfcorpus" and s in UNFUSED_BM25,  # many questions match <100 documents; the rest is arbitrary filler
            "llm_calls": sum(r["llm_calls"] for r in rs),
            "tokens_in": sum(r["tokens_in"] for r in rs),
            "tokens_out": sum(r["tokens_out"] for r in rs),
            "llm_cost_per_search_usd": llm_cost,  # rewrite calls only
            "llm_cost_is_estimate": estimated > 0,  # tokens counted offline; hidden thinking tokens are not included
            # The full cost per search needs a reference price for embedding search, which is not chosen yet.
            "cost_per_search_usd": None if s in USES_EMBEDDINGS else llm_cost,
            "uncosted_calls": uncosted,
            "estimated_calls": estimated,
            "fallbacks": sum(r["fallback"] for r in rs),
        }
    rel_par, rel_seq = relative_latency(par_ms), relative_latency(seq_ms)
    for s in setups:
        out["setups"][s]["latency_relative_to_A0"] = (
            None if rel_par.get(s) is None else {"parallel": rel_par[s], "sequential": rel_seq[s]})

    def compare(base: str, other: str) -> None:
        p = paired_difference(col(base, "ndcg10"), col(other, "ndcg10"))
        out["paired"][f"{other} vs {base}"] = {"ndcg10_diff": p.mean, "lo": p.lo, "hi": p.hi, "wins": p.wins, "losses": p.losses}

    if "A0" in setups:
        for s in setups:
            if s != "A0":
                compare("A0", s)
    for base, other in PAIRS:
        if base in setups and other in setups and f"{other} vs {base}" not in out["paired"]:
            compare(base, other)

    if "A4" in setups:
        a4 = col("A4", "ndcg10")
        for other in ("H0", "A5"):
            if other in setups:
                o = col(other, "ndcg10")
                out["extras"][f"{other}_vs_A4_misses"] = {
                    "a4_misses": int((a4 == 0).sum()),  # nothing relevant in A4's top 10
                    "rescued": int(((a4 == 0) & (o > 0)).sum()),
                    "a4_correct_at_1": int((a4 >= 0.99).sum()),
                    "pushed_down": int(((a4 >= 0.99) & (o < 0.99)).sum()),
                }
    if "A3" in setups and "A4" in setups:
        out["extras"]["ceiling_best_of_A3_A4"] = float(np.maximum(col("A3", "ndcg10"), col("A4", "ndcg10")).mean())
    return out
