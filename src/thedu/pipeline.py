"""Run one config on dev questions: retrieve, plan, rank, judge, and save as it goes."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from thedu import data
from thedu.config import RunConfig
from thedu.judge import question_scores
from thedu.llm import Gateway, Usage
from thedu.paths import SPIKE_DIR, runs_dir
from thedu.planner import BlindPlanner, GroundedPlanner
from thedu.pricing import load_pricing
from thedu.ranker import rank
from thedu.registry import RunDir
from thedu.report import summarize
from thedu.retrievers import Bm25Retriever, DenseRetriever

log = logging.getLogger(__name__)

ORDER = ["A0", "A4", "A2", "A3"]


class RunRefused(RuntimeError):
    """The run finished but its numbers cannot be trusted (too many fallbacks)."""


def make_gateway(cfg: RunConfig, offline: bool = False) -> tuple[Gateway, dict[str, float]]:
    from thedu.paths import CONFIGS_DIR

    pricing = load_pricing(cfg.pricing)
    rate = pricing.rate(cfg.llm.model)
    budget = yaml.safe_load((CONFIGS_DIR / "budget.yaml").read_text())["total_usd"]
    gw = Gateway(
        cfg.llm, rate, cache_path=runs_dir() / "cache" / "llm.jsonl",
        legacy_path=SPIKE_DIR / "rewrites.json" if cfg.planner.use_spike_cache else None,
        offline=offline, budget_usd=budget,
    )
    return gw, {"input": rate.input_per_mtok_usd, "output": rate.output_per_mtok_usd}


def run_dev(cfg: RunConfig, offline: bool = False, restart: bool = False, root: Path | None = None) -> tuple[RunDir, dict]:
    docs, queries, qrels = data.load(cfg.dataset, cfg.split)
    qids = data.select_qids(queries, cfg.n, cfg.seed)
    doc_ids, texts = list(docs), list(docs.values())
    setups = [s for s in ORDER if s in cfg.setups]

    gateway, rates = make_gateway(cfg, offline)
    run_hash = cfg.hash(rates)
    run = RunDir.open(cfg, run_hash, root or runs_dir(), rates, restart)
    todo = {q: [s for s in setups if (s, q) not in run.rows] for q in qids}
    todo = {q: s for q, s in todo.items() if s}
    log.info("run %s: %d questions, %d to do", run_hash, len(qids), len(todo))

    bm25 = Bm25Retriever(doc_ids, texts, cfg.bm25.k1, cfg.bm25.b)
    dense_hits: dict[str, tuple[list, float]] = {}
    if "A4" in setups and any("A4" in s for s in todo.values()):
        dense = DenseRetriever.from_model(cfg.dataset, doc_ids, texts, cfg.dense.model, cfg.dense.max_seq_length)
        need = [q for q, s in todo.items() if "A4" in s]
        t0 = time.perf_counter()
        found = dense.search_many([queries[q] for q in need], cfg.k)
        per_q_ms = (time.perf_counter() - t0) * 1000 / len(need)
        dense_hits = {q: (h, per_q_ms) for q, h in zip(need, found)}
    blind = BlindPlanner(gateway, cfg.planner) if "A2" in setups else None
    grounded = GroundedPlanner(gateway, cfg.planner, docs, cfg.dataset) if "A3" in setups else None

    def timed_search(text: str):
        t0 = time.perf_counter()
        hits = bm25.search(text, cfg.k)
        return hits, (time.perf_counter() - t0) * 1000

    def do_question(qid: str) -> None:
        q = queries[qid]
        base = None
        for s in todo[qid]:
            usage, fallback, ms, searched = Usage(), False, 0.0, [q]
            if s == "A0":
                hits, ms = timed_search(q)
            elif s == "A4":
                hits, ms = dense_hits[qid]
            else:
                if s == "A2":
                    plan = blind.plan(q)
                else:
                    first, ms = timed_search(q)
                    plan = grounded.plan(q, first)
                usage, fallback, searched = plan.usage, plan.fallback, plan.queries
                lists = []
                for text in plan.queries:
                    h, t = timed_search(text)
                    lists.append(h)
                    ms += t
                hits = rank(lists, q)
            sc = question_scores(hits, qrels[qid])
            run.append(s, qid, hits, {
                **sc, "llm_calls": usage.calls, "tokens_in": usage.tokens_in, "tokens_out": usage.tokens_out,
                "cost_usd": usage.cost_usd, "uncosted_calls": usage.uncosted_calls,
                "untimed_calls": usage.untimed_calls, "llm_ms": usage.llm_ms, "retrieval_ms": ms,
                "fallback": fallback, "queries": searched,
            })

    try:
        with ThreadPoolExecutor(max_workers=cfg.llm.workers) as pool:
            list(pool.map(do_question, todo))  # re-raises the first error
        metrics = summarize(run.rows, setups, qids)
        worst = max((metrics["setups"][s]["fallbacks"] for s in setups), default=0) / max(len(qids), 1)
        if worst > cfg.planner.max_fallback_rate:
            run.finish(metrics, status="refused")
            raise RunRefused(
                f"{worst:.0%} of questions fell back to the raw question (limit "
                f"{cfg.planner.max_fallback_rate:.0%}), so rewrite results would be meaningless. "
                "Fix the cause (see the log) and rerun with --restart; successful rewrites are cached."
            )
        run.finish(metrics)
        return run, metrics
    finally:
        run.close()
