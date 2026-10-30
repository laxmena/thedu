"""Run one config on its questions: retrieve, plan, rank, judge, and save as it goes."""

from __future__ import annotations

import json
import logging
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import yaml

from thedu import data
from thedu.config import RunConfig, resolve_instruction
from thedu.judge import question_scores
from thedu.llm import Gateway, Usage
from thedu.paths import CONFIGS_DIR, ROOT, SPIKE_DIR, runs_dir
from thedu.planner import BlindPlanner, GroundedPlanner
from thedu.pricing import load_pricing, load_rerank_pricing
from thedu.ranker import RrfRanker, drop_zero_scores
from thedu.ranker.listwise import ListwiseReranker
from thedu.ranker.rerank import CrossEncoderReranker
from thedu.registry import RunDir
from thedu.report import summarize
from thedu.retrievers import Bm25Retriever, DenseRetriever
from thedu.tokens import estimate_tokens

log = logging.getLogger(__name__)

ORDER = ["A0", "A0C", "A4", "A2", "A3", "H0", "A5", "A0R", "A3R", "A4R", "A0L", "A3L", "A4L"]
# What each setup is built from. H0 and A5 fuse two components with the ranker.
DEPS = {"A0": ("A0",), "A0C": ("A0C",), "A4": ("A4",), "A2": ("A2",), "A3": ("A3",), "H0": ("A0", "A4"), "A5": ("A3", "A4"),
        "A0R": ("A0",), "A3R": ("A3",), "A4R": ("A4",), "A0L": ("A0",), "A3L": ("A3",), "A4L": ("A4",)}
RERANKED = {"A0R", "A3R", "A4R"}  # one component, its top reordered by the cross-encoder
LISTWISE = {"A0L", "A3L", "A4L"}  # one component, its top reordered by an LLM
BM25_BASED = {"A0", "A0C", "A2", "A3"}  # lists whose zero-score filler is dropped before fusion
PROBE_QUESTIONS = 30
PROBE_ATTEMPT = 9  # a salt that skips both caches, so the probe measures real calls


class RunRefused(RuntimeError):
    """The run finished but its numbers cannot be trusted (too many fallbacks)."""


@dataclass
class Component:
    hits: list[tuple[str, float]]
    usage: Usage
    fallback: bool
    retrieval_ms: float
    queries: list[str]


def fuse(ranker: RrfRanker, named_lists: dict[str, list[tuple[str, float]]], question: str) -> list[tuple[str, float]]:
    """Fuse component lists. BM25-based lists lose their zero-score filler first; embedding lists are kept whole."""
    lists = [drop_zero_scores(h) if name in BM25_BASED else h for name, h in named_lists.items()]
    return ranker.rank(lists, question)


def make_gateway(cfg: RunConfig, offline: bool = False, llm=None) -> tuple[Gateway, dict[str, float]]:
    """The gateway for the rewrite model, or for `llm` (the listwise judge) with the same cache and budget."""
    llm = llm or cfg.llm
    pricing = load_pricing(cfg.pricing)
    rate = pricing.rate(llm.model)
    budget = yaml.safe_load((CONFIGS_DIR / "budget.yaml").read_text())["total_usd"]
    main_cache = ROOT / "runs" / "cache" / "llm.jsonl"
    cache = ROOT / llm.cache_file
    elsewhere = 0.0
    if cache != main_cache and main_cache.exists():  # a separate cache still counts against the one budget
        elsewhere = sum(json.loads(ln).get("cost_usd") or 0.0 for ln in main_cache.read_text().splitlines() if ln.strip())
    gw = Gateway(
        llm, rate, cache_path=cache, spent_elsewhere_usd=elsewhere,
        legacy_path=SPIKE_DIR / "rewrites.json" if cfg.planner.use_spike_cache else None,
        offline=offline, budget_usd=budget,
        estimator=estimate_tokens if cfg.planner.use_spike_cache and llm is cfg.llm else None,  # spike-cache calls have no token split
    )
    rates = {"input": rate.input_per_mtok_usd, "output": rate.output_per_mtok_usd}
    if cfg.uses_rerank() and llm is cfg.llm:
        rates["rerank"] = load_rerank_pricing().price_per_mtok_usd
    if cfg.listwise and llm is cfg.llm:
        lr = pricing.rate(cfg.listwise.llm.model)
        rates["listwise_input"], rates["listwise_output"] = lr.input_per_mtok_usd, lr.output_per_mtok_usd
    return gw, rates


def run_dev(cfg: RunConfig, offline: bool = False, restart: bool = False, root: Path | None = None,
            probe: bool = False) -> tuple[RunDir, dict]:
    if cfg.split != "dev":
        raise PermissionError("`thedu dev` runs dev questions only. Test runs belong to `thedu parity` and `thedu test-eval`.")
    docs, queries, qrels = data.load(cfg.dataset, cfg.split)
    return execute(cfg, docs, queries, qrels, offline, restart, root, probe=probe)


def execute(cfg: RunConfig, docs: dict[str, str], queries: dict[str, str], qrels: dict[str, dict[str, int]],
            offline: bool = False, restart: bool = False, root: Path | None = None, kind: str = "dev",
            probe: bool = False, dense: DenseRetriever | None = None) -> tuple[RunDir, dict]:
    """Run a config's setups on the given questions and save the run. Callers decide which split's answers to pass.

    probe: if some LLM calls have no recorded latency (the spike cache), time a few fresh calls so the
    latency of setups that include the rewrite can still be reported relative to A0.
    """
    qids = data.select_qids(queries, cfg.n, cfg.seed)
    doc_ids, texts = list(docs), list(docs.values())
    setups = [s for s in ORDER if s in cfg.setups]

    gateway, rates = make_gateway(cfg, offline)
    run_hash = cfg.hash(rates)
    run = RunDir.open(cfg, run_hash, root or runs_dir(), rates, restart, kind)
    todo = {q: [s for s in setups if (s, q) not in run.rows] for q in qids}
    todo = {q: s for q, s in todo.items() if s}
    log.info("run %s: %d questions, %d to do", run_hash, len(qids), len(todo))

    bm25 = Bm25Retriever(doc_ids, texts, cfg.bm25.k1, cfg.bm25.b, cfg.bm25.language)
    dense_hits: dict[str, tuple[list, float]] = {}
    need_dense = [q for q, ss in todo.items() if any("A4" in DEPS[s] for s in ss)]
    if need_dense:
        dense = dense or DenseRetriever.from_model(
            cfg.dataset, doc_ids, texts, cfg.dense.model, cfg.dense.max_seq_length,
            resolve_instruction(cfg.dataset, cfg.dense.query_instruction))
        t0 = time.perf_counter()
        found = dense.search_many([queries[q] for q in need_dense], cfg.k)
        per_q_ms = (time.perf_counter() - t0) * 1000 / len(need_dense)
        dense_hits = {q: (h, per_q_ms) for q, h in zip(need_dense, found)}
    chars = None
    if "A0C" in setups:  # words plus character 3 to 5-grams; built here so it is rebuilt on every swapped corpus
        from thedu.retrievers.variants import KeywordIndex, Variant

        chars = KeywordIndex(doc_ids, texts, Variant(chars="plus", k1=cfg.bm25.k1, b=cfg.bm25.b))
    blind = BlindPlanner(gateway, cfg.planner) if "A2" in setups else None
    grounded = GroundedPlanner(gateway, cfg.planner, docs, cfg.dataset) if {"A3", "A5", "A3R"} & set(setups) else None
    ranker = RrfRanker(cfg.ranker.rrf_k, cfg.ranker.depth)
    reranker = CrossEncoderReranker(docs, cfg.rerank.model, cfg.rerank.depth, cfg.rerank.max_length) \
        if any(x in RERANKED for x in setups) else None
    listwise = None
    if any(x in LISTWISE for x in setups):
        if cfg.listwise is None:
            raise ValueError("the xL setups need a `listwise:` section in the config")
        lgw, _ = make_gateway(cfg, offline, cfg.listwise.llm)
        listwise = ListwiseReranker(docs, lgw, cfg.rerank.depth, cfg.listwise.words, cfg.listwise.prompt)

    def timed_search(text: str):
        t0 = time.perf_counter()
        hits = bm25.search(text, cfg.k)
        return hits, (time.perf_counter() - t0) * 1000

    def build(name: str, qid: str) -> Component:
        q = queries[qid]
        if name == "A0":
            hits, ms = timed_search(q)
            return Component(hits, Usage(), False, ms, [q])
        if name == "A0C":
            t0 = time.perf_counter()
            hits = chars.search(q, cfg.k)
            return Component(hits, Usage(), False, (time.perf_counter() - t0) * 1000, [q])
        if name == "A4":
            hits, ms = dense_hits[qid]
            return Component(hits, Usage(), False, ms, [q])
        ms = 0.0
        if name == "A2":
            plan = blind.plan(q)
        else:  # A3
            first, ms = timed_search(q)
            plan = grounded.plan(q, first)
        lists = []
        for text in plan.queries:
            h, t = timed_search(text)
            lists.append(h)
            ms += t
        return Component(ranker.rank(lists, q), plan.usage, plan.fallback, ms, plan.queries)

    def do_question(qid: str) -> None:
        built: dict[str, Component] = {}

        def comp(name: str) -> Component:
            if name not in built:
                built[name] = build(name, qid)
            return built[name]

        for s in todo[qid]:
            parts = [comp(d) for d in DEPS[s]]
            rank_ms, rerank_tokens, lw_stats, lw_usage = 0.0, 0, None, Usage()
            if s in LISTWISE:
                t0 = time.perf_counter()
                hits, lw_usage, lw_stats = listwise.rerank(parts[0].hits, queries[qid])
                rank_ms = (time.perf_counter() - t0) * 1000
            elif s in RERANKED:
                t0 = time.perf_counter()
                hits, rerank_tokens = reranker.rerank(parts[0].hits, queries[qid])
                rank_ms = (time.perf_counter() - t0) * 1000
            elif len(parts) == 1:
                hits = parts[0].hits
            else:
                t0 = time.perf_counter()
                hits = fuse(ranker, {d: c.hits for d, c in zip(DEPS[s], parts)}, queries[qid])
                rank_ms = (time.perf_counter() - t0) * 1000
            usage = Usage.combine([c.usage for c in parts] + [lw_usage])
            run.append(s, qid, hits, {
                **question_scores(hits, qrels[qid]),
                "llm_calls": usage.calls, "tokens_in": usage.tokens_in, "tokens_out": usage.tokens_out,
                "cost_usd": usage.cost_usd, "uncosted_calls": usage.uncosted_calls,
                "estimated_calls": usage.estimated_calls, "untimed_calls": usage.untimed_calls,
                "providers": usage.providers,
                "parts": {d: {"retrieval_ms": c.retrieval_ms, "llm_ms": c.usage.llm_ms, "untimed": c.usage.untimed_calls}
                          for d, c in zip(DEPS[s], parts)},
                "rank_ms": rank_ms,
                "listwise": None if lw_stats is None else {"missing": lw_stats.missing, "repeated": lw_stats.repeated,
                                                           "invalid": lw_stats.invalid, "failed": lw_stats.failed,
                                                           "fallback": lw_stats.fallback},
                "rerank_tokens": rerank_tokens,
                "rerank_cost_usd": rerank_tokens * rates.get("rerank", 0.0) / 1e6,
                "fallback": any(c.fallback for c in parts),
                "queries": next((c.queries for d, c in zip(DEPS[s], parts) if d in ("A2", "A3")), [queries[qid]]),
            })

    try:
        with ThreadPoolExecutor(max_workers=cfg.llm.workers) as pool:
            list(pool.map(do_question, todo))  # re-raises the first error
        probe_ms = _llm_probe(run, gateway, grounded, bm25, queries, qids, cfg.k) if probe and not offline else _saved_probe(run)
        metrics = summarize(run.rows, setups, qids, cfg.dataset, probe_ms)
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


def _saved_probe(run: RunDir) -> float | None:
    f = run.path / "llm_probe.json"
    return json.loads(f.read_text())["median_ms"] if f.exists() else None


def _llm_probe(run: RunDir, gateway: Gateway, grounded: GroundedPlanner | None, bm25: Bm25Retriever,
               queries: dict[str, str], qids: list[str], k: int) -> float | None:
    """Median latency of a few fresh rewrite calls, used where cached calls have no recorded latency."""
    untimed = any(r["untimed_calls"] for r in run.rows.values())
    if not untimed or grounded is None:
        return _saved_probe(run)
    if (saved := _saved_probe(run)) is not None:
        return saved
    times = []
    for qid in qids[:PROBE_QUESTIONS]:
        prompt = grounded.prompt_for(queries[qid], bm25.search(queries[qid], k))
        r = gateway.call(prompt, attempt=PROBE_ATTEMPT)
        if not r.failed and r.latency_ms is not None:
            times.append(r.latency_ms)
    if not times:
        return None
    median = statistics.median(times)
    (run.path / "llm_probe.json").write_text(json.dumps(
        {"median_ms": median, "n_calls": len(times),
         "why": "cached rewrites carry no latency; this is the median of fresh calls to the same model"}, indent=2))
    return median
