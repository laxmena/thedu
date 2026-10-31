"""H2b, the depth sweep (JOURNAL.md, 2026-10-30 pre-registration). FiQA dev, 150 questions (seed 7).

The cross-encoder scores A0's top 500 once per question; the lists for depth 50, 100, 200 and 500 come from the same
scores. A4's top 50 are scored in the same job (question by question, interleaved) so the benchmark is timed under the
same load. Scores are saved per question and the job resumes. Retrieves with BM25 only; no LLM call; dev answers only.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from thedu import data
from thedu.analysis.diagnosis import read_trec
from thedu.analysis.hypotheses import A4R, RR
from thedu.analysis.steps import OUT, load_rows
from thedu.config import RerankConfig
from thedu.judge import ndcg_at_k, paired_difference, recall_at_k
from thedu.m5.analysis import verdict
from thedu.paths import ROOT
from thedu.pricing import load_rerank_pricing
from thedu.ranker.rerank import CrossEncoderReranker
from thedu.retrievers import Bm25Retriever

DEPTHS = (50, 100, 200, 500)
N, SEED = 150, 7
SCORES = OUT / "h2b-scores.jsonl"


def collect() -> dict:
    docs, queries, qrels = data.load("fiqa", "dev")
    qids = data.select_qids(queries, N, SEED)
    cfg = RerankConfig()
    rr = CrossEncoderReranker(docs, cfg.model, cfg.depth, cfg.max_length)
    rr._load()  # noqa: SLF001 - the scorer and the token counter, loaded once
    bm = Bm25Retriever(list(docs), list(docs.values()))
    a4 = read_trec(ROOT / "runs" / A4R["fiqa"] / "A4.trec")
    done = {json.loads(ln)["qid"]: json.loads(ln) for ln in SCORES.read_text().splitlines()} if SCORES.exists() else {}
    OUT.mkdir(parents=True, exist_ok=True)
    load_start = os.getloadavg()
    with open(SCORES, "a") as f:
        for q in qids:
            if q in done:
                continue
            t0 = time.perf_counter()
            h4 = [d for d, _ in a4[q]][:50]
            s4 = rr._scorer([(queries[q], docs[d]) for d in h4])  # noqa: SLF001
            ms4 = (time.perf_counter() - t0) * 1000
            cands = [d for d, s in bm.search(queries[q], 500) if s > 0]
            scores, seg_ms, lo = [], [], 0
            for hi in DEPTHS:
                part = cands[lo:hi]
                t0 = time.perf_counter()
                if part:
                    scores += rr._scorer([(queries[q], docs[d]) for d in part])  # noqa: SLF001
                seg_ms.append((time.perf_counter() - t0) * 1000)
                lo = hi
            qt = rr._count(queries[q])  # noqa: SLF001
            a4_tok = [qt + min(rr._count(docs[d]), max(cfg.max_length - qt, 0)) for d in h4]  # noqa: SLF001
            tok = [qt + min(rr._count(docs[d]), max(cfg.max_length - qt, 0)) for d in cands]  # noqa: SLF001
            rec = {"qid": q, "a0_candidates": cands, "a0_scores": scores, "a0_tokens": tok, "seg_ms": seg_ms,
                   "a4_top50": h4, "a4_scores": s4, "a4_ms": ms4, "a4_tokens": a4_tok, "load": os.getloadavg()[0]}
            f.write(json.dumps(rec) + "\n")
            f.flush()
    rows = [json.loads(ln) for ln in SCORES.read_text().splitlines()]
    return {"qids": qids, "rows": {r["qid"]: r for r in rows}, "load_start": load_start, "load_end": os.getloadavg()}


def head(cands: list[str], scores: list[float], d: int) -> list[str]:
    """The reranked list: the first d candidates by score (stable on ties), then the rest in A0's order."""
    order = sorted(range(min(d, len(cands))), key=lambda i: -scores[i])
    return [cands[i] for i in order] + cands[d:]


def analyze(coll: dict) -> dict:
    _, queries, qrels = data.load("fiqa", "dev")
    qids = coll["qids"]
    rows = coll["rows"]
    price = load_rerank_pricing().price_per_mtok_usd
    base = load_rows(A4R["fiqa"])["A4R"]
    a4r_trec = read_trec(ROOT / "runs" / A4R["fiqa"] / "A4R.trec")
    old_a0r = load_rows(RR["fiqa"])["A0R"]
    bench = {"ndcg10": np.array([base[q]["ndcg10"] for q in qids]),
             "recall10": np.array([recall_at_k([d for d, _ in a4r_trec[q]], qrels[q], 10) for q in qids])}
    fresh_a4 = np.array([ndcg_at_k(head(rows[q]["a4_top50"], rows[q]["a4_scores"], 50), qrels[q], 10) for q in qids])
    out: dict = {"n": len(qids), "load_start": coll["load_start"], "load_end": coll["load_end"],
                 "check_A4R_fresh_vs_run_ndcg10_mean_abs_diff": float(np.mean(np.abs(fresh_a4 - bench["ndcg10"]))),
                 "benchmark_A4R_depth50": {"ndcg10": float(bench["ndcg10"].mean()), "recall10": float(bench["recall10"].mean())},
                 "depths": {}}
    t4 = np.mean([rows[q]["a4_ms"] for q in qids])
    out["benchmark_A4R_depth50"]["cost_per_search_usd"] = float(np.mean([sum(rows[q]["a4_tokens"]) for q in qids]) * price / 1e6)
    cum = np.cumsum([np.mean([rows[q]["seg_ms"][i] for q in qids]) for i in range(len(DEPTHS))])
    for i, d in enumerate(DEPTHS):
        nd = np.array([ndcg_at_k(head(rows[q]["a0_candidates"], rows[q]["a0_scores"], d), qrels[q], 10) for q in qids])
        rc = np.array([recall_at_k(head(rows[q]["a0_candidates"], rows[q]["a0_scores"], d), qrels[q], 10) for q in qids])
        pairs = np.array([min(d, len(rows[q]["a0_candidates"])) for q in qids])
        cost = np.array([sum(rows[q]["a0_tokens"][:d]) * price / 1e6 for q in qids])
        pn, pr = paired_difference(bench["ndcg10"], nd), paired_difference(bench["recall10"], rc)
        out["depths"][d] = {"ndcg10": float(nd.mean()), "recall10": float(rc.mean()), "pairs_per_search": float(pairs.mean()),
                            "cost_per_search_usd": float(cost.mean()), "relative_time_to_A4R_depth50": float(cum[i] / t4),
                            "vs_A4R_ndcg10": {"diff": pn.mean, "lo": pn.lo, "hi": pn.hi, "verdict": verdict(pn)},
                            "vs_A4R_recall10": {"diff": pr.mean, "lo": pr.lo, "hi": pr.hi}}
        if d == 50:
            out["check_A0R_depth50_vs_run_ndcg10_mean_abs_diff"] = float(np.mean(np.abs(nd - np.array([old_a0r[q]["ndcg10"] for q in qids]))))
    c50 = out["depths"][50]["cost_per_search_usd"]
    for d in DEPTHS:
        out["depths"][d]["cost_relative_to_A0R_depth50"] = out["depths"][d]["cost_per_search_usd"] / c50
    out["time_note"] = "relative_time_to_A4R_depth50 = cumulative scoring time of the first d A0 candidates over the scoring time of A4's top 50 in the same job"
    (OUT / "h2b-depth-sweep.json").write_text(json.dumps(out, indent=2))
    return out
