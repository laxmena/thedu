"""The keyword-index sweep, RM3, and embedding-filtered feedback (JOURNAL.md, 2026-10-28).

Dev questions and dev answers only; no LLM calls; the index is plain BM25 with no model at index time.
Every result file records its settings and the code version. Comparisons are paired, question by question,
against the current A0, which is recomputed here and checked against the reference run's A0.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np

from thedu import data
from thedu.judge import paired_difference, question_scores
from thedu.paths import ROOT
from thedu.registry import code_version
from thedu.retrievers.variants import KeywordIndex, Variant, words

DATASETS = ("scifact", "nfcorpus", "fiqa")
# The pinned-provider runs whose A0, A3 and A4 are the references (same questions, same judge).
REF_RUN = {"scifact": "eed781f01f", "nfcorpus": "e9b3631de6", "fiqa": "890ad993c9"}
OUT = ROOT / "runs" / "sweeps"
K = 100

RM3 = {"fb_docs": 10, "fb_terms": 15, "orig_weight": 0.5}
EFB = {"feedback_docs": 20, "min_docs": 2, "keep_terms": 12, "query_repeat": 3, "min_letters": 3}


def index_variants() -> list[Variant]:
    """As pre-registered: one change at a time against the current index."""
    vs = [Variant(stemmer="porter"), Variant(stemmer="none")]
    vs += [Variant(k1=k1, b=b) for k1, b in product((0.6, 0.9, 1.2, 1.5), (0.2, 0.4, 0.6, 0.75)) if (k1, b) != (0.9, 0.4)]
    vs += [Variant(title_weight=w) for w in (0.5, 1.0, 2.0)]
    vs += [Variant(bigrams=True), Variant(chars="only"), Variant(chars="plus")]
    return vs


def reference_scores(dataset: str) -> dict[str, dict[str, dict[str, float]]]:
    """setup -> qid -> {ndcg10, recall100} from the reference run."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    for line in (ROOT / "runs" / REF_RUN[dataset] / "per_question.jsonl").read_text().splitlines():
        r = json.loads(line)
        out.setdefault(r["setup"], {})[r["qid"]] = {"ndcg10": r["ndcg10"], "recall100": r["recall100"]}
    return out


def _score(search, queries: dict[str, str], qrels: dict, qids: list[str], text=None) -> dict[str, dict[str, float]]:
    return {q: question_scores(search(text(q) if text else queries[q], K), qrels[q]) for q in qids}


def paired(base: dict, other: dict, qids: list[str], metric: str) -> dict:
    p = paired_difference([base[q][metric] for q in qids], [other[q][metric] for q in qids])
    return {"diff": p.mean, "lo": p.lo, "hi": p.hi, "wins": p.wins, "losses": p.losses}


def summarize(scores: dict, base: dict, qids: list[str]) -> dict:
    return {"ndcg10": float(np.mean([scores[q]["ndcg10"] for q in qids])),
            "recall100": float(np.mean([scores[q]["recall100"] for q in qids])),
            "vs_A0_ndcg10": paired(base, scores, qids, "ndcg10"),
            "vs_A0_recall100": paired(base, scores, qids, "recall100")}


def _save(name: str, dataset: str, payload: dict) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}-{dataset}.json"
    path.write_text(json.dumps({"dataset": dataset, "reference_run": REF_RUN[dataset], **code_version(), **payload}, indent=1))
    return path


def _prepare(dataset: str):
    docs, queries, qrels = data.load(dataset, "dev")
    qids = sorted(queries)
    base_index = KeywordIndex(list(docs), list(docs.values()), Variant())
    base = _score(base_index.search, queries, qrels, qids)
    ref = reference_scores(dataset)
    drift = max(abs(base[q]["ndcg10"] - ref["A0"][q]["ndcg10"]) for q in qids)
    if drift > 1e-9:
        raise RuntimeError(f"the recomputed A0 differs from run {REF_RUN[dataset]} (max nDCG@10 gap {drift:.2e}); stop")
    return docs, queries, qrels, qids, base_index, base, ref


# ---------- index sweep ----------

def index_sweep(dataset: str) -> Path:
    docs, queries, qrels, qids, _, base, _ = _prepare(dataset)
    titled = data.read_titled_docs(dataset)
    ids = list(docs)
    has_titles = any(t for t, _ in titled.values())
    results: dict[str, dict] = {"current": {"ndcg10": float(np.mean([base[q]["ndcg10"] for q in qids])),
                                            "recall100": float(np.mean([base[q]["recall100"] for q in qids]))}}
    for v in index_variants():
        t0 = time.time()
        if v.title_weight is not None:
            if not has_titles:
                results[v.label()] = {"not_applicable": "this corpus has no titles"}
                continue
            idx = KeywordIndex(ids, [titled[d][1] for d in ids], v, [titled[d][0] for d in ids])
        else:
            idx = KeywordIndex(ids, list(docs.values()), v)
        scores = _score(idx.search, queries, qrels, qids)
        results[v.label()] = {**summarize(scores, base, qids), "seconds": round(time.time() - t0, 1)}
    return _save("index", dataset, {"n_questions": len(qids), "variants": results})


# ---------- RM3 ----------

def _weighted_search(index: KeywordIndex, weights: dict[str, float], k: int) -> list[tuple[str, float]]:
    s = np.zeros(len(index.doc_ids))
    for term, w in weights.items():
        s += w * index._body.get_scores([term])
    from thedu.retrievers.order import top_k

    return [(index.doc_ids[i], float(s[i])) for i in top_k(s, index._ranks, k)]


def rm3_weights(question: str, first: list[tuple[str, float]], docs: dict[str, str], cfg: dict = RM3) -> dict[str, float] | None:
    """Query term weights after pseudo-relevance feedback; None if the first search matched nothing."""
    top = [(d, s) for d, s in first[: cfg["fb_docs"]] if s > 0]
    total = sum(s for _, s in top)
    if total <= 0:
        return None
    exp: Counter[str] = Counter()
    for (d, s), toks in zip(top, words([docs[d] for d, _ in top], "snowball")):
        if toks:
            for t, c in Counter(toks).items():
                exp[t] += (s / total) * c / len(toks)
    best = sorted(exp.items(), key=lambda x: (-x[1], x[0]))[: cfg["fb_terms"]]
    z = sum(w for _, w in best)
    qt = words([question], "snowball")[0]
    out: Counter[str] = Counter()
    for t, c in Counter(qt).items():
        out[t] += cfg["orig_weight"] * c / len(qt)
    for t, w in best:
        out[t] += (1 - cfg["orig_weight"]) * w / z
    return dict(out)


def rm3_sweep(dataset: str) -> Path:
    docs, queries, qrels, qids, idx, base, ref = _prepare(dataset)
    scores, expansions = {}, {}
    for q in qids:
        first = idx.search(queries[q], K)
        w = rm3_weights(queries[q], first, docs)
        hits = first if w is None else _weighted_search(idx, w, K)
        scores[q] = question_scores(hits, qrels[q])
        expansions[q] = None if w is None else sorted(w, key=lambda t: -w[t])[:8]
    out = {"n_questions": len(qids), "settings": RM3, **summarize(scores, base, qids),
           "vs_A3_ndcg10": paired(ref["A3"], scores, qids, "ndcg10"),
           "vs_A4_ndcg10": paired(ref["A4"], scores, qids, "ndcg10"),
           "per_question": scores, "top_terms_example": dict(list(expansions.items())[:15])}
    return _save("rm3", dataset, out)


# ---------- embedding-filtered feedback ----------

_WORD = re.compile(r"[^\W\d_]+")


def candidate_terms(question: str, top_docs: list[str], cfg: dict = EFB) -> list[str]:
    """Distinct lowercase word forms in at least `min_docs` of the feedback documents, not stopwords, not already
    in the question (compared by stem)."""
    import bm25s.stopwords as sw

    stop = set(sw.STOPWORDS_EN)
    qstems = set(words([question], "snowball")[0])
    df: Counter[str] = Counter()
    for text in top_docs:
        df.update({w for w in _WORD.findall(text.lower()) if len(w) >= cfg["min_letters"] and w not in stop})
    cands = sorted(w for w, c in df.items() if c >= cfg["min_docs"])
    stems = words(cands, "snowball") if cands else []
    return [w for w, st in zip(cands, stems) if not (set(st) & qstems) and st]  # st empty: a stopword by another route


def efb_sweep(dataset: str, probe_questions: int = 30) -> Path:
    from sentence_transformers import SentenceTransformer

    from thedu.config import resolve_instruction

    docs, queries, qrels, qids, idx, base, ref = _prepare(dataset)
    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    model.max_seq_length = 512
    prefix = f"Instruct: {resolve_instruction(dataset, 'task')}\nQuery:"
    cache: dict[str, np.ndarray] = {}

    def choose(q: str, first: list, use_cache: bool) -> list[str]:
        cands = candidate_terms(queries[q], [docs[d] for d, s in first[: EFB["feedback_docs"]] if s > 0])
        if not cands:
            return []
        todo = [t for t in cands if not (use_cache and t in cache)]
        if todo:
            vecs = model.encode(todo, batch_size=64, normalize_embeddings=True)
            for t, v in zip(todo, vecs):
                if use_cache:
                    cache[t] = v
            local = dict(zip(todo, vecs))
        else:
            local = {}
        tv = np.stack([cache[t] if use_cache and t in cache else local[t] for t in cands])
        qv = model.encode([prefix + queries[q]], normalize_embeddings=True)[0]
        sims = tv @ qv
        order = sorted(range(len(cands)), key=lambda i: (-sims[i], cands[i]))[: EFB["keep_terms"]]
        return [cands[i] for i in order]

    scores, chosen = {}, {}
    for q in qids:
        first = idx.search(queries[q], K)
        terms = choose(q, first, True)
        text = " ".join([queries[q]] * EFB["query_repeat"] + terms)
        scores[q] = question_scores(idx.search(text, K), qrels[q])
        chosen[q] = terms
    # Latency with no term-vector cache: first search + feedback (terms and question embedded) + second search.
    t_first, t_total = [], []
    for q in qids[:probe_questions]:
        t0 = time.perf_counter()
        first = idx.search(queries[q], K)
        t1 = time.perf_counter()
        terms = choose(q, first, False)
        idx.search(" ".join([queries[q]] * EFB["query_repeat"] + terms), K)
        t_first.append(t1 - t0)
        t_total.append(time.perf_counter() - t0)
    rm3_file = OUT / f"rm3-{dataset}.json"
    out = {"n_questions": len(qids), "settings": EFB, **summarize(scores, base, qids),
           "vs_A3_ndcg10": paired(ref["A3"], scores, qids, "ndcg10"),
           "vs_A4_ndcg10": paired(ref["A4"], scores, qids, "ndcg10"),
           "latency_relative_to_A0": float(np.mean(t_total) / np.mean(t_first)),
           "latency_note": f"{probe_questions} questions, no term-vector cache; measured while other jobs shared the machine",
           "per_question": scores, "chosen_terms_example": dict(list(chosen.items())[:15])}
    if rm3_file.exists():
        rm3 = json.loads(rm3_file.read_text())["per_question"]
        out["vs_RM3_ndcg10"] = paired(rm3, scores, qids, "ndcg10")
    return _save("efb", dataset, out)


# ---------- the pre-registered keep rule ----------

def keep_verdicts(name: str) -> dict[str, dict]:
    """Variant (or the method) -> datasets where its paired nDCG@10 range is above zero; kept if on at least two of three."""
    files = {d: json.loads((OUT / f"{name}-{d}.json").read_text()) for d in DATASETS if (OUT / f"{name}-{d}.json").exists()}
    rows: dict[str, dict] = {}
    for d, f in files.items():
        items = f["variants"] if "variants" in f else {name: f}
        for label, r in items.items():
            if "vs_A0_ndcg10" in r:
                e = rows.setdefault(label, {"datasets": {}})
                e["datasets"][d] = r["vs_A0_ndcg10"]
    for e in rows.values():
        wins = [d for d, p in e["datasets"].items() if p["lo"] > 0]
        e["above_zero_on"], e["kept"] = wins, len(wins) >= 2
        e["mean_gain"] = float(np.mean([p["diff"] for p in e["datasets"].values()]))
    return rows
