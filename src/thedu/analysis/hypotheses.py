"""H1 to H4 (JOURNAL.md, 2026-10-30 pre-registration): free analyses on existing dev runs. No LLM call, no test answer."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from thedu import data
from thedu.analysis.diagnosis import read_trec
from thedu.analysis.steps import OUT, load_rows
from thedu.judge import paired_difference
from thedu.m5.analysis import verdict
from thedu.paths import ROOT
from thedu.ranker import RrfRanker
from thedu.retrievers import Bm25Retriever

RR = {"nfcorpus": "8dbc179fa6", "scifact": "d8fe5f9aff", "fiqa": "00fb868eaa"}  # A0, A3, A4, A0R, A3R
A4R = {"nfcorpus": "3598a84a25", "scifact": "6972fad3ba", "fiqa": "4432badf81"}  # A4, A4R
GROUPS = {"a": "both have one", "b": "only A4", "c": "only the keyword list", "d": "neither"}
ADVICE = ("should i", "is it worth", "is it better", "what's the best", "would you", "good idea", "recommend")
DEPTH = 50


def top50(hits: list[tuple[str, float]]) -> list[str]:
    return [d for d, s in hits if s > 0][:DEPTH]


def group_of(first: list[str], a4: list[str], rels: dict[str, int]) -> str:
    x, y = any(d in rels for d in first), any(d in rels for d in a4)
    return "a" if x and y else "b" if y else "c" if x else "d"


def _rows(dataset: str) -> dict[str, dict[str, np.ndarray]]:
    rr, a4 = load_rows(RR[dataset]), load_rows(A4R[dataset])
    qids = sorted(rr["A0"])
    pick = lambda rows, s: np.array([rows[s][q]["ndcg10"] for q in qids])  # noqa: E731
    return {"qids": qids, "A0R": pick(rr, "A0R"), "A3R": pick(rr, "A3R"), "A4R": pick(a4, "A4R"), "A4": pick(a4, "A4")}


def _test(base: np.ndarray, arm: np.ndarray) -> dict:
    p = paired_difference(base, arm)
    return {"diff": p.mean, "lo": p.lo, "hi": p.hi, "wins": p.wins, "losses": p.losses, "verdict": verdict(p)}


def h1(dataset: str) -> dict:
    _, _, qrels = data.load(dataset, "dev")
    r = _rows(dataset)
    qids = r["qids"]
    runs = {s: read_trec(ROOT / "runs" / (A4R[dataset] if s == "A4" else RR[dataset]) / f"{s}.trec") for s in ("A0", "A3", "A4")}
    out: dict = {"n_questions": len(qids)}
    for first, arm in (("A0", "A0R"), ("A3", "A3R")):
        grp = np.array([group_of(top50(runs[first][q]), top50(runs["A4"][q]), qrels[q]) for q in qids])
        diff = r[arm] - r["A4R"]
        rows = {}
        for g in "abcd":
            m = grp == g
            n = int(m.sum())
            rows[g] = {"label": GROUPS[g], "n": n, "share": n / len(qids), "mean_diff": float(diff[m].mean()) if n else None,
                       "contribution": float(m.mean() * diff[m].mean()) if n else 0.0,
                       "test": _test(r["A4R"][m], r[arm][m]) if n >= 5 else None}
        out[f"{arm}_minus_A4R"] = {"overall": float(diff.mean()), "groups": rows,
                                   "overall_test": _test(r["A4R"], r[arm]), "group_a_matches": bool(rows["a"]["test"] and rows["a"]["test"]["lo"] > -0.01)}
        if first == "A0":
            out["groups_A0"] = {q: g for q, g in zip(qids, grp)}
    return out


def _bins(rank: int | None) -> str:
    if rank is None or rank > 1000:
        return "not in top 1000"
    return "51-100" if rank <= 100 else "101-500" if rank <= 500 else "501-1000"


def h2(dataset: str, groups_a0: dict[str, str]) -> dict:
    docs, queries, qrels = data.load(dataset, "dev")
    qids = sorted(queries)
    bm = Bm25Retriever(list(docs), list(docs.values()))
    rr = load_rows(RR[dataset])
    saved = read_trec(ROOT / "runs" / RR[dataset] / "A3.trec")
    deep, shallow = RrfRanker(60, 1000), RrfRanker(60, 100)
    ranked: dict[str, dict[str, list[str]]] = {"A0": {}, "A3": {}}
    ranked_all: dict[str, dict[str, list[str]]] = {"A0": {}, "A3": {}}
    for q in qids:
        h = bm.search(queries[q], 1000)
        ranked_all["A0"][q] = [d for d, _ in h]
        ranked["A0"][q] = [d for d, s in h if s > 0]
        texts = rr["A3"][q]["queries"]
        lists = [bm.search(t, 1000) for t in texts]
        fused = deep.rank(lists, queries[q])
        matched = {d for hits in lists for d, s in hits if s > 0}
        ranked_all["A3"][q] = [d for d, _ in fused]
        ranked["A3"][q] = [d for d, _ in fused if d in matched]
        check = [d for d, _ in shallow.rank([x[:100] for x in lists], queries[q])]
        if check != [d for d, _ in saved[q]]:
            raise RuntimeError(f"A3 does not reproduce the saved list for question {q}; the deep lists would not be the run's")
    out: dict = {"A3_top100_reproduces_saved_run": True, "recall": {}}
    for s in ("A0", "A3"):
        out["recall"][s] = {}
        for k in (100, 500, 1000):
            for name, lists_ in (("matched_only", ranked[s]), ("as_ranked", ranked_all[s])):
                v = [len({d for d in lists_[q][:k]} & set(qrels[q])) / len(qrels[q]) for q in qids]
                out["recall"][s][f"{name}@{k}"] = float(np.mean(v))
    bins = {"51-100": 0, "101-500": 0, "501-1000": 0, "not in top 1000": 0}
    n_b = 0
    for q in qids:
        if groups_a0[q] != "b":
            continue
        n_b += 1
        pos = [i + 1 for i, d in enumerate(ranked["A0"][q]) if d in qrels[q]]
        bins[_bins(pos[0] if pos else None)] += 1
    out["group_b_first_relevant_rank_in_A0"] = {"n": n_b, **bins,
                                                 "within_top_1000_share": (n_b - bins["not in top 1000"]) / n_b if n_b else None}
    return out


def h3(path: Path = OUT / "h3-fiqa-review.md", n: int = 20) -> dict:
    docs, queries, qrels = data.load("fiqa", "dev")
    r = _rows("fiqa")
    qids = r["qids"]
    diff = r["A0R"] - r["A4R"]
    order = sorted(range(len(qids)), key=lambda i: (diff[i], qids[i]))[:n]
    a0r, a4r = (read_trec(ROOT / "runs" / run / f"{s}.trec") for run, s in ((RR["fiqa"], "A0R"), (A4R["fiqa"], "A4R")))
    lines = ["# H3: the 20 FiQA dev questions where A0R loses most to A4R", "",
             "Relevant = listed in the FiQA judgments. \"not judged relevant\" is not the same as not relevant: the judgments list only relevant documents.", ""]
    for rank, i in enumerate(order, 1):
        q = qids[i]
        lines += [f"## {rank}. question {q}: {queries[q]}", f"A0R nDCG@10 {r['A0R'][i]:.3f}, A4R {r['A4R'][i]:.3f}, difference {diff[i]:+.3f}; judged relevant documents: {len(qrels[q])}", ""]
        for name, run in (("A0R", a0r), ("A4R", a4r)):
            lines.append(f"**{name} top 3**")
            for pos, (d, _) in enumerate(run[q][:3], 1):
                mark = "RELEVANT" if d in qrels[q] else "not judged relevant"
                lines.append(f"{pos}. [{mark}] {' '.join(docs[d].split())[:300]}")
            lines.append("")
    path.write_text("\n".join(lines))
    return {"file": str(path.relative_to(ROOT)), "questions": [qids[i] for i in order]}


def h4(seed: int = 7) -> dict:
    _, queries, _ = data.load("fiqa", "dev")
    r = _rows("fiqa")
    qids = r["qids"]
    label = {q: "advice" if any(p in queries[q].lower().replace("’", "'") for p in ADVICE) else "factual" for q in qids}
    sample = random.Random(seed).sample(qids, 30)
    out: dict = {"spot_check": [{"qid": q, "label": label[q], "question": queries[q]} for q in sample], "groups": {}}
    diff = r["A0R"] - r["A4R"]
    for g in ("advice", "factual"):
        m = np.array([label[q] == g for q in qids])
        out["groups"][g] = {"n": int(m.sum()), "mean_diff": float(diff[m].mean()), "test": _test(r["A4R"][m], r["A0R"][m])}
    a, f = (np.array([label[q] == g for q in qids]) for g in ("advice", "factual"))
    out["advice_minus_factual_gap"] = float(diff[a].mean() - diff[f].mean())
    return out


def run_all() -> dict:
    out: dict = {"H1": {}, "H2": {}}
    for ds in ("scifact", "nfcorpus", "fiqa"):
        h = h1(ds)
        groups_a0 = h.pop("groups_A0")
        out["H1"][ds] = h
        out["H2"][ds] = h2(ds, groups_a0)
    out["H3"] = h3()
    out["H4"] = h4()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "h1-h4.json").write_text(json.dumps(out, indent=2, default=float))
    return out


def h1_recall50() -> dict:
    """Added after the H1 result (JOURNAL.md): in group (a), Recall@50 over all relevant documents, A0 against A4."""
    out = {}
    for ds in ("scifact", "nfcorpus", "fiqa"):
        _, _, qrels = data.load(ds, "dev")
        runs = {s: read_trec(ROOT / "runs" / (A4R[ds] if s == "A4" else RR[ds]) / f"{s}.trec") for s in ("A0", "A4")}
        qids = sorted(runs["A0"])
        grp = [q for q in qids if group_of(top50(runs["A0"][q]), top50(runs["A4"][q]), qrels[q]) == "a"]
        rec = {s: np.array([len(set(top50(runs[s][q])) & set(qrels[q])) / len(qrels[q]) for q in grp]) for s in ("A0", "A4")}
        p = paired_difference(rec["A4"], rec["A0"])
        out[ds] = {"n": len(grp), "A0": float(rec["A0"].mean()), "A4": float(rec["A4"].mean()),
                   "A0_minus_A4": {"diff": p.mean, "lo": p.lo, "hi": p.hi}}
    return out
