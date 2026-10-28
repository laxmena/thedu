"""Spike: does one LLM rewrite help BM25 at all? Throwaway code, no framework.

Compares three setups on SciFact dev questions (the BEIR "train" split):
  A0  BM25 (bm25s, Anserini-style k1=0.9, b=0.4)
  A4  embeddings (Qwen3-Embedding-0.6B), exact search
  A2  one LLM rewrite appended to the question, then BM25

Run from the repo root:
  uv run python scripts/spike.py                      # 100 questions, OpenRouter
  uv run python scripts/spike.py --n 300 --skip-dense # more questions, no embeddings
  uv run python scripts/spike.py --base-url http://localhost:11434/v1 --model qwen3:4b
                                                      # local model via Ollama

The LLM can be any OpenAI-compatible endpoint. By default it uses OpenRouter with
OPENROUTER_API_KEY from .env. Set THEDU_MODEL in .env to change the default model.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import bm25s
import numpy as np
import Stemmer
from dotenv import load_dotenv

DATA = Path("data")
OUT = Path("runs/spike")
SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"

PROMPT = """You help a keyword search engine (BM25) find scientific abstracts.
Rewrite the claim below as search keywords: its important terms, plus synonyms and
related technical terms that an abstract supporting or refuting it would likely use.
Output only keywords separated by spaces, at most 30 words.

Claim: {q}
/no_think"""

STEMMER = Stemmer.Stemmer("english")


# ---------- data ----------

def load_scifact() -> tuple[list[str], list[str], dict[str, str], dict[str, dict[str, int]]]:
    root = DATA / "scifact"
    if not root.exists():
        DATA.mkdir(exist_ok=True)
        zip_path = DATA / "scifact.zip"
        print("Downloading SciFact from the BEIR mirror...")
        urllib.request.urlretrieve(SCIFACT_URL, zip_path)
        zipfile.ZipFile(zip_path).extractall(DATA)

    with open(root / "corpus.jsonl") as f:
        corpus = [json.loads(line) for line in f]
    doc_ids = [str(d["_id"]) for d in corpus]
    texts = [f"{d.get('title', '')} {d['text']}".strip() for d in corpus]

    with open(root / "queries.jsonl") as f:
        questions = {str(q["_id"]): q["text"] for q in map(json.loads, f)}

    qrels: dict[str, dict[str, int]] = {}
    with open(root / "qrels" / "train.tsv") as f:  # dev questions only; never test
        next(f)
        for line in f:
            qid, did, score = line.rstrip("\n").split("\t")
            if int(score) > 0:
                qrels.setdefault(qid, {})[did] = int(score)
    return doc_ids, texts, questions, qrels


# ---------- metric ----------

def ndcg_at_10(ranked: list[str], rels: dict[str, int]) -> float:
    dcg = sum(rels.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:10]))
    ideal = sorted(rels.values(), reverse=True)[:10]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def paired_bootstrap(diffs: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = [rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---------- retrievers ----------

def bm25_index(texts: list[str]) -> bm25s.BM25:
    tokens = bm25s.tokenize(texts, stopwords="en", stemmer=STEMMER, show_progress=False)
    retriever = bm25s.BM25(k1=0.9, b=0.4)
    retriever.index(tokens, show_progress=False)
    return retriever


def bm25_search(retriever: bm25s.BM25, doc_ids: list[str], queries: list[str], k: int = 10) -> list[list[str]]:
    tokens = bm25s.tokenize(queries, stopwords="en", stemmer=STEMMER, show_progress=False)
    results, _ = retriever.retrieve(tokens, k=k, show_progress=False)
    return [[doc_ids[i] for i in row] for row in results]


def dense_search(texts: list[str], doc_ids: list[str], queries: list[str], k: int = 10) -> list[list[str]]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    model.max_seq_length = 512
    cache = OUT / "doc_embeddings.npy"
    if cache.exists():
        docs = np.load(cache)
    else:
        print("Embedding documents (one time, cached)...")
        docs = model.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=True)
        np.save(cache, docs)
    qs = model.encode(queries, prompt_name="query", normalize_embeddings=True)
    top = np.argsort(-(qs @ docs.T), axis=1)[:, :k]
    return [[doc_ids[i] for i in row] for row in top]


# ---------- rewriter ----------

def rewrite_all(questions: list[str], model: str, base_url: str, key_env: str) -> tuple[list[str], int, int]:
    from openai import OpenAI

    cache_path = OUT / "rewrites.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    client = OpenAI(base_url=base_url, api_key=os.environ.get(key_env, "not-needed"))

    def one(q: str) -> tuple[str, int, bool]:
        key = f"{model}::{q}"
        if key in cache:
            return cache[key], 0, True
        for attempt in range(3):
            try:
                r = client.chat.completions.create(
                    model=model,
                    temperature=0,
                    max_tokens=300,
                    messages=[{"role": "user", "content": PROMPT.format(q=q)}],
                )
                text = r.choices[0].message.content or ""
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
                used = r.usage.total_tokens if r.usage else 0
                return text, used, True
            except Exception as e:  # noqa: BLE001 - spike: retry anything, report at the end
                if attempt == 2:
                    print(f"  rewrite failed after 3 tries: {type(e).__name__}")
                time.sleep(2**attempt)
        return "", 0, False

    with ThreadPoolExecutor(max_workers=8) as pool:
        outs = list(pool.map(one, questions))

    for q, (text, _, ok) in zip(questions, outs):
        if ok:
            cache[f"{model}::{q}"] = text
    cache_path.write_text(json.dumps(cache, indent=1, ensure_ascii=False))

    failures = sum(1 for _, _, ok in outs if not ok)
    tokens = sum(t for _, t, _ in outs)
    return [t for t, _, _ in outs], tokens, failures


# ---------- main ----------

def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100, help="number of dev questions (max 809)")
    ap.add_argument("--model", default=os.environ.get("THEDU_MODEL", "qwen/qwen3-8b"))
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--skip-dense", action="store_true", help="skip the embedding baseline")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    doc_ids, texts, questions, qrels = load_scifact()
    qids = sorted(q for q in qrels if q in questions)
    random.Random(args.seed).shuffle(qids)
    qids = qids[: args.n]
    qs = [questions[q] for q in qids]
    print(f"SciFact: {len(texts)} docs, {len(qids)} dev questions, rewriter: {args.model}\n")

    retriever = bm25_index(texts)
    runs: dict[str, list[list[str]]] = {"A0 BM25": bm25_search(retriever, doc_ids, qs)}
    if not args.skip_dense:
        runs["A4 embeddings"] = dense_search(texts, doc_ids, qs)

    rewrites, tokens, failures = rewrite_all(qs, args.model, args.base_url, args.key_env)
    expanded = [f"{q} {w}" for q, w in zip(qs, rewrites)]
    runs["A2 rewrite + BM25"] = bm25_search(retriever, doc_ids, expanded)

    scores = {name: np.array([ndcg_at_10(r, qrels[q]) for r, q in zip(ranked, qids)]) for name, ranked in runs.items()}

    print(f"{'Setup':<20}{'nDCG@10':>9}")
    for name, s in scores.items():
        print(f"{name:<20}{s.mean():>9.4f}")

    print("\nPaired difference vs A0 BM25 (same questions, 95% bootstrap):")
    summary = {"n": len(qids), "model": args.model, "tokens": tokens, "failures": failures,
               "ndcg10": {k: float(v.mean()) for k, v in scores.items()}, "vs_bm25": {}}
    for name, s in scores.items():
        if name == "A0 BM25":
            continue
        d = s - scores["A0 BM25"]
        lo, hi = paired_bootstrap(d)
        wins, losses = int((d > 0).sum()), int((d < 0).sum())
        summary["vs_bm25"][name] = {"mean": float(d.mean()), "ci95": [lo, hi], "wins": wins, "losses": losses}
        print(f"  {name:<20}{d.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]  wins {wins}, losses {losses}")

    print(f"\nRewriter: {tokens} tokens this run, {failures} failed calls")
    print("\nFive sample rewrites:")
    for q, w in list(zip(qs, rewrites))[:5]:
        print(f"  Q: {q}\n  R: {w}\n")

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved {OUT / 'summary.json'} and {OUT / 'rewrites.json'}")


if __name__ == "__main__":
    main()
