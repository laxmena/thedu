"""Spike: does LLM query rewriting help BM25? Throwaway code, no framework.

Setups (all scored on the top 10 with nDCG@10, on dev questions only):
  A0  BM25 (bm25s, Anserini-style k1=0.9, b=0.4)
  A4  embeddings (Qwen3-Embedding-0.6B), exact search
  A2  blind rewrite: the LLM sees only the question, then BM25
  A3  grounded rewrite: BM25 first, the LLM reads the top results and writes terms
      from the relevant ones, then BM25 again (repeat for --rounds)

A2 and A3 search with the question repeated --query-repeat times plus the LLM's terms,
so the added words can't outweigh the question (the query2doc lesson).

Run from the repo root:
  uv run python scripts/spike.py                                  # SciFact, 100 questions, all setups
  uv run python scripts/spike.py --dataset nfcorpus               # NFCorpus dev questions
  uv run python scripts/spike.py --n 0 --setups A0,A4,A3          # every dev question
  uv run python scripts/spike.py --prompt v1 --query-repeat 1     # reproduce the first run
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
from threading import Lock

import bm25s
import numpy as np
import Stemmer
from dotenv import load_dotenv

DATA = Path("data")
OUT = Path("runs/spike")
BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
# Which BEIR qrels file holds the dev questions. SciFact has no dev file; its "train" split is our dev.
DEV_SPLIT = {"scifact": "train", "nfcorpus": "dev"}
K = 10          # results scored per question (nDCG@10)
FEEDBACK_K = 5  # results the A3 rewriter reads per round
SNIPPET_WORDS = 60

PROMPTS = {
    # A2, v1: the first spike prompt. Kept so old runs stay reproducible.
    "v1": """You help a keyword search engine (BM25) find scientific abstracts.
Rewrite the claim below as search keywords: its important terms, plus synonyms and
related technical terms that an abstract supporting or refuting it would likely use.
Output only keywords separated by spaces, at most 30 words.

Claim: {q}
/no_think""",
    # A2, v2: fewer, sharper words. In practice it drifted into neighbouring topics.
    "v2": """You help a keyword search engine (BM25) find the scientific abstract that a claim is about.
Write at most 12 extra search terms that such an abstract would likely contain but the claim
does not: synonyms, full names for abbreviations, specific technical terms.
Rules: no generic research words (study, trial, effect, analysis, outcome, research),
no word repeated, nothing already in the claim. Output only the terms, separated by spaces.

Claim: {q}
/no_think""",
}

# A3: grounded in what the search actually returned (corpus-steered expansion).
A3_PROMPT = """You help a keyword search engine (BM25) find documents relevant to a query.
Below are the current top search results for the query. Some are relevant, some are not.

{snippets}

Query: {q}

Write up to 15 search terms that would help find documents about exactly this query.
Take terms from the results that are relevant to the query: specific names, technical terms,
abbreviations, the wording those documents use. Ignore results about a different topic.
No generic words (study, trial, effect, analysis, outcome, research, patients).
Output only the terms, separated by spaces.
/no_think"""

STEMMER = Stemmer.Stemmer("english")


# ---------- data ----------

def load_dataset(name: str) -> tuple[list[str], list[str], dict[str, str], dict[str, dict[str, int]]]:
    root = DATA / name
    if not root.exists():
        DATA.mkdir(exist_ok=True)
        zip_path = DATA / f"{name}.zip"
        print(f"Downloading {name} from the BEIR mirror...")
        urllib.request.urlretrieve(BEIR_URL.format(name=name), zip_path)
        zipfile.ZipFile(zip_path).extractall(DATA)

    with open(root / "corpus.jsonl") as f:
        corpus = [json.loads(line) for line in f]
    doc_ids = [str(d["_id"]) for d in corpus]
    texts = [f"{d.get('title', '')} {d['text']}".strip() for d in corpus]

    with open(root / "queries.jsonl") as f:
        questions = {str(q["_id"]): q["text"] for q in map(json.loads, f)}

    qrels: dict[str, dict[str, int]] = {}
    with open(root / "qrels" / f"{DEV_SPLIT[name]}.tsv") as f:  # dev questions only; never test
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


def bm25_search(retriever: bm25s.BM25, doc_ids: list[str], queries: list[str], k: int = K) -> list[list[str]]:
    tokens = bm25s.tokenize(queries, stopwords="en", stemmer=STEMMER, show_progress=False)
    results, _ = retriever.retrieve(tokens, k=k, show_progress=False)
    return [[doc_ids[i] for i in row] for row in results]


def dense_search(dataset: str, texts: list[str], doc_ids: list[str], queries: list[str], k: int = K) -> list[list[str]]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    model.max_seq_length = 512
    # SciFact keeps its original file name so the first run's 14-minute embedding is reused.
    cache = OUT / ("doc_embeddings.npy" if dataset == "scifact" else f"doc_embeddings_{dataset}.npy")
    if cache.exists():
        docs = np.load(cache)
    else:
        print(f"Embedding {len(texts)} {dataset} documents (one time, cached)...")
        docs = model.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=True)
        np.save(cache, docs)
    qs = model.encode(queries, prompt_name="query", normalize_embeddings=True)
    top = np.argsort(-(qs @ docs.T), axis=1)[:, :k]
    return [[doc_ids[i] for i in row] for row in top]


# ---------- LLM ----------

class Rewriter:
    """One OpenAI-compatible client with a disk cache. Cache keys include everything that changes the prompt."""

    def __init__(self, model: str, base_url: str, key_env: str):
        from openai import AuthenticationError, NotFoundError, OpenAI, PermissionDeniedError

        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=os.environ.get(key_env, "not-needed"))
        self.cache_path = OUT / "rewrites.json"
        self.cache: dict[str, str] = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        self.lock = Lock()
        self.tokens = 0
        self.failures = 0
        try:  # check the model and key once before sending hundreds of calls
            self.client.chat.completions.create(model=model, max_tokens=5, messages=[{"role": "user", "content": "ok"}])
        except NotFoundError:
            raise SystemExit(
                f"\nModel '{model}' was not found at {base_url}.\n"
                "List the Qwen models OpenRouter has right now with:\n"
                "  curl -s https://openrouter.ai/api/v1/models | python3 -c \"import json,sys; "
                "[print(m['id']) for m in json.load(sys.stdin)['data'] if m['id'].startswith('qwen/')]\"\n"
                "then rerun with --model <id>, or set THEDU_MODEL=<id> in .env."
            )
        except (AuthenticationError, PermissionDeniedError):
            raise SystemExit(f"\nThe API key in {key_env} was rejected. Check the key in .env and its credit limit.")

    def __call__(self, cache_key: str, prompt: str) -> tuple[str, bool]:
        key = f"{self.model}::{cache_key}"
        with self.lock:
            if key in self.cache:
                return self.cache[key], True
        for attempt in range(3):
            try:
                r = self.client.chat.completions.create(
                    model=self.model, temperature=0, max_tokens=300,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = re.sub(r"<think>.*?</think>", "", r.choices[0].message.content or "", flags=re.S).strip()
                with self.lock:
                    self.tokens += r.usage.total_tokens if r.usage else 0
                    self.cache[key] = text
                return text, True
            except Exception as e:  # noqa: BLE001 - spike: retry anything, report at the end
                if attempt == 2:
                    print(f"  rewrite failed after 3 tries: {type(e).__name__}: {str(e)[:120]}")
                time.sleep(2**attempt)
        with self.lock:
            self.failures += 1
        return "", False

    def save(self) -> None:
        self.cache_path.write_text(json.dumps(self.cache, indent=1, ensure_ascii=False))


def snippet(text: str) -> str:
    words = text.split()
    return " ".join(words[:SNIPPET_WORDS]) + (" ..." if len(words) > SNIPPET_WORDS else "")


# ---------- main ----------

def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(DEV_SPLIT), default="scifact")
    ap.add_argument("--n", type=int, default=100, help="number of dev questions; 0 = all")
    ap.add_argument("--setups", default="A0,A4,A2,A3", help="comma-separated: A0,A4,A2,A3")
    ap.add_argument("--model", default=os.environ.get("THEDU_MODEL", "qwen/qwen3-14b"))
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--prompt", choices=sorted(PROMPTS), default="v1", help="A2 prompt version")
    ap.add_argument("--query-repeat", type=int, default=3,
                    help="repeat the question N times so the added words can't outweigh it (A2 and A3)")
    ap.add_argument("--rounds", type=int, default=1, help="A3 feedback rounds")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--skip-dense", action="store_true", help="same as leaving A4 out of --setups")
    args = ap.parse_args()

    setups = [s.strip().upper() for s in args.setups.split(",") if s.strip()]
    if args.skip_dense and "A4" in setups:
        setups.remove("A4")
    if "A0" not in setups:
        setups.insert(0, "A0")  # everything is compared against BM25

    OUT.mkdir(parents=True, exist_ok=True)
    doc_ids, texts, questions, qrels = load_dataset(args.dataset)
    by_id = dict(zip(doc_ids, texts))
    qids = sorted(q for q in qrels if q in questions)
    random.Random(args.seed).shuffle(qids)
    if args.n:
        qids = qids[: args.n]
    qs = [questions[q] for q in qids]
    print(f"{args.dataset}: {len(texts)} docs, {len(qids)} dev questions | setups {','.join(setups)} | "
          f"rewriter {args.model}, A2 prompt {args.prompt}, question x{args.query_repeat}, A3 rounds {args.rounds}\n")

    retriever = bm25_index(texts)
    runs: dict[str, list[list[str]]] = {}
    examples: dict[str, list[str]] = {}

    runs["A0"] = bm25_search(retriever, doc_ids, qs)
    if "A4" in setups:
        runs["A4"] = dense_search(args.dataset, texts, doc_ids, qs)

    rewriter = None
    if "A2" in setups or "A3" in setups:
        rewriter = Rewriter(args.model, args.base_url, args.key_env)

    def with_question(q: str, terms: str) -> str:
        return " ".join([q] * args.query_repeat + [terms])

    if "A2" in setups:
        # v1 keeps its original cache key so the first run's rewrites are reused.
        def a2(q: str) -> str:
            key = q if args.prompt == "v1" else f"{args.prompt}::{q}"
            return rewriter(key, PROMPTS[args.prompt].format(q=q))[0]

        with ThreadPoolExecutor(max_workers=8) as pool:
            terms = list(pool.map(a2, qs))
        runs["A2"] = bm25_search(retriever, doc_ids, [with_question(q, t) for q, t in zip(qs, terms)])
        examples["A2"] = terms

    if "A3" in setups:
        def a3(i: int) -> tuple[list[str], list[str]]:
            q = qs[i]
            ranked = runs["A0"][i]
            history: list[str] = []
            for r in range(1, args.rounds + 1):
                shown = ranked[:FEEDBACK_K]
                snippets = "\n".join(f"[{j + 1}] {snippet(by_id[d])}" for j, d in enumerate(shown))
                key = f"a3::{args.dataset}::x{args.query_repeat}::r{r}::{'|'.join(shown)}::{q}"
                terms, _ = rewriter(key, A3_PROMPT.format(snippets=snippets, q=q))
                history.append(terms)
                ranked = bm25_search(retriever, doc_ids, [with_question(q, terms)])[0]
            return ranked, history

        with ThreadPoolExecutor(max_workers=8) as pool:
            outs = list(pool.map(a3, range(len(qs))))
        runs["A3"] = [r for r, _ in outs]
        examples["A3"] = [" | ".join(h) for _, h in outs]

    if rewriter:
        rewriter.save()
        if rewriter.failures > len(qs) * 0.05:
            raise SystemExit(f"\n{rewriter.failures} rewrites failed, so A2/A3 results would be meaningless. "
                             "Fix the cause above and rerun. Successful rewrites are cached.")

    names = {"A0": "A0 BM25", "A4": "A4 embeddings", "A2": "A2 blind rewrite", "A3": "A3 grounded rewrite"}
    scores = {s: np.array([ndcg_at_10(r, qrels[q]) for r, q in zip(runs[s], qids)]) for s in setups if s in runs}

    print(f"{'Setup':<24}{'nDCG@10':>9}")
    for s, v in scores.items():
        print(f"{names[s]:<24}{v.mean():>9.4f}")

    def compare(a: str, b: str) -> dict:
        d = scores[b] - scores[a]
        lo, hi = paired_bootstrap(d)
        return {"mean": float(d.mean()), "ci95": [lo, hi], "wins": int((d > 0).sum()), "losses": int((d < 0).sum())}

    pairs = [("A0", s) for s in scores if s != "A0"]
    if "A2" in scores and "A3" in scores:
        pairs.append(("A2", "A3"))
    if "A4" in scores and "A3" in scores:
        pairs.append(("A4", "A3"))
    print("\nPaired differences (same questions, 95% bootstrap range):")
    comparisons = {}
    for a, b in pairs:
        c = compare(a, b)
        comparisons[f"{b} vs {a}"] = c
        print(f"  {b} vs {a}: {c['mean']:+.4f}  [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]  "
              f"wins {c['wins']}, losses {c['losses']}")

    # How often each setup keeps or breaks what BM25 already got right
    a0 = scores["A0"]
    print(f"\nBM25 already had a correct answer at #1 for {(a0 >= 0.99).sum()} questions "
          f"and nothing correct in the top 10 for {(a0 == 0).sum()}.")
    for s in ("A2", "A3", "A4"):
        if s in scores:
            v = scores[s]
            print(f"  {names[s]}: pushed down {((a0 >= 0.99) & (v < 0.99)).sum()} of the first, "
                  f"rescued {((a0 == 0) & (v > 0)).sum()} of the second")

    if rewriter:
        print(f"\nRewriter: {rewriter.tokens} tokens this run, {rewriter.failures} failed calls")
    for s, terms in examples.items():
        print(f"\nThree sample {names[s]} terms:")
        for q, t in list(zip(qs, terms))[:3]:
            print(f"  Q: {q}\n  T: {t}\n")

    summary = {
        "dataset": args.dataset, "n": len(qids), "setups": setups, "model": args.model,
        "prompt": args.prompt, "query_repeat": args.query_repeat, "rounds": args.rounds, "seed": args.seed,
        "tokens": rewriter.tokens if rewriter else 0, "failures": rewriter.failures if rewriter else 0,
        "ndcg10": {names[s]: float(v.mean()) for s, v in scores.items()},
        "comparisons": comparisons,
        "per_question": {s: v.round(4).tolist() for s, v in scores.items()},
        "qids": qids,
    }
    # Every run is kept in runs/spike/history/; summary.json is always the latest run.
    history = OUT / "history"
    history.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = (f"{stamp}-{args.dataset}-{'-'.join(s for s in setups)}-{args.model.split('/')[-1]}"
            f"-{args.prompt}-q{args.query_repeat}-r{args.rounds}-n{len(qids)}.json")
    summary["time"] = stamp
    (history / name).write_text(json.dumps(summary, indent=2))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved {history / name} (also as {OUT / 'summary.json'})")


if __name__ == "__main__":
    main()
