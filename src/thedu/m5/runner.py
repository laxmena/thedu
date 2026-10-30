"""`thedu m5 run`: one swap level and condition of the made-up names test (docs/m5_codenames.md, v4 and its amendment).

FiQA dev, the 94 questions that contain an entity, the swapped corpus (plus glossary documents) and the swapped questions.
Only changed documents are re-embedded; the others keep the vectors of the original corpus. Level 0 is the original FiQA
and the same for both conditions, so it is run once (as condition A).
"""

from __future__ import annotations

import logging
import time

import numpy as np

from thedu import data
from thedu.config import LLMConfig, M5Config, PlannerConfig, ProviderConfig, RankerConfig, RunConfig, resolve_instruction
from thedu.m5 import swap
from thedu.paths import runs_dir
from thedu.pipeline import execute
from thedu.retrievers.dense import DenseRetriever, embed_documents, query_encoder

log = logging.getLogger(__name__)

SETUPS = ["A0", "A3", "A0C", "A4", "A0R", "A3R", "A4R"]
BASE = "fiqa-Qwen_Qwen3-Embedding-0.6B-512.npy"


def run_config(level: int, condition: str) -> RunConfig:
    return RunConfig(
        dataset="fiqa", split="dev", setups=SETUPS,
        llm=LLMConfig(model="qwen/qwen3-14b", temperature=0.0, seed=7, max_tokens=300,
                      provider=ProviderConfig(order=["DeepInfra"], allow_fallbacks=False)),
        planner=PlannerConfig(grounded_prompt="grounded_v1", query_repeat=3, feedback_k=5, use_spike_cache=False),
        ranker=RankerConfig(rrf_k=60, depth=100), m5=M5Config(level=level, condition=condition))


def dense_for(cfg: RunConfig, original: dict[str, str], swapped: dict[str, str], level: int) -> DenseRetriever:
    """Vectors for the swapped corpus: unchanged documents reuse the original matrix, changed and glossary documents
    are embedded (chunked and resumable, cached per level; the two conditions share the same documents)."""
    from sentence_transformers import SentenceTransformer

    base = np.load(runs_dir() / "embeddings" / BASE)
    ids = list(swapped)
    if len(base) != len(original) or ids[: len(original)] != list(original):
        raise RuntimeError("the swapped corpus does not start with the original documents in their original order")
    changed = [i for i, d in enumerate(ids) if i >= len(original) or swapped[d] != original[d]]
    model = SentenceTransformer(cfg.dense.model)
    model.max_seq_length = cfg.dense.max_seq_length
    cache = runs_dir() / "embeddings" / f"m5-fiqa-L{level}-changed.npy"
    t0 = time.perf_counter()
    if not changed:  # level 0: the original corpus, nothing to embed
        new = np.zeros((0, base.shape[1]), dtype=base.dtype)
    else:
        new = np.load(cache) if cache.exists() else embed_documents(model, [swapped[ids[i]] for i in changed], cache, f"m5 L{level}")
    log.info("m5 level %d: %d of %d documents embedded in %.0f s", level, len(changed), len(ids), time.perf_counter() - t0)
    vecs = np.concatenate([base, np.zeros((len(ids) - len(base), base.shape[1]), dtype=base.dtype)])
    vecs[changed] = new
    return DenseRetriever(ids, vecs, query_encoder(model, resolve_instruction("fiqa", cfg.dense.query_instruction)))


def run_level(level: int, condition: str, restart: bool = False):
    if level == 0 and condition != "A":
        raise ValueError("level 0 is the same corpus for both conditions; run it as condition A")
    ents = swap.load_entities()
    docs, questions, qrels = data.load("fiqa", "dev")
    affected = swap.affected_questions(ents, questions)
    new_docs, new_q = swap.build(ents, level, condition, docs, {q: questions[q] for q in affected})
    cfg = run_config(level, condition)
    dense = dense_for(cfg, docs, new_docs, level)
    log.info("m5 level %d condition %s: %d questions, %d documents", level, condition, len(new_q), len(new_docs))
    return execute(cfg, new_docs, new_q, {q: qrels[q] for q in affected}, restart=restart, dense=dense)
