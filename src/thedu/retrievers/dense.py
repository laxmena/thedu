"""Embedding search: exact (brute force) cosine similarity over normalised vectors."""

from __future__ import annotations

import logging
import math
import shutil
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from thedu.paths import SPIKE_DIR, runs_dir
from thedu.retrievers.order import tie_ranks, top_k

log = logging.getLogger(__name__)

SPIKE_MODEL = "Qwen/Qwen3-Embedding-0.6B"


def embed_documents(model, texts: list[str], cache: Path, label: str, chunk: int = 2048) -> np.ndarray:
    """Embed in chunks, each saved the moment it is done, so a crash or a closed laptop loses at most one chunk.
    Rerunning picks up from the saved chunks. The finished matrix is saved to `cache` and the chunks removed."""
    part = cache.with_suffix(".partial")
    part.mkdir(parents=True, exist_ok=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    n_chunks = math.ceil(len(texts) / chunk)
    pieces, t0, done_now = [], time.perf_counter(), 0
    for i in range(n_chunks):
        f = part / f"{i:05d}.npy"
        want = min(chunk, len(texts) - i * chunk)
        arr = np.load(f) if f.exists() else None
        if arr is None or len(arr) != want:
            arr = model.encode(texts[i * chunk : i * chunk + want], batch_size=16, normalize_embeddings=True,
                               show_progress_bar=False)
            tmp = f.with_name(f.name + ".tmp")
            with open(tmp, "wb") as fh:
                np.save(fh, arr)
            tmp.replace(f)  # a chunk file is either whole or absent
            done_now += want
            elapsed = time.perf_counter() - t0
            left = len(texts) - i * chunk - want
            log.info("%s: chunk %d/%d done (%d of %d docs); %.0f docs/s; about %.0f min left", label, i + 1, n_chunks,
                     i * chunk + want, len(texts), done_now / elapsed, left / (done_now / elapsed) / 60)
        pieces.append(arr)
    vecs = np.concatenate(pieces)
    np.save(cache, vecs)
    shutil.rmtree(part)
    return vecs


def query_encoder(model, query_instruction: str | None) -> Callable[[list[str]], np.ndarray]:
    def encode(queries: list[str]) -> np.ndarray:
        if query_instruction is None:
            return model.encode(queries, prompt_name="query", normalize_embeddings=True)
        prefix = f"Instruct: {query_instruction}\nQuery:"  # the template MTEB uses for Qwen3-Embedding
        return model.encode([prefix + q for q in queries], normalize_embeddings=True)

    return encode


class DenseRetriever:
    def __init__(self, doc_ids: list[str], doc_vecs: np.ndarray, encode_queries: Callable[[list[str]], np.ndarray]):
        if len(doc_ids) != len(doc_vecs):
            raise ValueError(f"{len(doc_ids)} documents but {len(doc_vecs)} embeddings")
        self.doc_ids = doc_ids
        self._docs = doc_vecs
        self._encode = encode_queries
        self._ranks = tie_ranks(doc_ids)

    def search(self, text: str, k: int) -> list[tuple[str, float]]:
        return self.search_many([text], k)[0]

    def search_many(self, texts: list[str], k: int) -> list[list[tuple[str, float]]]:
        sims = self._encode(texts) @ self._docs.T
        return [[(self.doc_ids[i], float(sims[r, i])) for i in top_k(sims[r], self._ranks, k)] for r in range(len(sims))]

    @classmethod
    def from_model(cls, dataset: str, doc_ids: list[str], texts: list[str], model_name: str, max_seq_length: int,
                   query_instruction: str | None = None):
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        model.max_seq_length = max_seq_length
        slug = model_name.replace("/", "_")
        cache = runs_dir() / "embeddings" / f"{dataset}-{slug}-{max_seq_length}.npy"
        # The spike's SciFact/NFCorpus embeddings (same model and length) are reused, not recomputed.
        legacy = SPIKE_DIR / ("doc_embeddings.npy" if dataset == "scifact" else f"doc_embeddings_{dataset}.npy")
        if not cache.exists() and legacy.exists() and (model_name, max_seq_length) == (SPIKE_MODEL, 512):
            cache.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(legacy, cache)
        vecs = np.load(cache) if cache.exists() else embed_documents(model, texts, cache, dataset)

        return cls(doc_ids, vecs, query_encoder(model, query_instruction))
