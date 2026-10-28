"""Embedding search: exact (brute force) cosine similarity over normalised vectors."""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np

from thedu.paths import SPIKE_DIR, runs_dir

log = logging.getLogger(__name__)

SPIKE_MODEL = "Qwen/Qwen3-Embedding-0.6B"


class DenseRetriever:
    def __init__(self, doc_ids: list[str], doc_vecs: np.ndarray, encode_queries: Callable[[list[str]], np.ndarray]):
        if len(doc_ids) != len(doc_vecs):
            raise ValueError(f"{len(doc_ids)} documents but {len(doc_vecs)} embeddings")
        self.doc_ids = doc_ids
        self._docs = doc_vecs
        self._encode = encode_queries

    def search(self, text: str, k: int) -> list[tuple[str, float]]:
        return self.search_many([text], k)[0]

    def search_many(self, texts: list[str], k: int) -> list[list[tuple[str, float]]]:
        sims = self._encode(texts) @ self._docs.T
        top = np.argsort(-sims, axis=1)[:, : min(k, len(self.doc_ids))]
        return [[(self.doc_ids[i], float(sims[r, i])) for i in row] for r, row in enumerate(top)]

    @classmethod
    def from_model(cls, dataset: str, doc_ids: list[str], texts: list[str], model_name: str, max_seq_length: int):
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
        if cache.exists():
            vecs = np.load(cache)
        else:
            log.info("embedding %d %s documents (one time, cached)", len(texts), dataset)
            vecs = model.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=True)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, vecs)

        def encode(queries: list[str]) -> np.ndarray:
            return model.encode(queries, prompt_name="query", normalize_embeddings=True)

        return cls(doc_ids, vecs, encode)
