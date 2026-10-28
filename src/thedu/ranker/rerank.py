"""Cross-encoder reranking: reorder the top of one candidate list by how well each document answers the question.

The candidates are the first `depth` hits with score-zero filler removed; everything below them keeps its original
order, so the set of documents (and Recall@100) never changes. The model scores (question, document) pairs.
"""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock

Hit = tuple[str, float]
Scorer = Callable[[list[tuple[str, str]]], list[float]]


class CrossEncoderReranker:
    def __init__(self, docs: dict[str, str], model: str = "BAAI/bge-reranker-v2-m3", depth: int = 50,
                 max_length: int = 512, batch_size: int = 16, scorer: Scorer | None = None,
                 count_tokens: Callable[[str], int] | None = None):
        self.docs, self.model_name, self.depth, self.max_length, self.batch_size = docs, model, depth, max_length, batch_size
        self._scorer, self._count = scorer, count_tokens
        self._lock = Lock()  # one model on one device: calls take turns

    def _load(self) -> None:
        if self._scorer is not None:
            return
        from sentence_transformers import CrossEncoder

        ce = CrossEncoder(self.model_name, max_length=self.max_length)
        tok = ce.tokenizer
        self._scorer = lambda pairs: [float(x) for x in ce.predict(pairs, batch_size=self.batch_size)]
        self._count = lambda text: len(tok(text, add_special_tokens=False)["input_ids"])

    def rerank(self, hits: list[Hit], question: str) -> tuple[list[Hit], int]:
        """Return the reordered list and the tokens billed at the reference rate:
        (question tokens x documents) + document tokens, each pair cut at max_length."""
        candidates = [h for h in hits if h[1] > 0][: self.depth]  # score-zero filler is not a candidate
        if not candidates:
            return hits, 0
        with self._lock:
            self._load()
            scores = self._scorer([(question, self.docs[d]) for d, _ in candidates])
        q = self._count(question)
        tokens = sum(q + min(self._count(self.docs[d]), max(self.max_length - q, 0)) for d, _ in candidates)
        order = sorted(range(len(candidates)), key=lambda i: -scores[i])  # stable: ties keep the original order
        head = [(candidates[i][0], scores[i]) for i in order]
        seen = {d for d, _ in head}
        tail = [h for h in hits if h[0] not in seen and h[1] > 0]  # ranks below the candidates, original order
        return head + tail, tokens

    def rank(self, lists: list[list[Hit]], question: str) -> list[Hit]:
        """The ranker contract: one list in, the reranked list out."""
        if len(lists) != 1:
            raise ValueError("the reranker takes exactly one candidate list")
        return self.rerank(lists[0], question)[0]
