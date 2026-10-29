"""Data part: load(name, split) -> Docs, Queries, Qrels. Dev only; test answers have their own guarded module."""

from thedu.data.loader import (
    DATASETS, Docs, Qrels, Queries, ensure_downloaded, load, read_docs, read_queries, read_titled_docs, select_qids,
)

__all__ = ["DATASETS", "Docs", "Qrels", "Queries", "ensure_downloaded", "load", "read_docs", "read_queries", "read_titled_docs", "select_qids"]
