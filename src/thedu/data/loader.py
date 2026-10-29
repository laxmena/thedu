"""BEIR datasets with dev and test kept apart.

On first use a dataset is downloaded and its qrels are split into
data/qrels/<name>/dev.tsv and test.tsv (both made read-only). `load()` only ever
reads dev.tsv. Test answers are read only by the guarded module next to this one.
"""

from __future__ import annotations

import json
import logging
import os
import random
import unicodedata
import urllib.request
import zipfile
from pathlib import Path

from thedu.paths import DATA_DIR, QRELS_DIR

log = logging.getLogger(__name__)

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
DATASETS = ("scifact", "nfcorpus", "fiqa", "indicqa_ta")
# Which BEIR qrels file holds our dev questions. SciFact has no dev file; its "train" split is our dev.
DEV_FILE = {"scifact": "train", "nfcorpus": "dev", "fiqa": "dev"}

Docs = dict[str, str]
Queries = dict[str, str]
Qrels = dict[str, dict[str, int]]


def read_qrels(path: Path) -> Qrels:
    qrels: Qrels = {}
    with open(path) as f:
        next(f)  # header
        for line in f:
            qid, did, score = line.rstrip("\n").split("\t")
            if int(score) > 0:
                qrels.setdefault(qid, {})[did] = int(score)
    return qrels


def _clean(name: str, text: str) -> str:
    """IndicQA Tamil text is NFC-normalised (joiners kept); the English datasets are read as they are."""
    return unicodedata.normalize("NFC", text) if name == "indicqa_ta" else text


def ensure_downloaded(name: str) -> None:
    if name == "indicqa_ta":
        from thedu.data import indicqa

        indicqa.ensure()  # the split decides which questions are dev and which are test
        return
    root = DATA_DIR / name
    if not (root / "corpus.jsonl").exists():
        DATA_DIR.mkdir(exist_ok=True)
        zip_path = DATA_DIR / f"{name}.zip"
        log.info("downloading %s from the BEIR mirror", name)
        urllib.request.urlretrieve(BEIR_URL.format(name=name), zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(DATA_DIR)
    out = QRELS_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    for split, src in (("dev", DEV_FILE[name]), ("test", "test")):
        dst = out / f"{split}.tsv"
        if not dst.exists():
            dst.write_bytes((root / "qrels" / f"{src}.tsv").read_bytes())
            os.chmod(dst, 0o444)


def read_docs(name: str) -> Docs:
    """The corpus, with no answers attached."""
    ensure_downloaded(name)
    docs: Docs = {}
    with open(DATA_DIR / name / "corpus.jsonl") as f:
        for line in f:
            d = json.loads(line)
            docs[str(d["_id"])] = _clean(name, f"{d.get('title', '')} {d['text']}".strip())
    return docs


def read_titled_docs(name: str) -> dict[str, tuple[str, str]]:
    """The corpus as (title, body) per document, for variants that score the title as its own field."""
    ensure_downloaded(name)
    out: dict[str, tuple[str, str]] = {}
    with open(DATA_DIR / name / "corpus.jsonl") as f:
        for line in f:
            d = json.loads(line)
            out[str(d["_id"])] = (_clean(name, (d.get("title") or "").strip()), _clean(name, d["text"]))
    return out


def read_queries(name: str) -> Queries:
    """Every question text of the dataset (dev and test), with no answers attached."""
    ensure_downloaded(name)
    with open(DATA_DIR / name / "queries.jsonl") as f:
        return {str(q["_id"]): _clean(name, q["text"]) for q in map(json.loads, f)}


def load(name: str, split: str) -> tuple[Docs, Queries, Qrels]:
    """Docs, dev queries and dev qrels. Only the "dev" split can be loaded here."""
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {DATASETS}")
    if split != "dev":
        raise PermissionError(
            f"split {split!r} cannot be loaded here. Test answers are only reachable "
            "through `thedu parity` and `thedu test-eval`."
        )
    ensure_downloaded(name)
    all_queries = read_queries(name)
    qrels = read_qrels(QRELS_DIR / name / "dev.tsv")
    qids = sorted(q for q in qrels if q in all_queries)
    return read_docs(name), {q: all_queries[q] for q in qids}, {q: qrels[q] for q in qids}


def select_qids(queries: Queries, n: int | None, seed: int) -> list[str]:
    """All dev question ids (sorted), or a seeded random sample of n of them."""
    qids = sorted(queries)
    if n:
        random.Random(seed).shuffle(qids)
        qids = qids[:n]
    return qids
