"""IndicQARetrieval, Tamil (mteb/IndicQARetrieval, configs ta-corpus, ta-queries, ta-qrels, split test, CC0).

The dataset is one pool of 253 passages and 1,803 questions. We split it by passage (seed 7): every question goes
with its relevant passage, so questions about one passage never sit on both sides. The corpus stays whole in both
splits (as in BEIR); only questions and judgments are divided. The split is configs/indicqa_ta_split.json (frozen);
`ensure()` writes data/qrels/indicqa_ta/{dev,test}.tsv from it. Passages and questions are NFC-normalised when read.

`fetch()` needs pyarrow, which is not a project dependency (it is only used once to turn parquet into jsonl):
run `uv run --with pyarrow thedu data-fetch indicqa_ta`.
"""

from __future__ import annotations

import json
import os
import random
import statistics
import unicodedata
from pathlib import Path

from thedu.paths import CONFIGS_DIR, DATA_DIR, QRELS_DIR

NAME = "indicqa_ta"
REPO = "mteb/IndicQARetrieval"
# The revision MTEB evaluates (mteb/tasks/retrieval/multilingual/indic_qa_retrieval.py, read 2026-10-28).
REVISION = "51e8b328988795d658f6f34acd34044e9346e2ee"
ROOT = DATA_DIR / NAME
SPLIT_FILE = CONFIGS_DIR / "indicqa_ta_split.json"
SPLIT_SEED = 7
EXPECTED = {"passages": 253, "questions": 1803, "judgments": 1804}
ZWJ, ZWNJ = "‍", "‌"


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def fetch() -> None:
    """Download the Tamil parquet files at the pinned revision and write them as jsonl and tsv, text verbatim."""
    import pyarrow.parquet as pq  # not a project dependency; see the module docstring
    from huggingface_hub import hf_hub_download

    raw = ROOT / "raw"
    tables = {k: pq.read_table(hf_hub_download(REPO, f"ta-{k}/test-00000-of-00001.parquet", repo_type="dataset",
                                               revision=REVISION, local_dir=raw)).to_pylist()
              for k in ("corpus", "queries", "qrels")}
    (ROOT / "qrels").mkdir(parents=True, exist_ok=True)
    with open(ROOT / "corpus.jsonl", "w") as f:
        f.writelines(json.dumps({"_id": r["_id"], "title": r.get("title") or "", "text": r["text"]}, ensure_ascii=False) + "\n"
                     for r in tables["corpus"])
    with open(ROOT / "queries.jsonl", "w") as f:
        f.writelines(json.dumps({"_id": r["_id"], "text": r["text"]}, ensure_ascii=False) + "\n" for r in tables["queries"])
    with open(ROOT / "qrels" / "all.tsv", "w") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        f.writelines(f"{r['query-id']}\t{r['corpus-id']}\t{int(r['score'])}\n" for r in tables["qrels"])
    (ROOT / "REVISION.json").write_text(json.dumps(
        {"repo": REPO, "revision": REVISION, "configs": ["ta-corpus", "ta-queries", "ta-qrels"], "split": "test", "license": "cc0-1.0"},
        indent=2) + "\n")


def _read_all_qrels() -> list[tuple[str, str, int]]:
    rows = []
    with open(ROOT / "qrels" / "all.tsv") as f:
        next(f)
        for line in f:
            q, d, s = line.rstrip("\n").split("\t")
            rows.append((q, d, int(s)))
    return rows


def _read_jsonl(name: str) -> list[dict]:
    with open(ROOT / name) as f:
        return [json.loads(line) for line in f]


def make_split(passage_ids: list[str], seed: int = SPLIT_SEED) -> dict[str, list[str]]:
    """Half the passages to dev (rounded down: 126 of 253), the rest to test; seeded shuffle of the sorted ids."""
    ids = sorted(passage_ids)
    random.Random(seed).shuffle(ids)
    cut = len(ids) // 2
    return {"dev": sorted(ids[:cut]), "test": sorted(ids[cut:])}


def write_split() -> dict:
    """Write configs/indicqa_ta_split.json once (refuses to overwrite)."""
    if SPLIT_FILE.exists():
        raise FileExistsError(f"{SPLIT_FILE} exists; a frozen split is never rewritten")
    split = make_split([d["_id"] for d in _read_jsonl("corpus.jsonl")])
    qrels = _read_all_qrels()
    sp = {"dev_passages": split["dev"], "test_passages": split["test"]}
    skip = straddlers(qrels, sp)
    parts = split_qrels(qrels, sp | {"excluded_questions": skip})
    doc = {
        "dataset": NAME, "source": f"{REPO}@{REVISION}", "seed": SPLIT_SEED,
        "rule": "sorted passage ids, random.Random(seed).shuffle, first floor(n/2) to dev, the rest to test; "
                "a question goes with its relevant passage; a question whose passages fall on both sides is in neither split; "
                "the corpus is shared",
        "passages": {"dev": len(split["dev"]), "test": len(split["test"])},
        "questions": {s: len({q for q, _, _ in rows}) for s, rows in parts.items()},
        "excluded_questions": skip,
        "dev_passages": split["dev"], "test_passages": split["test"],
    }
    SPLIT_FILE.write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def read_split() -> dict:
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(f"{SPLIT_FILE} is missing; create it once with `thedu data-split`")
    return json.loads(SPLIT_FILE.read_text())


def straddlers(qrels: list[tuple[str, str, int]], split: dict) -> list[str]:
    """Questions whose relevant passages fall on both sides of the split (they belong to neither)."""
    side = {p: "dev" for p in split["dev_passages"]} | {p: "test" for p in split["test_passages"]}
    sides: dict[str, set[str]] = {}
    for q, d, s in qrels:
        if s > 0:
            sides.setdefault(q, set()).add(side[d])
    return sorted(q for q, v in sides.items() if len(v) > 1)


def split_qrels(qrels: list[tuple[str, str, int]], split: dict) -> dict[str, list[tuple[str, str, int]]]:
    """Judgments by side, decided by the passage. Questions listed in split["excluded_questions"] are in neither."""
    side = {p: "dev" for p in split["dev_passages"]} | {p: "test" for p in split["test_passages"]}
    skip = set(split.get("excluded_questions", []))
    out: dict[str, list] = {"dev": [], "test": []}
    for q, d, s in qrels:
        if d not in side:
            raise ValueError(f"judgment points at passage {d} that is not in the corpus")
        if q not in skip:
            out[side[d]].append((q, d, s))
    if set(straddlers(qrels, split)) - skip:
        raise ValueError("a question has relevant passages on both sides of the split and is not excluded")
    return out


def ensure() -> None:
    """Make data/qrels/indicqa_ta/{dev,test}.tsv from the split file (read-only once written)."""
    if not (ROOT / "corpus.jsonl").exists():
        raise FileNotFoundError("IndicQA Tamil is not downloaded; run `uv run --with pyarrow thedu data-fetch indicqa_ta`")
    out = QRELS_DIR / NAME
    if all((out / f"{s}.tsv").exists() for s in ("dev", "test")):
        return
    parts = split_qrels(_read_all_qrels(), read_split())
    out.mkdir(parents=True, exist_ok=True)
    for s, rows in parts.items():
        dst = out / f"{s}.tsv"
        if not dst.exists():
            dst.write_text("query-id\tcorpus-id\tscore\n" + "".join(f"{q}\t{d}\t{sc}\n" for q, d, sc in rows))
            os.chmod(dst, 0o444)


def _spread(values: list[int | float]) -> dict:
    return {"min": min(values), "median": statistics.median(values), "mean": round(statistics.fmean(values), 1), "max": max(values)}


def _joiners(texts: list[str]) -> dict:
    return {"zwj_chars": sum(t.count(ZWJ) for t in texts), "zwnj_chars": sum(t.count(ZWNJ) for t in texts),
            "texts_with_joiner": sum(1 for t in texts if ZWJ in t or ZWNJ in t),
            "texts_changed_by_nfc": sum(1 for t in texts if nfc(t) != t)}


def stats(counters: dict | None = None) -> dict:
    """Descriptive counts of the whole pool (before the split; no model, nothing tuned on them).
    `counters` maps a tokenizer name to `f(list[str]) -> list[int]` (token counts per passage)."""
    corpus, queries, qrels = _read_jsonl("corpus.jsonl"), _read_jsonl("queries.jsonl"), _read_all_qrels()
    pas = {d["_id"]: f"{d.get('title', '')} {d['text']}".strip() for d in corpus}
    qs = {q["_id"]: q["text"] for q in queries}
    rel = [(q, d) for q, d, s in qrels if s > 0]
    per_passage: dict[str, int] = {}
    for q, d in {(q, d) for q, d in rel if d in pas}:
        per_passage[d] = per_passage.get(d, 0) + 1
    counts = {"passages": len(pas), "questions": len(qs), "judgments": len(qrels)}
    out: dict = {
        "source": f"{REPO}@{REVISION}", "counts": counts, "expected": EXPECTED, "counts_match_expected": counts == EXPECTED,
        "questions_per_passage": _spread([per_passage.get(p, 0) for p in pas]) | {"passages_with_no_question": sum(1 for p in pas if p not in per_passage)},
        "passage_chars": _spread([len(nfc(t)) for t in pas.values()]),
        "question_chars": _spread([len(nfc(t)) for t in qs.values()]),
        "questions_with_missing_passage": sorted({q for q, d in rel if d not in pas}),
        "judgments_with_unknown_question": sorted({q for q, _ in rel if q not in qs}),
        "questions_without_judgment": sorted(q for q in qs if q not in {q2 for q2, _ in rel}),
        "questions_with_several_passages": sorted(q for q in qs if len({d for q2, d in rel if q2 == q}) > 1),
        "joiners_passages": _joiners(list(pas.values())), "joiners_questions": _joiners(list(qs.values())),
    }
    for name, count in (counters or {}).items():
        n = count([nfc(t) for t in pas.values()])
        out[f"passage_tokens_{name}"] = _spread(n) | {"share_over_512": round(sum(x > 512 for x in n) / len(n), 4), "n_over_512": sum(x > 512 for x in n)}
    if SPLIT_FILE.exists():
        sp = json.loads(SPLIT_FILE.read_text())
        out["split"] = {"seed": sp["seed"], "passages": sp["passages"], "questions": sp["questions"]}
    return out
