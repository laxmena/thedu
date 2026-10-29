"""`thedu m5 preview`: look at the swap before anything runs. Five affected questions in both conditions with a
relevant document around a mention, the counts, and the residual audit. Reads dev data only; calls no model."""

from __future__ import annotations

import json
import random

from thedu import data
from thedu.m5 import swap
from thedu.paths import runs_dir

PAIRS = 5


def _excerpt(text: str, sw: swap.Swapper, width: int = 110) -> tuple[str, str] | None:
    m = sw._pat.search(text) if sw._pat else None  # noqa: SLF001
    if not m:
        return None
    a, b = max(0, m.start() - width), min(len(text), m.end() + width)
    cut = text[a:b]
    return " ".join(cut.split()), " ".join(sw.apply(cut).split())


def preview(seed: int = swap.SHUFFLE_SEED) -> dict:
    ents = swap.load_entities()
    docs, questions, qrels = data.load("fiqa", "dev")
    affected = swap.affected_questions(ents, questions)
    full = swap.Swapper(ents)
    out: dict = {"entities": len(ents), "dev_questions": len(questions), "affected_questions": len(affected), "levels": {}, "pairs": []}
    for level in swap.LEVELS[1:]:
        sw = swap.Swapper(swap.level_entities(ents, level))
        changed = sum(1 for t in docs.values() if sw.apply(t) != t)
        aff_q = sum(1 for q in affected if sw.apply(questions[q]) != questions[q])
        out["levels"][level] = {"entities": len(sw.swapped), "documents_changed": changed, "glossary_documents": len(sw.glossary()),
                                "affected_questions_changed": aff_q, "audit": swap.audit(ents, level, docs, questions)}
    kinds = {"orgs_products": 0, "places_currencies": 0, "both": 0}
    by_type = {e.name: e.type for e in ents}
    for q in affected:
        ts = {("o" if by_type[n] in swap.ORG_TYPES else "p") for n in full.entities_in(questions[q])}
        kinds["both" if len(ts) == 2 else "orgs_products" if ts == {"o"} else "places_currencies"] += 1
    out["entity_split"] = kinds
    for q in random.Random(seed).sample(affected, PAIRS):
        names = sorted(full.entities_in(questions[q]))
        ex = next((e for d in qrels[q] if (e := _excerpt(docs[d], full))), None)
        ent = {e.name: e for e in ents}
        out["pairs"].append({
            "qid": q, "entities": names, "original": questions[q], "condition_A": full.apply(questions[q]),
            "condition_B": full.apply(questions[q], "description"),
            "document_excerpt": {"original": ex[0], "swapped": ex[1]} if ex else None,
            "glossary": [f"{ent[n].codename}: {ent[n].description}" for n in names]})
    path = runs_dir() / "analysis" / "m5-preview.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out
