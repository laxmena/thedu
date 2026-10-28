"""Step 1 of the entity list: candidate names found by rule, with no model. A language model filters them next."""

from __future__ import annotations

import re
from collections import Counter

CAP = re.compile(r"\b[A-Z][A-Za-z0-9&'-]+(?:\s+[A-Z][A-Za-z0-9&'-]+){0,3}\b")
TICKER = re.compile(r"\b[A-Z]{2,5}\b")
SENT_START = re.compile(r"(?:^|[.!?:]\s+|\n)$")


def _phrases(text: str) -> list[str]:
    """Capitalised phrases and tickers that are not just a sentence's first word."""
    out = []
    for m in list(CAP.finditer(text)) + list(TICKER.finditer(text)):
        if SENT_START.search(text[max(0, m.start() - 3): m.start()]) and " " not in m.group(0):
            continue  # a single capitalised word at a sentence start says nothing
        out.append(m.group(0))
    return out


def find_candidates(docs: dict[str, str], questions: dict[str, str], min_docs: int = 3, min_questions: int = 1,
                    limit: int = 700) -> list[dict]:
    """Phrases that occur in at least `min_questions` dev questions and `min_docs` documents, most common in the
    questions first. Single words that are usually written in lower case elsewhere (ordinary words) are dropped."""
    doc_count: Counter[str] = Counter()
    lower_words: Counter[str] = Counter()
    for text in docs.values():
        doc_count.update(set(_phrases(text)))
        lower_words.update(re.findall(r"\b[a-z][a-z0-9'-]+\b", text))
    # Questions are short and often written in lower case, so match the document-side phrases case-insensitively.
    qtext = [re.sub(r"[^a-z0-9&' -]+", " ", t.lower()) for t in questions.values()]
    q_count: Counter[str] = Counter()
    for p, dc in doc_count.items():
        if dc >= min_docs:
            pat = re.compile(r"(?<![a-z0-9])" + re.escape(p.lower()) + r"(?![a-z0-9])")
            q_count[p] = sum(1 for t in qtext if pat.search(t))
    rows = []
    for p, qc in q_count.items():
        dc = doc_count[p]
        if qc < min_questions or dc < min_docs:
            continue
        if " " not in p and p.isalpha() and lower_words[p.lower()] >= 0.3 * dc:
            continue  # an ordinary word that happened to be capitalised
        rows.append({"phrase": p, "dev_questions": qc, "docs": dc})
    rows.sort(key=lambda r: (-r["dev_questions"], -r["docs"], r["phrase"]))
    return rows[:limit]
