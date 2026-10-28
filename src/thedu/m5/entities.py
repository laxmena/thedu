"""The entity check list for the made-up names test: rule-found candidates, filtered and described by the pinned
model through the gateway, codenames from a seed. The output is a CSV for a human to review before it is frozen."""

from __future__ import annotations

import csv
import random
import re
from collections import defaultdict
from pathlib import Path

from thedu import data
from thedu.config import LLMConfig, ProviderConfig, RunConfig
from thedu.llm import Gateway
from thedu.m5.candidates import find_candidates
from thedu.paths import ROOT
from thedu.pipeline import make_gateway
from thedu.prompts import load_prompt

OUT = ROOT / "configs" / "m5_entities.csv"
LEADING = re.compile(r"^(?:the|in|an|a|with|if|our|of)\s+", re.I)
BATCH = 40
CODENAME_SEED = 7
COLUMNS = ["entity", "type", "aliases", "dev_questions", "docs", "codename", "description", "keep", "note"]
ONSETS = ["b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z", "br", "dr", "gr", "tr", "kl", "pl"]
VOWELS = ["a", "e", "i", "o", "u"]
CODAS = ["", "n", "r", "x", "s", "l"]


def clean(phrase: str) -> str:
    """Strip a leading 'The/In/An...' so 'The UK' and 'UK' are one candidate."""
    return LEADING.sub("", phrase).strip()


def merge(cands: list[dict]) -> list[dict]:
    """Merge variants that differ only by a leading function word or by case; drop contractions."""
    groups: dict[str, dict] = {}
    for c in cands:
        if "'" in c["phrase"] and not c["phrase"].endswith("'s"):
            continue
        p = clean(c["phrase"])
        if len(p) < 2:
            continue
        g = groups.setdefault(p.lower(), {"phrase": p, "variants": set(), "dev_questions": 0, "docs": 0})
        g["variants"].add(p)
        g["dev_questions"] = max(g["dev_questions"], c["dev_questions"])
        g["docs"] = max(g["docs"], c["docs"])
    return sorted(groups.values(), key=lambda g: (-g["dev_questions"], -g["docs"], g["phrase"]))


def _example(phrase: str, questions: dict[str, str], docs: dict[str, str]) -> str:
    pat = re.compile(r"(?<![A-Za-z0-9])" + re.escape(phrase) + r"(?![A-Za-z0-9])", re.I)
    for text in list(questions.values()) + list(docs.values()):
        m = pat.search(text)
        if m:
            a, b = max(0, m.start() - 50), min(len(text), m.end() + 50)
            return " ".join(text[a:b].split())
    return ""


def ask(gateway: Gateway, batch: list[dict], questions: dict, docs: dict) -> dict[int, tuple[bool, str, str, str]]:
    items = "\n".join(f"{i + 1}. {g['phrase']}   (e.g. \"...{_example(g['phrase'], questions, docs)}...\")" for i, g in enumerate(batch))
    r = gateway.call(load_prompt("m5_entities").format(items=items) + "\n/no_think")
    out = {}
    for line in r.text.splitlines():
        parts = [x.strip() for x in line.split("|")]
        if len(parts) >= 5 and parts[0].rstrip(".").isdigit():
            out[int(parts[0].rstrip("."))] = (parts[1].upper().startswith("KEEP"), parts[2], parts[3].lower(), parts[4])
    return out


def codenames(n: int, taken: set[str], seed: int = CODENAME_SEED) -> list[str]:
    """Pronounceable invented words of 6 to 8 letters, none of which occurs in the corpus or the questions."""
    rng = random.Random(seed)
    out: list[str] = []
    while len(out) < n:
        w = "".join(rng.choice(ONSETS) + rng.choice(VOWELS) for _ in range(rng.choice((3, 3, 4)))) + rng.choice(CODAS)
        if 6 <= len(w) <= 8 and w not in taken and w not in out:
            out.append(w)
    return [w.capitalize() for w in out]


def count_mentions(aliases: set[str], questions: dict[str, str], docs: dict[str, str]) -> tuple[int, int]:
    """Dev questions and documents that contain any alias (whole word, any case)."""
    pat = re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(re.escape(a) for a in sorted(aliases, key=len, reverse=True)) + r")(?![A-Za-z0-9])", re.I)
    return sum(1 for t in questions.values() if pat.search(t)), sum(1 for t in docs.values() if pat.search(t))


def build(limit_entities: int = 200) -> Path:
    docs, questions, _ = data.load("fiqa", "dev")
    merged = merge(find_candidates(docs, questions, limit=700))
    cfg = RunConfig(dataset="fiqa", setups=["A0"], llm=LLMConfig(provider=ProviderConfig(order=["DeepInfra"]), max_tokens=4000))  # room for 40 answer lines
    gateway, _ = make_gateway(cfg)
    verdict: dict[str, tuple[bool, str, str, str]] = {}
    for i in range(0, len(merged), BATCH):
        batch = merged[i:i + BATCH]
        got = ask(gateway, batch, questions, docs)
        for j, g in enumerate(batch, start=1):
            verdict[g["phrase"]] = got.get(j, (False, g["phrase"], "other", "UNPARSED"))
    entities: dict[str, dict] = defaultdict(lambda: {"aliases": set(), "dev_questions": 0, "docs": 0, "type": "", "description": ""})
    for g in merged:
        keep, canon, typ, desc = verdict[g["phrase"]]
        if not keep:
            continue
        e = entities[canon]
        e["aliases"] |= g["variants"] | {canon}
        e["dev_questions"] = max(e["dev_questions"], g["dev_questions"])
        e["docs"] = max(e["docs"], g["docs"])
        e["type"], e["description"] = e["type"] or typ, e["description"] or desc
    for e in entities.values():
        e["dev_questions"], e["docs"] = count_mentions(e["aliases"], questions, docs)
    ranked = sorted(entities.items(), key=lambda kv: (-kv[1]["dev_questions"], -kv[1]["docs"], kv[0]))[:limit_entities]
    taken = {w for t in list(docs.values()) + list(questions.values()) for w in re.findall(r"[a-z]+", t.lower())}
    names = codenames(len(ranked), taken)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, COLUMNS)
        w.writeheader()
        for (canon, e), code in zip(ranked, names):
            w.writerow({"entity": canon, "type": e["type"], "aliases": "|".join(sorted(e["aliases"])),
                        "dev_questions": e["dev_questions"], "docs": e["docs"], "codename": code,
                        "description": e["description"], "keep": "y", "note": ""})
    return OUT
