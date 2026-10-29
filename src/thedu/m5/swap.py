"""The made-up names swap (docs/m5_codenames.md): replace real names in FiQA with invented codenames.

One mapping from the frozen check list (configs/m5_entities.csv, keep=y). Every alias of an entity maps to the entity's
codename. Matching is whole-word and case-preserving, the longest alias first; acronyms (US, UK, IRA) match only in
capitals so that "us" stays a pronoun. A possessive or a `$TICKER` keeps its apostrophe or dollar sign because only the
name itself is replaced. Levels are nested prefixes of one seeded shuffle of the entities.

Condition A puts the codename in questions and documents; condition B puts the plain description in the questions
and the codename in the documents. At level 0 nothing changes and there is no glossary.
"""

from __future__ import annotations

import csv
import random
import re
from dataclasses import dataclass
from pathlib import Path

from thedu.paths import ROOT

ENTITIES = ROOT / "configs" / "m5_entities.csv"
LEVELS = (0, 25, 50, 100)
SHUFFLE_SEED = 7
ORG_TYPES = {"company", "agency", "exchange", "fund", "index", "account", "website", "product", "person", "other"}
PLACE_TYPES = {"country", "region", "place", "city", "currency"}


@dataclass(frozen=True)
class Entity:
    name: str
    type: str
    aliases: tuple[str, ...]
    codename: str
    description: str


def load_entities(path: Path = ENTITIES) -> list[Entity]:
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["keep"].strip().lower() == "y"]
    return [Entity(r["entity"], r["type"], tuple(a for a in r["aliases"].split("|") if a), r["codename"], r["description"]) for r in rows]


def is_acronym(alias: str) -> bool:
    return alias.isupper() and " " not in alias and len(alias) <= 5


def level_entities(entities: list[Entity], level: int, seed: int = SHUFFLE_SEED) -> list[Entity]:
    """The entities swapped at a level: a prefix of one seeded shuffle, so 25 is inside 50 is inside 100."""
    if level not in LEVELS:
        raise ValueError(f"level {level} not in {LEVELS}")
    order = sorted(entities, key=lambda e: e.name)
    random.Random(seed).shuffle(order)
    return order[: round(len(order) * level / 100)]


def _case(codename: str, matched: str) -> str:
    if matched.islower():
        return codename.lower()
    if matched.isupper() and len(matched) > 1:
        return codename.upper()
    return codename


class Swapper:
    def __init__(self, swapped: list[Entity]):
        self.swapped = swapped
        self._acr: dict[str, Entity] = {}
        self._word: dict[str, Entity] = {}
        alts: list[tuple[str, str]] = []
        for e in swapped:
            for a in set(e.aliases) | {e.name}:
                if is_acronym(a):
                    self._acr[a] = e
                    alts.append((a, re.escape(a)))
                else:
                    self._word[a.lower()] = e
                    alts.append((a, f"(?i:{re.escape(a)})"))
        alts.sort(key=lambda x: (-len(x[0]), x[0]))  # longest alias first
        self._pat = re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(p for _, p in alts) + r")(?![A-Za-z0-9])") if alts else None

    def _entity(self, matched: str) -> Entity:
        return self._acr.get(matched) or self._word[matched.lower()]

    def apply(self, text: str, mode: str = "codename") -> str:
        """mode "codename": the invented name; mode "description": the plain description from the check list."""
        if self._pat is None:
            return text
        if mode == "codename":
            return self._pat.sub(lambda m: _case(self._entity(m.group(0)).codename, m.group(0)), text)
        if mode == "description":
            return self._pat.sub(lambda m: self._entity(m.group(0)).description, text)
        raise ValueError(mode)

    def entities_in(self, text: str) -> set[str]:
        return {self._entity(m.group(0)).name for m in self._pat.finditer(text)} if self._pat else set()

    def glossary(self) -> dict[str, str]:
        """One short document per swapped codename; relevant to no question."""
        return {f"m5-glossary-{e.codename.lower()}": f"{e.codename}: {e.description}" for e in self.swapped}


def affected_questions(entities: list[Entity], questions: dict[str, str]) -> list[str]:
    """Questions that contain at least one entity of the full list (the evaluation set, fixed across levels)."""
    sw = Swapper(entities)
    return sorted(q for q, t in questions.items() if sw.entities_in(t))


def build(entities: list[Entity], level: int, condition: str, docs: dict[str, str], questions: dict[str, str],
          ) -> tuple[dict[str, str], dict[str, str]]:
    """(documents, questions) at a level and condition. Document ids never change; glossary documents are added."""
    if condition not in ("A", "B"):
        raise ValueError(condition)
    sw = Swapper(level_entities(entities, level))
    new_docs = {d: sw.apply(t) for d, t in docs.items()}
    new_docs.update(sw.glossary())
    new_q = {q: sw.apply(t, "codename" if condition == "A" else "description") for q, t in questions.items()}
    return new_docs, new_q


def audit(entities: list[Entity], level: int, docs: dict[str, str], questions: dict[str, str], top: int = 12) -> dict:
    """Original names left after the swap, counted with a looser pattern than the swap itself uses: an alias at the
    start of a word followed by any letters (so American, Indians, Europeans count), any case; acronyms of up to
    three letters only as whole words in capitals. `share_left` is left / found-before over documents and questions
    of condition A, glossary excluded. `leftover` lists the commonest leftover forms."""
    swapped = level_entities(entities, level)
    if not swapped:
        return {"level": level, "loose_before": 0, "loose_after": 0, "share_left": 0.0, "leftover": {}}
    alts = []
    for e in swapped:
        for a in set(e.aliases) | {e.name}:
            alts.append(rf"(?<![A-Za-z0-9]){re.escape(a)}(?![A-Za-z0-9])" if len(a) <= 3 and a.isupper()
                        else rf"(?<![A-Za-z0-9])(?i:{re.escape(a)})[A-Za-z]*")
    pat = re.compile("|".join(sorted(alts, key=len, reverse=True)))
    sw = Swapper(swapped)
    texts = list(docs.values()) + list(questions.values())
    before = sum(len(pat.findall(t)) for t in texts)
    left: dict[str, int] = {}
    for t in texts:
        for m in pat.findall(sw.apply(t)):
            left[m] = left.get(m, 0) + 1
    after = sum(left.values())
    common = dict(sorted(left.items(), key=lambda kv: -kv[1])[:top])
    return {"level": level, "loose_before": before, "loose_after": after, "share_left": after / before if before else 0.0, "leftover": common}
