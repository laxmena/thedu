"""The made-up names swap."""

import re

import pytest

from thedu import data
from thedu.m5 import swap
from thedu.m5.swap import Entity, Swapper

TOY = [
    Entity("United States", "country", ("America", "US", "USA", "United States", "United States of America"), "Kodatrar", "a federal republic"),
    Entity("Individual Retirement Account", "account", ("IRA", "IRAs", "Individual Retirement Account"), "Rokugur", "a retirement savings account"),
    Entity("Roth Individual Retirement Account", "account", ("Roth IRA", "Roth Individual Retirement Account"), "Kevodran", "a tax-free savings account"),
    Entity("Tesla", "company", ("Tesla", "TSLA"), "Valdrix", "an electric car maker"),
]


def test_longest_alias_first_and_case_preserved():
    sw = Swapper(TOY)
    assert sw.apply("A Roth IRA is not an IRA.") == "A Kevodran is not an ROKUGUR."
    assert sw.apply("united states of america and America") == "kodatrar and Kodatrar"


def test_acronyms_only_in_capitals():
    sw = Swapper(TOY)
    assert sw.apply("Give us the US rate") == "Give us the KODATRAR rate"


def test_possessive_ticker_and_hyphen_forms():
    sw = Swapper(TOY)
    assert sw.apply("Tesla's stock, $TSLA, US-based") == "Valdrix's stock, $VALDRIX, KODATRAR-based"


def test_whole_words_only():
    sw = Swapper(TOY)
    assert sw.apply("Teslas and Americana") == "Teslas and Americana"


def test_description_mode_and_mapping_is_invertible():
    sw = Swapper(TOY)
    assert sw.apply("Tesla", "description") == "an electric car maker"
    back = {e.codename: e.name for e in TOY}
    assert sw.apply("Tesla ").strip() in back


def test_glossary_has_one_document_per_swapped_entity():
    g = Swapper(TOY).glossary()
    assert len(g) == 4 and g["m5-glossary-valdrix"] == "Valdrix: an electric car maker"


@pytest.fixture(scope="module")
def entities():
    return swap.load_entities()


def test_levels_are_nested_and_seeded(entities):
    sets = {lv: {e.name for e in swap.level_entities(entities, lv)} for lv in swap.LEVELS}
    assert len(sets[0]) == 0 and len(sets[100]) == len(entities) == 54
    assert sets[25] < sets[50] < sets[100]
    assert sets == {lv: {e.name for e in swap.level_entities(entities, lv)} for lv in swap.LEVELS}


def test_codenames_are_unique_and_not_in_the_corpus(entities):
    docs, questions, _ = data.load("fiqa", "dev")
    words = {w for t in list(docs.values()) + list(questions.values()) for w in re.findall(r"[a-z]+", t.lower())}
    names = [e.codename.lower() for e in entities]
    assert len(set(names)) == len(names) and not set(names) & words
    assert all(e.description and e.codename for e in entities)


def test_level_zero_is_the_original_and_ids_and_qrels_are_unchanged(entities):
    docs, questions, qrels = data.load("fiqa", "dev")
    sub = dict(list(docs.items())[:300])
    for cond in "AB":
        d0, q0 = swap.build(entities, 0, cond, sub, questions)
        assert d0 == sub and q0 == questions
    d100, q100 = swap.build(entities, 100, "A", sub, questions)
    assert list(d100)[: len(sub)] == list(sub)  # same ids, same order; glossary added after
    assert len(d100) == len(sub) + 54
    assert all(d.startswith("m5-glossary-") for d in list(d100)[len(sub):])
    assert not set(qrels) - set(questions)


def test_condition_b_questions_hold_no_codename(entities):
    _, questions, _ = data.load("fiqa", "dev")
    aff = swap.affected_questions(entities, questions)
    _, qa = swap.build(entities, 100, "A", {}, {q: questions[q] for q in aff})
    _, qb = swap.build(entities, 100, "B", {}, {q: questions[q] for q in aff})
    codes = {e.codename.lower() for e in entities}
    assert all(not any(c in qb[q].lower() for c in codes) for q in aff)
    assert all(any(c in qa[q].lower() for c in codes) for q in aff)


def test_audit_counts_leftover_derived_forms():
    a = swap.audit(TOY, 100, {"d": "Tesla is American. Tesla again."}, {})
    assert (a["loose_before"], a["loose_after"]) == (3, 1)  # "American" is not swapped: a demonym leaks the name
    assert a["leftover"] == {"American": 1}


def test_demonym_maps_to_a_derived_form_and_never_the_bare_codename():
    ents = swap.load_entities()
    india = next(e for e in ents if e.name == "India")
    sw = Swapper([india])
    c = india.codename
    assert sw.apply("An Indian and two Indians in India") == f"An {c}ian and two {c}ians in {c}"
    assert all(suffix for e in ents for _, suffix in e.forms)


def test_word_boundaries_indiana_is_not_india():
    ents = swap.load_entities()
    sw = Swapper([next(e for e in ents if e.name == "India")])
    assert sw.apply("Indiana, indiatimes.com and Indiana University") == "Indiana, indiatimes.com and Indiana University"


def test_plurals_map_to_codename_plus_s(entities=None):
    ents = swap.load_entities()
    ira = next(e for e in ents if e.name == "Individual Retirement Account")
    sw = Swapper([ira])
    assert sw.apply("two IRAs, one IRA") == f"two {ira.codename}s, one {ira.codename.upper()}"  # acronym case kept as it was


def test_amendment_overrides_the_frozen_bare_mapping():
    ents = swap.load_entities()
    us = next(e for e in ents if e.name == "United States")
    assert "Americans" not in us.aliases and ("Americans", "ians") in us.forms


def test_a_bare_codename_form_is_refused(tmp_path):
    bad = tmp_path / "a.csv"
    bad.write_text("entity,alias,suffix,kind\nIndia,Indian,,demonym\n")
    with pytest.raises(ValueError):
        swap.load_entities(amendment=bad)


def test_exact_case_alias_leaves_the_ordinary_phrase_alone():
    ents = swap.load_entities()
    tp = next(e for e in ents if e.name == "Trading Places")
    sw = Swapper([tp])
    assert sw.apply("the movie Trading Places, or just trading places with someone") == \
        f"the movie {tp.codename}, or just trading places with someone"


def test_britain_is_an_alias_of_the_uk_and_english_is_not_swapped():
    uk = next(e for e in swap.load_entities() if e.name == "United Kingdom")
    sw = Swapper([uk])
    assert sw.apply("In Britain they speak English") == f"In {uk.codename} they speak English"
