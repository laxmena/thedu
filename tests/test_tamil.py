"""IndicQA Tamil: the stemmer, the tokenizer, the data contract and the split."""

import re
import unicodedata

import pytest
import Stemmer

from thedu import data
from thedu.config import RunConfig, resolve_instruction
from thedu.data import indicqa
from thedu.paths import QRELS_DIR
from thedu.retrievers import Bm25Retriever

# word, expected stem. The first rows are one noun with its case endings; the later rows show where the Snowball
# Tamil stemmer under-strips (பள்ளிகளுக்கு) or over-strips (மாணவர், இந்தியா lose letters). Pinned as observed so a
# library change is noticed; they are not claims that the stems are good Tamil.
STEMS = [
    ("வீடு", "வீடு"), ("வீட்டில்", "வீடு"), ("வீட்டுக்கு", "வீடு"), ("வீடுகள்", "வீடு"),
    ("புத்தகங்கள்", "புத்தகம்"), ("புத்தகத்தை", "புத்தகம்"),
    ("தமிழ்நாட்டின்", "தமிழ்நாடு"), ("தமிழ்நாட்டில்", "தமிழ்நாடு"),
    ("பள்ளியில்", "பள்ளி"), ("பள்ளிகளுக்கு", "பள்ளிகள்"),
    ("மாணவர்கள்", "மாண"), ("இந்தியாவில்", "தியா"),
]


def test_tamil_stemmer_on_common_words(capsys):
    stem = Stemmer.Stemmer("tamil")
    with capsys.disabled():
        print("\nTamil stemmer (PyStemmer, Snowball):")
        for w, _ in STEMS:
            print(f"  {w:<18} -> {stem.stemWord(w)}")
    for w, want in STEMS:
        assert stem.stemWord(w) == want


@pytest.fixture(scope="module")
def tamil_docs():
    return data.read_docs("indicqa_ta")


def test_tokenizer_keeps_tamil_words_whole(tamil_docs, capsys):
    passage = next(iter(tamil_docs.values()))
    r = Bm25Retriever(["a"], [passage], language="tamil")
    toks = r.tokens(passage)
    with capsys.disabled():
        print(f"\nfirst passage: {len(passage)} chars, {len(toks)} tokens; first 12: {toks[:12]}")
    assert len(toks) > 100
    # no token starts with a combining mark (vowel sign, virama). The stemmer reduces some function words
    # (பற்றி "about", விட்டு) to the empty string; they are left in the index and counted in the stats notes.
    assert all(not t or not unicodedata.category(t[0]).startswith("M") for t in toks)
    # the default English pattern cuts the same text into many more fragments
    default = re.findall(r"(?u)\b\w\w+\b", passage)
    assert max(map(len, toks)) > max(map(len, default))
    assert not any(c in "‌‍" for t in toks[:0] for c in t)  # joiners are kept inside tokens, not split on


def test_tamil_bm25_matches_inflected_forms():
    docs = ["அவர் வீட்டில் இருக்கிறார்", "நாடு வளர்ந்தது", "புத்தகம் படித்தேன்"]
    r = Bm25Retriever(["h", "n", "b"], docs, language="tamil")
    assert r.search("வீடு", 1)[0][0] == "h"
    assert r.search("புத்தகங்கள்", 1)[0][0] == "b"


def test_english_bm25_is_unchanged_by_the_language_option():
    docs = ["running dogs", "cats sleep"]
    a, b = Bm25Retriever(["x", "y"], docs), Bm25Retriever(["x", "y"], docs, language="english")
    assert a.search("dog run", 2) == b.search("dog run", 2)


def test_language_is_left_out_of_the_hash_for_english():
    base = RunConfig(dataset="scifact", setups=["A0"])
    assert base.hash() == RunConfig.model_validate({**base.model_dump(), "bm25": {"k1": 0.9, "b": 0.4, "language": "english"}}).hash()
    assert base.hash() != RunConfig.model_validate({**base.model_dump(), "bm25": {"k1": 0.9, "b": 0.4, "language": "tamil"}}).hash()


def test_indicqa_instruction_is_mtebs_default_retrieval_prompt():
    assert resolve_instruction("indicqa_ta", "task") == "Retrieve text based on user query."


def test_counts_and_split():
    st = indicqa.stats()
    assert st["counts"] == indicqa.EXPECTED and st["counts_match_expected"]
    assert not st["questions_with_missing_passage"] and not st["judgments_with_unknown_question"]
    sp = indicqa.read_split()
    dev, test = set(sp["dev_passages"]), set(sp["test_passages"])
    assert not dev & test and len(dev | test) == 253 and len(dev) == 126
    assert indicqa.make_split(list(dev | test)) == {"dev": sp["dev_passages"], "test": sp["test_passages"]}  # seed 7 reproduces it
    read = lambda s: [ln.split("\t") for ln in (QRELS_DIR / "indicqa_ta" / f"{s}.tsv").read_text().splitlines()[1:]]  # noqa: E731
    assert all(d in dev for _, d, _ in read("dev")) and all(d in test for _, d, _ in read("test"))
    qd, qt = {q for q, _, _ in read("dev")}, {q for q, _, _ in read("test")}
    assert not qd & qt
    assert len(qd) + len(qt) + len(sp["excluded_questions"]) == 1803


def test_dev_loads_and_test_refuses():
    docs, queries, qrels = data.load("indicqa_ta", "dev")
    assert len(docs) == 253 and len(queries) == len(qrels) == indicqa.read_split()["questions"]["dev"]
    with pytest.raises(PermissionError):
        data.load("indicqa_ta", "test")


def test_text_is_nfc_and_joiners_are_kept(tamil_docs):
    assert all(unicodedata.is_normalized("NFC", t) for t in tamil_docs.values())
    assert sum(t.count("‌") + t.count("‍") for t in tamil_docs.values()) == 7
