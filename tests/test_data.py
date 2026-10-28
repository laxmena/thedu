import re
from pathlib import Path

import pytest

from thedu import data
from thedu.paths import QRELS_DIR, ROOT

# Published BEIR sizes: documents, dev questions (our dev split), test questions.
PUBLISHED = {
    "scifact": (5183, 809, 300),
    "nfcorpus": (3633, 324, 323),
    "fiqa": (57638, 500, 648),
}


@pytest.mark.parametrize("name", data.DATASETS)
def test_counts_match_published_sizes(name):
    n_docs, n_dev, n_test = PUBLISHED[name]
    docs, queries, qrels = data.load(name, "dev")
    assert len(docs) == n_docs
    assert len(queries) == len(qrels) == n_dev
    test_file = (QRELS_DIR / name / "test.tsv").read_text().splitlines()[1:]
    assert len({ln.split("\t")[0] for ln in test_file if int(ln.split("\t")[2]) > 0}) == n_test


@pytest.mark.parametrize("name", data.DATASETS)
def test_loading_test_answers_through_load_fails(name):
    with pytest.raises(PermissionError):
        data.load(name, "test")
    with pytest.raises(PermissionError):
        data.load(name, "train")


def test_test_answers_are_not_exported():
    assert not hasattr(data, "load_test_answers")


def test_only_parity_and_test_eval_import_the_test_answers_door():
    allowed = {"parity.py", "test_eval.py", "test_answers.py"}
    offenders = []
    for py in (ROOT / "src" / "thedu").rglob("*.py"):
        if py.name not in allowed and re.search(r"test_answers|load_test_answers", py.read_text()):
            offenders.append(str(py.relative_to(ROOT)))
    assert not offenders, f"these files reach for test answers: {offenders}"


def test_dev_command_refuses_a_test_config():
    from thedu.config import RunConfig
    from thedu.pipeline import run_dev

    with pytest.raises(PermissionError):
        run_dev(RunConfig(dataset="scifact", split="test", setups=["A0"]), offline=True)
