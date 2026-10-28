import json

import pytest

from thedu.config import RunConfig
from thedu.registry import RunDir, VersionMismatch, code_version

CFG = RunConfig(dataset="scifact", setups=["A0"])


def test_rows_and_rankings_are_written_as_the_run_goes(tmp_path):
    run = RunDir.open(CFG, "abc", tmp_path, {})
    run.append("A0", "q1", [("d1", 2.0), ("d2", 1.0)], {"ndcg10": 1.0})
    # nothing closed or finished yet, and it is already on disk
    assert (tmp_path / "abc/per_question.jsonl").read_text().count("\n") == 1
    assert (tmp_path / "abc/A0.trec").read_text() == "q1 Q0 d1 1 2 A0\nq1 Q0 d2 2 1 A0\n"
    assert (tmp_path / "abc/config.yaml").exists()


def test_resume_skips_done_questions_and_drops_uncommitted_rankings(tmp_path):
    run = RunDir.open(CFG, "abc", tmp_path, {})
    run.append("A0", "q1", [("d1", 1.0)], {"ndcg10": 1.0})
    with open(tmp_path / "abc/A0.trec", "a") as f:  # a crash after the ranking, before the score row
        f.write("q2 Q0 d9 1 1 A0\n")
    run.close()
    again = RunDir.open(CFG, "abc", tmp_path, {})
    assert set(again.rows) == {("A0", "q1")}
    assert "q2" not in (tmp_path / "abc/A0.trec").read_text()


def test_resume_under_different_code_is_refused(tmp_path):
    RunDir.open(CFG, "abc", tmp_path, {}).close()
    meta = tmp_path / "abc/meta.json"
    m = json.loads(meta.read_text())
    m["git_commit"] = "0" * 40
    meta.write_text(json.dumps(m))
    with pytest.raises(VersionMismatch, match="mix two code versions"):
        RunDir.open(CFG, "abc", tmp_path, {})
    again = RunDir.open(CFG, "abc", tmp_path, {}, restart=True)  # sets the old run aside
    assert list(tmp_path.glob("abc.superseded-*")) and not again.rows


def test_run_records_the_code_version(tmp_path):
    RunDir.open(CFG, "abc", tmp_path, {}).close()
    m = json.loads((tmp_path / "abc/meta.json").read_text())
    assert {k: m[k] for k in ("git_commit", "code_hash")} == code_version()
