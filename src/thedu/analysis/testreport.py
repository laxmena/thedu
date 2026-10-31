"""The test-set report (JOURNAL.md, 2026-10-31 pre-registration): per dataset, nDCG@10 and Recall@100 per setup, the
paired tests with verdicts, and dev against test. Reads finished run folders only (per-question scores); it loads
no answers. Flags differences of more than 0.02 between dev and test and any verdict that changed."""

from __future__ import annotations

import json

import numpy as np
import yaml

from thedu.analysis.hypotheses import A4R, RR
from thedu.analysis.steps import OUT, load_rows
from thedu.judge import bootstrap_interval, paired_difference
from thedu.m5.analysis import verdict
from thedu.paths import runs_dir

SETUPS = ("A0", "A3", "A4", "A0R", "A3R", "A4R")
# (arm, base, primary?): the difference is arm minus base
TESTS = (("A0R", "A4R", True), ("A3R", "A4R", True), ("A3R", "A0R", False), ("A4R", "A4", False), ("A0R", "A4", False), ("A3", "A0", False))
FLAG = 0.02


def test_run(dataset: str) -> str:
    for cfg in sorted(runs_dir().glob("*/config.yaml")):
        meta = json.loads((cfg.parent / "meta.json").read_text())
        if meta.get("kind") == "test" and yaml.safe_load(cfg.read_text())["dataset"] == dataset and (cfg.parent / "testeval.json").exists():
            return cfg.parent.name
    raise FileNotFoundError(f"no finished test run for {dataset}")


def _scores(rows: dict, setup: str, qids: list[str], key: str = "ndcg10") -> np.ndarray:
    return np.array([rows[setup][q][key] for q in qids])


def label(p, primary: bool) -> str:
    if primary:
        return verdict(p)  # beats / matches / benchmark better / inconclusive
    return "positive" if p.lo > 0 else "negative" if p.hi < 0 else "includes zero"


def one(rows_by_setup: dict) -> dict:
    qids = sorted(rows_by_setup["A0"])
    out: dict = {"n": len(qids), "setups": {}, "tests": {}}
    for s in SETUPS:
        n, r = bootstrap_interval(_scores(rows_by_setup, s, qids)), bootstrap_interval(_scores(rows_by_setup, s, qids, "recall100"))
        out["setups"][s] = {"ndcg10": n.mean, "ndcg10_lo": n.lo, "ndcg10_hi": n.hi, "recall100": r.mean}
    for arm, base, primary in TESTS:
        p = paired_difference(_scores(rows_by_setup, base, qids), _scores(rows_by_setup, arm, qids))
        out["tests"][f"{arm} - {base}"] = {"diff": p.mean, "lo": p.lo, "hi": p.hi, "verdict": label(p, primary), "primary": primary}
    return out


def dev_rows(dataset: str) -> dict:
    rr, a4 = load_rows(RR[dataset]), load_rows(A4R[dataset])
    return {**{s: rr[s] for s in ("A0", "A3", "A4", "A0R", "A3R")}, "A4R": a4["A4R"]}


def report() -> dict:
    out: dict = {"datasets": {}, "flags": []}
    for ds in ("nfcorpus", "scifact", "fiqa"):
        run = test_run(ds)
        te = one(load_rows(run))
        dv = one(dev_rows(ds))
        meta = json.loads((runs_dir() / run / "testeval.json").read_text())
        out["datasets"][ds] = {"test_run": run, "test": te, "dev": dv, "load_start": meta["load_start"], "load_end": meta["load_end"],
                               "parity": meta["parity"]}
        for s in SETUPS:
            d = te["setups"][s]["ndcg10"] - dv["setups"][s]["ndcg10"]
            if abs(d) > FLAG:
                out["flags"].append(f"{ds} {s} nDCG@10 differs from dev by {d:+.4f} (test {te['setups'][s]['ndcg10']:.4f}, dev {dv['setups'][s]['ndcg10']:.4f})")
        for k, t in te["tests"].items():
            d = t["diff"] - dv["tests"][k]["diff"]
            if abs(d) > FLAG:
                out["flags"].append(f"{ds} {k} differs from dev by {d:+.4f} (test {t['diff']:+.4f}, dev {dv['tests'][k]['diff']:+.4f})")
            if t["verdict"] != dv["tests"][k]["verdict"]:
                out["flags"].append(f"{ds} {k}: verdict changed from '{dv['tests'][k]['verdict']}' (dev) to '{t['verdict']}' (test)")
    (OUT).mkdir(parents=True, exist_ok=True)
    (OUT / "test-report.json").write_text(json.dumps(out, indent=2))
    return out
