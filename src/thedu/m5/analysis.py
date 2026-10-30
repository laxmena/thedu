"""`thedu m5 analyze`: the pre-registered M5 tests from finished runs (docs/m5_codenames.md). Reads dev runs; calls nothing."""

from __future__ import annotations

import json

import numpy as np
import yaml

from thedu import data
from thedu.analysis.steps import OUT, load_rows
from thedu.judge import paired_difference
from thedu.m5 import swap
from thedu.paths import runs_dir

FREE = ["A0", "A3", "A0R", "A3R", "A0C"]  # the embedding-free arms
LEVELS = (0, 25, 50, 100)


def find_runs() -> dict[tuple[int, str], str]:
    """(level, condition) -> run folder name; level 0 serves both conditions."""
    found: dict[tuple[int, str], str] = {}
    for cfg in sorted(runs_dir().glob("*/config.yaml")):
        c = yaml.safe_load(cfg.read_text())
        if c.get("m5") and (cfg.parent / "metrics.json").exists():
            found[(c["m5"]["level"], c["m5"]["condition"])] = cfg.parent.name
    if (0, "A") in found:
        found.setdefault((0, "B"), found[(0, "A")])
    return found


def verdict(p, margin: float = -0.01) -> str:
    """Pre-registered labels for arm minus benchmark: beat, match, the benchmark is better, or inconclusive."""
    if p.lo > 0:
        return "beats"
    if p.lo > margin:
        return "matches"
    return "benchmark better" if p.hi < 0 else "inconclusive"


def _diff(a, b) -> dict:
    p = paired_difference(a, b)
    return {"diff": p.mean, "lo": p.lo, "hi": p.hi, "wins": p.wins, "losses": p.losses}


def analyze() -> dict:
    runs = find_runs()
    rows = {k: load_rows(v) for k, v in runs.items()}
    qids = sorted(next(iter(rows.values()))["A4"])
    nd = {k: {s: np.array([r[s][q]["ndcg10"] for q in qids]) for s in r} for k, r in rows.items()}
    ents = swap.load_entities()
    docs, questions, qrels = data.load("fiqa", "dev")
    types = {e.name: e.type for e in ents}
    full = swap.Swapper(ents)
    kind = {}
    for q in qids:
        ks = {"o" if types[n] in swap.ORG_TYPES else "p" for n in full.entities_in(questions[q])}
        kind[q] = "orgs_products" if ks == {"o"} else "places_currencies" if ks == {"p"} else "both"
    in_passage = {q: any(full.entities_in(docs[d]) for d in qrels[q] if d in docs) for q in qids}
    out: dict = {"runs": {f"{k[0]}{k[1]}": v for k, v in runs.items()}, "n_questions": len(qids), "ndcg10": {}, "primary": {}, "secondary": {}}
    for (lv, cond), arms in sorted(nd.items()):
        out["ndcg10"][f"L{lv}{cond}"] = {s: float(v.mean()) for s, v in arms.items()}

    def gap(cond, lv, arm, bench, idx=None):
        a = nd[(lv, cond)]
        x = a[bench] if idx is None else a[bench][idx]
        y = a[arm] if idx is None else a[arm][idx]
        return x, y

    def tests(cond, idx=None, lv=100) -> dict:
        t = {}
        for arm in FREE:
            a, b = gap(cond, lv, arm, "A4", idx)
            t[f"{arm} vs A4"] = _diff(a, b) | {"verdict": verdict(paired_difference(a, b))}
        for arm in ("A0R", "A3R"):
            a, b = gap(cond, lv, arm, "A4R", idx)
            t[f"{arm} vs A4R"] = _diff(a, b) | {"verdict": verdict(paired_difference(a, b))}
        return t

    for cond in "AB":
        if (100, cond) not in nd:
            continue
        t = tests(cond)
        p1 = [k for k in t if k.endswith("vs A4") and t[k]["verdict"] in ("matches", "beats")]
        out["primary"][cond] = {"tests": t, "P1_any_free_arm_matches_or_beats_A4": p1,
                                "P2_A0R_vs_A4R": t["A0R vs A4R"]["verdict"], "P3_A3R_vs_A4R": t["A3R vs A4R"]["verdict"],
                                "any_of_P1_P2_P3": bool(p1) or t["A0R vs A4R"]["verdict"] in ("matches", "beats") or t["A3R vs A4R"]["verdict"] in ("matches", "beats")}
    out["expectation_holds"] = all(out["primary"].get(c, {}).get("any_of_P1_P2_P3", False) for c in "AB")
    # secondary
    sec = out["secondary"]
    for cond in "AB":
        if (100, cond) in nd and (0, cond) in nd:
            a4 = nd[(100, cond)]["A4"] - nd[(0, cond)]["A4"]
            p = paired_difference(np.zeros_like(a4), a4)
            sec[f"S1_A4_fall_{cond}"] = {"diff": p.mean, "lo": p.lo, "hi": p.hi, "upper_below_zero": bool(p.hi < 0)}
            sec[f"S2_gap_rise_{cond}"] = {}
            for arm in FREE:
                g0 = nd[(0, cond)][arm] - nd[(0, cond)]["A4"]
                g1 = nd[(100, cond)][arm] - nd[(100, cond)]["A4"]
                p = paired_difference(g0, g1)
                sec[f"S2_gap_rise_{cond}"][arm] = {"diff": p.mean, "lo": p.lo, "hi": p.hi, "lower_above_zero": bool(p.lo > 0)}
        if (100, cond) in nd:
            for name, groups in (("S3", kind), ("S4", {q: ("passage_has_entity" if in_passage[q] else "passage_has_none") for q in qids})):
                sec[f"{name}_{cond}"] = {}
                for g in sorted(set(groups.values())):
                    idx = np.array([i for i, q in enumerate(qids) if groups[q] == g])
                    sec[f"{name}_{cond}"][g] = {"n": int(len(idx)), "tests": tests(cond, idx) if len(idx) >= 5 else "fewer than 5 questions"}
        first = {}
        for arm in FREE:
            for lv in LEVELS:
                if (lv, cond) in nd and verdict(paired_difference(nd[(lv, cond)]["A4"], nd[(lv, cond)][arm])) in ("matches", "beats"):
                    first[arm] = lv
                    break
        sec[f"lowest_level_matching_A4_{cond}"] = first
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "m5-results.json").write_text(json.dumps(out, indent=2, default=float))
    return out
