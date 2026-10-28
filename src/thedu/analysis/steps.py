"""The four analysis steps. They read finished runs under runs/ and the dev answers; nothing here
retrieves, calls an LLM, or reads test answers. Results go to runs/analysis/."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from thedu import data
from thedu.analysis.features import (POST_HOC, PRE_SEARCH, SCALE_FREE, CollectionStats, post_hoc_features,
                                     pre_search_features)
from thedu.analysis.labels import LABELS, label
from thedu.analysis.stats import (apply_stump, bucket, condition, cross_validated_gain, fit_stump, group_effect,
                                  mean_interval, tercile_thresholds)
from thedu.judge import paired_difference
from thedu.paths import ROOT
from thedu.registry import code_version

OUT = ROOT / "runs" / "analysis"
BINARY = {"has_code"}


def _run_dir(run: str) -> Path:
    return ROOT / "runs" / run


def load_rows(run: str) -> dict[str, dict[str, dict]]:
    rows: dict[str, dict[str, dict]] = {}
    for line in (_run_dir(run) / "per_question.jsonl").read_text().splitlines():
        r = json.loads(line)
        rows.setdefault(r["setup"], {})[r["qid"]] = r
    return rows


def _a0_top_scores(run: str) -> dict[str, list[float]]:
    top: dict[str, list[float]] = {}
    for line in (_run_dir(run) / "A0.trec").read_text().splitlines():
        qid, _, _, rank, score, _ = line.split()
        if int(rank) <= 2:
            top.setdefault(qid, []).append(float(score))
    return top


def _save(name: str, payload: dict, inputs: dict) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps({"meta": {"inputs": inputs, **code_version()}, **payload}, indent=2))
    return path


def question_table(dataset: str, run: str, post_hoc: bool) -> list[dict]:
    """One row per question: features, the three nDCG@10 values and the label. Dev answers only."""
    docs, queries, qrels = data.load(dataset, "dev")
    rows = load_rows(run)
    stats, top = CollectionStats(docs), _a0_top_scores(run)
    table = []
    for qid in sorted(rows["A0"]):
        a3, a4, a5 = (rows[s][qid]["ndcg10"] for s in ("A3", "A4", "A5"))
        feats = pre_search_features(queries[qid], stats, top.get(qid, []))
        if post_hoc:
            feats |= post_hoc_features(queries[qid], [docs[d] for d in qrels[qid]])
        table.append({"qid": qid, "a3": a3, "a4": a4, "a5": a5, "h0": rows["H0"][qid]["ndcg10"],
                      "label": label(a3, a4), "features": feats})
    return table


def _col(table: list[dict], f: str) -> np.ndarray:
    return np.array([r["features"][f] for r in table], dtype=float)


# ---------- 1. explore (SciFact only) ----------

def explore(run: str, dataset: str = "scifact") -> dict:
    if dataset != "scifact":
        raise ValueError("exploration is SciFact only; NFCorpus is touched by `confirm`, with fixed rules")
    table = question_table(dataset, run, post_hoc=True)
    d34 = np.array([r["a3"] - r["a4"] for r in table])
    d54 = np.array([r["a5"] - r["a4"] for r in table])
    labels = np.array([r["label"] for r in table])
    out: dict = {"dataset": dataset, "n": len(table), "label_counts": dict(Counter(map(str, labels))),
                 "a3_minus_a4": mean_interval(d34), "thresholds": {}, "tables": {}}
    for f in PRE_SEARCH + POST_HOC:
        v = _col(table, f)
        th = tercile_thresholds(v) if f not in BINARY else (0.5, 0.5)
        out["thresholds"][f] = list(th)
        buckets = np.array([bucket(x, f in BINARY, th) for x in v])
        t = {}
        for b in ("low", "mid", "high", "none") if f not in BINARY else ("no", "yes"):
            m = buckets == b
            if not m.any():
                continue
            t[b] = {"n": int(m.sum()),
                    **{f"share_{lab}": float((labels[m] == lab).mean()) for lab in LABELS},
                    "a3_minus_a4": mean_interval(d34[m]), "a5_minus_a4": mean_interval(d54[m])}
        out["tables"][f] = t
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"features-{dataset}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in table))
    _save(f"explore-{dataset}.json", out, {"run": run})
    return out


# ---------- 2. stability ----------

def stability(original: str, regens: list[str]) -> dict:
    orig, o = load_rows(original), None
    regen_rows = {r: load_rows(r)["A3"] for r in regens}
    qids = sorted(set.intersection(*(set(v) for v in regen_rows.values())))
    a4 = {q: orig["A4"][q]["ndcg10"] for q in qids}
    sources = {"cached": {q: orig["A3"][q] for q in qids}, **{f"regen {r}": regen_rows[r] for r in regens}}
    labels = {k: {q: label(v[q]["ndcg10"], a4[q]) for q in qids} for k, v in sources.items()}
    out: dict = {"n": len(qids), "a4_mean": float(np.mean(list(a4.values()))), "sources": {}, "pairs": {}}
    for k, v in sources.items():
        provs = Counter(p for q in qids for p in v[q].get("providers", []) if k != "cached")
        out["sources"][k] = {"a3_mean": float(np.mean([v[q]["ndcg10"] for q in qids])),
                             "label_counts": dict(Counter(labels[k].values())),
                             "providers": dict(provs) if k != "cached" else "unknown (spike cache)"}
    names = list(sources)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            same_label = [labels[a][q] == labels[b][q] for q in qids]
            same_text = [sources[a][q]["queries"] == sources[b][q]["queries"] for q in qids]
            rw_a = [q for q in qids if labels[a][q] == "rewrite-wins"]
            pair = {"same_label": sum(same_label), "same_rewrite_text": sum(same_text),
                    "mean_abs_ndcg_change": float(np.mean([abs(sources[a][q]["ndcg10"] - sources[b][q]["ndcg10"]) for q in qids])),
                    "rewrite_wins_in_first": len(rw_a),
                    "of_those_still_rewrite_wins": sum(labels[b][q] == "rewrite-wins" for q in rw_a)}
            pa, pb = ({q: sources[x][q].get("providers", [None]) for q in qids} for x in (a, b))
            if "cached" not in (a, b):
                diff_text = [q for q in qids if not same_text[qids.index(q)]]
                pair["rewrites_that_differ"] = len(diff_text)
                pair["of_those_same_provider"] = sum(pa[q] == pb[q] for q in diff_text)
                pair["of_those_different_provider"] = sum(pa[q] != pb[q] for q in diff_text)
                pair["same_provider_overall"] = sum(pa[q] == pb[q] for q in qids)
                same_p = [q for q in qids if pa[q] == pb[q]]
                diff_p = [q for q in qids if pa[q] != pb[q]]
                rate = lambda g: None if not g else sum(not same_text[qids.index(q)] for q in g) / len(g)  # noqa: E731
                pair["rewrite_differs_rate_same_provider"] = rate(same_p)
                pair["rewrite_differs_rate_different_provider"] = rate(diff_p)
                pair["questions_same_provider"], pair["questions_different_provider"] = len(same_p), len(diff_p)
            out["pairs"][f"{a} vs {b}"] = pair
    _save("stability.json", out, {"original": original, "regens": regens})
    return out


# ---------- 3. confirm (NFCorpus, rules frozen first) ----------

def load_frozen_rules(path: str) -> dict:
    from thedu.cli import MANIFEST, read_manifest

    p = (ROOT / path).resolve()
    rel = str(p.relative_to(ROOT))
    stored = read_manifest() if MANIFEST.exists() else {}
    if stored.get(rel) != hashlib.sha256(p.read_bytes()).hexdigest():
        raise RuntimeError(f"{rel} must be frozen and unchanged before NFCorpus is looked at "
                           f"(`thedu freeze --add {rel}`). Rules are written first, then tested.")
    return yaml.safe_load(p.read_text())


def confirm(run: str, rules_path: str, dataset: str = "nfcorpus") -> dict:
    spec = load_frozen_rules(rules_path)
    table = question_table(dataset, run, post_hoc=False)
    d34 = np.array([r["a3"] - r["a4"] for r in table])
    d54 = np.array([r["a5"] - r["a4"] for r in table])
    crit = spec["criterion"]
    out: dict = {"dataset": dataset, "n": len(table), "label_counts": dict(Counter(r["label"] for r in table)),
                 "criterion": crit, "rules": {}, "held": []}
    for rule in spec["rules"]:
        v = _col(table, rule["feature"])
        with np.errstate(invalid="ignore"):
            mask = (v <= rule["threshold"]) if rule["op"] == "<=" else (v >= rule["threshold"])
        primary = group_effect(d34, mask)
        held = bool(primary["lo"] > crit["lower_bound_above"] and primary["share"] >= crit["min_group_share"])
        out["rules"][rule["name"]] = {
            "rule": f"{rule['feature']} {rule['op']} {rule['threshold']}", "a3_minus_a4_in_vs_out": primary,
            "a5_minus_a4_in_vs_out": group_effect(d54, mask), "held": held,
            "mean_a3_minus_a4": {"in": float(d34[mask].mean()) if mask.any() else None,
                                 "out": float(d34[~mask].mean()) if (~mask).any() else None}}
        if held:
            out["held"].append({"name": rule["name"], "feature": rule["feature"]})
    _save(f"confirm-{dataset}.json", out, {"run": run, "rules": rules_path,
                                          "rules_sha256": hashlib.sha256((ROOT / rules_path).read_bytes()).hexdigest()})
    return out


# ---------- 4. route ----------

def route(scifact_run: str, nfcorpus_run: str, dataset: str = "nfcorpus") -> dict:
    confirmed = json.loads((OUT / f"confirm-{dataset}.json").read_text())
    if not confirmed["held"]:
        raise RuntimeError("no rule held on NFCorpus, so no router is fitted (see runs/analysis/confirm-nfcorpus.json)")
    allowed = sorted({h["feature"] for h in confirmed["held"]})
    assert set(allowed) <= set(SCALE_FREE) and not set(allowed) & set(POST_HOC)
    sf, nf = question_table("scifact", scifact_run, post_hoc=False), question_table(dataset, nfcorpus_run, post_hoc=False)
    sf_x = {f: _col(sf, f) for f in PRE_SEARCH}
    nf_x = {f: _col(nf, f) for f in PRE_SEARCH}
    col = lambda t, k: np.array([r[k] for r in t])  # noqa: E731
    metrics = json.loads((_run_dir(nfcorpus_run) / "metrics.json").read_text())["setups"]
    out: dict = {"allowed_features": allowed, "fit_on": "scifact", "evaluated_on": dataset, "versions": {}}
    ref = {"A4": col(nf, "a4"), "H0": col(nf, "h0"), "A5": col(nf, "a5"), "A3": col(nf, "a3")}
    out["reference_mean_ndcg10"] = {k: float(v.mean()) for k, v in ref.items()}
    out["ceiling_best_of_A4_A5"] = float(np.maximum(ref["A4"], ref["A5"]).mean())
    out["ceiling_best_of_A3_A4"] = float(np.maximum(ref["A3"], ref["A4"]).mean())
    for version, key in (("A4 vs A5 (main)", "a5"), ("A4 vs A3", "a3")):
        gain_sf = col(sf, key) - col(sf, "a4")
        stump = fit_stump(sf_x, gain_sf, allowed)
        fire_nf = apply_stump(stump, nf_x, len(nf))
        routed = np.where(fire_nf, col(nf, key), col(nf, "a4"))
        iv = mean_interval(routed)
        paid = metrics["A5" if key == "a5" else "A3"]["llm_cost_per_search_usd"]
        comps = {f"vs {k}": paired_difference(ref[k], routed) for k in (["A4", "H0", "A5"] if key == "a5" else ["A4", "A3"])}
        out["versions"][version] = {
            "stump_fit_on_scifact": stump,
            "scifact_cv_mean_gain_vs_always_free": cross_validated_gain(sf_x, gain_sf, allowed),
            "scifact_always_paid_mean_gain": float(gain_sf.mean()),
            "scifact_share_fired": float(apply_stump(stump, sf_x, len(sf)).mean()),
            "nfcorpus_ndcg10": iv, "nfcorpus_share_calling_llm": float(fire_nf.mean()),
            "llm_cost_per_search_usd": None if paid is None else float(fire_nf.mean() * paid),
            "llm_cost_is_estimate": metrics["A5" if key == "a5" else "A3"]["llm_cost_is_estimate"],
            "paired": {k: {"diff": p.mean, "lo": p.lo, "hi": p.hi, "wins": p.wins, "losses": p.losses} for k, p in comps.items()},
        }
    _save(f"route-{dataset}.json", out, {"scifact_run": scifact_run, "nfcorpus_run": nfcorpus_run})
    return out


# ---------- 5. the frozen router, evaluated (no fitting) ----------

def router_eval(run: str, rules_path: str, dataset: str) -> dict:
    spec = load_frozen_rules(rules_path)
    table = question_table(dataset, run, post_hoc=False)
    arr = {k: np.array([r[k] for r in table]) for k in ("a3", "a4", "a5", "h0")}
    fire = condition(_col(table, spec["feature"]), spec["op"], spec["threshold"])
    routed = np.where(fire, arr["a5"], arr["a4"])
    metrics = json.loads((_run_dir(run) / "metrics.json").read_text())["setups"]
    paid = metrics["A5"]["llm_cost_per_search_usd"]
    paired = {k: paired_difference(arr[key], routed) for k, key in (("A4", "a4"), ("H0", "h0"), ("A5", "a5"))}
    primary = paired["A4"]
    out = {
        "dataset": dataset, "n": len(table), "rule": f"{spec['feature']} {spec['op']} {spec['threshold']}",
        "reference_mean_ndcg10": {"A3": float(arr["a3"].mean()), "A4": float(arr["a4"].mean()),
                                  "H0": float(arr["h0"].mean()), "A5": float(arr["a5"].mean())},
        "router_ndcg10": mean_interval(routed), "share_calling_llm": float(fire.mean()),
        "n_calling_llm": int(fire.sum()),
        "llm_cost_per_search_usd": None if paid is None else float(fire.mean() * paid),
        "a5_llm_cost_per_search_usd": paid, "llm_cost_is_estimate": metrics["A5"]["llm_cost_is_estimate"],
        "paired": {k: {"diff": p.mean, "lo": p.lo, "hi": p.hi, "wins": p.wins, "losses": p.losses} for k, p in paired.items()},
        "ceiling_best_of_A4_A5": float(np.maximum(arr["a4"], arr["a5"]).mean()),
        "criterion": spec["criterion"],
        "success": bool(primary.lo > spec["criterion"]["paired_vs_free_lower_bound_above"]),
    }
    _save(f"router-frozen-{dataset}.json", out, {"run": run, "rules": rules_path,
                                                 "rules_sha256": hashlib.sha256((ROOT / rules_path).read_bytes()).hexdigest()})
    return out
