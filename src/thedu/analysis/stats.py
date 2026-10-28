from __future__ import annotations

import numpy as np

from thedu.judge import bootstrap_interval

N_BOOT = 2000


def tercile_thresholds(values: np.ndarray) -> tuple[float, float]:
    v = values[~np.isnan(values)]
    lo, hi = np.quantile(v, [1 / 3, 2 / 3])
    return float(lo), float(hi)


def bucket(value: float, feature_is_binary: bool, thresholds: tuple[float, float]) -> str:
    if np.isnan(value):
        return "none"
    if feature_is_binary:
        return "yes" if value >= 0.5 else "no"
    lo, hi = thresholds
    return "low" if value <= lo else "mid" if value <= hi else "high"


def group_effect(d: np.ndarray, mask: np.ndarray, seed: int = 0) -> dict:
    """Mean of d inside the group minus mean of d outside it, with a 95% bootstrap range over questions."""
    mask = np.asarray(mask, dtype=bool)
    n = len(d)
    point = d[mask].mean() - d[~mask].mean() if mask.any() and (~mask).any() else float("nan")
    idx = np.random.default_rng(seed).integers(0, n, size=(N_BOOT, n))
    dm, mm = d[idx], mask[idx]
    g_n, r_n = mm.sum(1), (~mm).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        stat = (dm * mm).sum(1) / g_n - (dm * ~mm).sum(1) / r_n
    stat = stat[(g_n > 0) & (r_n > 0)]
    lo, hi = (float(np.percentile(stat, 2.5)), float(np.percentile(stat, 97.5))) if len(stat) else (float("nan"),) * 2
    return {"n_group": int(mask.sum()), "share": float(mask.mean()), "effect": float(point), "lo": lo, "hi": hi}


def mean_interval(d: np.ndarray) -> dict | None:
    if len(d) < 2:
        return None
    iv = bootstrap_interval(d)
    return {"mean": iv.mean, "lo": iv.lo, "hi": iv.hi}


# ---------- the router: a one-feature threshold ("decision stump") ----------

def condition(values: np.ndarray, op: str, threshold: float) -> np.ndarray:
    """True where the rule fires. A missing feature (NaN) never fires."""
    with np.errstate(invalid="ignore"):
        return (values <= threshold) if op == "<=" else (values >= threshold)


def fit_stump(features: dict[str, np.ndarray], gain: np.ndarray, allowed: list[str]) -> dict:
    """Pick the (feature, op, threshold) whose firing questions have the largest total gain.

    gain = nDCG@10 of the paid option minus nDCG@10 of the free one, per question. The router pays for the
    option on the questions where the rule fires. Fit on the training questions only.
    """
    best = {"feature": None, "op": None, "threshold": None, "total_gain": 0.0}  # the "never pay" router
    for f in allowed:
        v = features[f]
        valid = v[~np.isnan(v)]
        if len(valid) == 0:
            continue
        for t in np.unique(np.quantile(valid, np.linspace(0.05, 0.95, 19))):
            for op in ("<=", ">="):
                total = float(gain[condition(v, op, float(t))].sum())
                if total > best["total_gain"] + 1e-12:
                    best = {"feature": f, "op": op, "threshold": float(t), "total_gain": total}
    return best


def apply_stump(stump: dict, features: dict[str, np.ndarray], n: int) -> np.ndarray:
    if stump["feature"] is None:
        return np.zeros(n, dtype=bool)
    return condition(features[stump["feature"]], stump["op"], stump["threshold"])


def cross_validated_gain(features: dict[str, np.ndarray], gain: np.ndarray, allowed: list[str],
                         folds: int = 5, seed: int = 0) -> float:
    """Mean per-question gain of the stump when each question is routed by a stump fit without it."""
    n = len(gain)
    order = np.random.default_rng(seed).permutation(n)
    routed = np.zeros(n)
    for k in range(folds):
        test = order[k::folds]
        train = np.setdiff1d(order, test)
        stump = fit_stump({f: v[train] for f, v in features.items()}, gain[train], allowed)
        fire = apply_stump(stump, {f: v[test] for f, v in features.items()}, len(test))
        routed[test] = np.where(fire, gain[test], 0.0)
    return float(routed.mean())
