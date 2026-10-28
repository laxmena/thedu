from __future__ import annotations

MARGIN = 0.1  # nDCG@10 difference that counts as a win
LABELS = ("rewrite-wins", "embed-wins", "tie", "both-fail")
_EPS = 1e-9


def label(a3: float, a4: float, margin: float = MARGIN) -> str:
    """rewrite-wins: A3 beats A4 by at least `margin`; embed-wins: the reverse; both-fail: both score 0; else tie."""
    if a3 == 0 and a4 == 0:
        return "both-fail"
    d = a3 - a4
    if d >= margin - _EPS:
        return "rewrite-wins"
    if d <= -margin + _EPS:
        return "embed-wins"
    return "tie"
