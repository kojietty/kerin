"""Trifecta combo probability expansion.

Given p_top3[i] (calibrated probability that car i finishes in the top 3),
expand to a probability for each ordered triple (i, j, k) with distinct cars.

Two methods are implemented:

  plackett_luce  — convert p_top3 into strength s = -log(1 - p_top3) and apply
                   the standard Plackett-Luce ordering formula.
  product        — naive normalized product of p_top3 values.

The Plackett-Luce path is the recommended default. Both share the same output
shape: a dict[str, float] keyed by combo strings like "3-5-1".
"""
from __future__ import annotations

import math
from itertools import permutations


def expand_trifecta(
    p_top3: dict[int, float],
    *,
    method: str = "plackett_luce",
) -> dict[str, float]:
    """Return mapping of "i-j-k" -> probability."""
    cars = sorted(p_top3.keys())
    probs = {c: max(min(p_top3[c], 0.999), 0.001) for c in cars}
    if method == "plackett_luce":
        strengths = {c: -math.log(1.0 - probs[c]) for c in cars}
        return _pl_expand(strengths)
    elif method == "product":
        return _product_expand(probs)
    else:
        raise ValueError(f"unknown method: {method}")


def _pl_expand(strengths: dict[int, float]) -> dict[str, float]:
    cars = list(strengths.keys())
    total = sum(strengths.values())
    out: dict[str, float] = {}
    for i, j, k in permutations(cars, 3):
        p_i = strengths[i] / total
        denom2 = total - strengths[i]
        if denom2 <= 0:
            continue
        p_j = strengths[j] / denom2
        denom3 = denom2 - strengths[j]
        if denom3 <= 0:
            continue
        p_k = strengths[k] / denom3
        out[f"{i}-{j}-{k}"] = p_i * p_j * p_k
    return out


def _product_expand(probs: dict[int, float]) -> dict[str, float]:
    cars = list(probs.keys())
    raw: dict[str, float] = {}
    for i, j, k in permutations(cars, 3):
        raw[f"{i}-{j}-{k}"] = probs[i] * probs[j] * probs[k]
    s = sum(raw.values()) or 1.0
    return {c: v / s for c, v in raw.items()}
