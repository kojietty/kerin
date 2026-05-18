"""Smoke tests for trifecta combo expansion."""
from __future__ import annotations

import math

from keirin.models.combo import expand_trifecta


def test_plackett_luce_sums_to_one_within_eps():
    p_top3 = {1: 0.45, 2: 0.30, 3: 0.50, 4: 0.20, 5: 0.55, 6: 0.25, 7: 0.40}
    combos = expand_trifecta(p_top3, method="plackett_luce")
    # 7 cars × 6 × 5 = 210 ordered triples
    assert len(combos) == 7 * 6 * 5
    assert math.isclose(sum(combos.values()), 1.0, abs_tol=1e-6)


def test_product_sums_to_one():
    p_top3 = {1: 0.5, 2: 0.3, 3: 0.4, 4: 0.2, 5: 0.6, 6: 0.25, 7: 0.45}
    combos = expand_trifecta(p_top3, method="product")
    assert math.isclose(sum(combos.values()), 1.0, abs_tol=1e-6)


def test_higher_strength_wins():
    p_top3 = {1: 0.9, 2: 0.1, 3: 0.1}
    combos = expand_trifecta(p_top3, method="plackett_luce")
    # Combo starting with 1 should dominate
    p_1xx = sum(v for k, v in combos.items() if k.startswith("1-"))
    p_not_1xx = sum(v for k, v in combos.items() if not k.startswith("1-"))
    assert p_1xx > p_not_1xx
