"""ライン力学・展開シミュレーションのテスト."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from keirin.simulation import simulate_race
from keirin.simulation.line_strength import compute_line_strengths
from keirin.simulation.race_sim import _pl_win_top3


def _race_df(rows: list[dict]) -> pd.DataFrame:
    base = {
        "rating": 80.0, "b_count": 2, "nige_cnt": 1, "makuri_cnt": 1,
        "sashi_cnt": 1, "mark_cnt": 1, "bank_length": 400,
        "line_id": None, "line_pos": None,
    }
    return pd.DataFrame([{**base, **r} for r in rows])


def _strong_vs_weak() -> pd.DataFrame:
    """強い3車ライン (1-2-3) vs 弱い2車ライン (4-5) + 単騎2車 (6,7)."""
    return _race_df([
        {"car_no": 1, "line_id": 1, "line_pos": 1, "rating": 95, "b_count": 9, "nige_cnt": 6},
        {"car_no": 2, "line_id": 1, "line_pos": 2, "rating": 90, "sashi_cnt": 5, "mark_cnt": 4},
        {"car_no": 3, "line_id": 1, "line_pos": 3, "rating": 85},
        {"car_no": 4, "line_id": 2, "line_pos": 1, "rating": 70, "b_count": 1},
        {"car_no": 5, "line_id": 2, "line_pos": 2, "rating": 68},
        {"car_no": 6, "rating": 72},
        {"car_no": 7, "rating": 65},
    ])


def test_line_strengths_basic():
    lines = compute_line_strengths(_strong_vs_weak())
    by_len = sorted(lines, key=lambda l: -len(l.cars))
    assert len(lines) == 4                     # 3車 + 2車 + 単騎×2
    big = by_len[0]
    assert big.cars == [1, 2, 3]
    assert not big.is_solo
    weak = next(l for l in lines if l.cars == [4, 5])
    assert big.front_power > weak.front_power  # B回数・得点とも上
    solos = [l for l in lines if l.is_solo]
    assert all(l.guard_power == 0.0 for l in solos)


def test_simulate_probabilities_valid():
    sim = simulate_race(_strong_vs_weak())
    assert sim is not None
    assert abs(sum(sim.p_win.values()) - 1.0) < 1e-6
    assert abs(sum(sim.p_top3.values()) - 3.0) < 0.05
    assert all(0.0 <= v <= 1.0 for v in sim.p_win.values())
    assert all(0.0 <= v <= 1.0 for v in sim.p_top3.values())
    assert set(sim.p_win.keys()) == {1, 2, 3, 4, 5, 6, 7}


def test_strong_line_favored():
    sim = simulate_race(_strong_vs_weak())
    # 強ラインの先頭2人が弱ラインの誰よりも勝率で上
    assert sim.p_win[1] > sim.p_win[4]
    assert sim.p_win[2] > sim.p_win[5]
    # 主導権確率もライン強度を反映
    assert sim.p_front[1] > sim.p_front[4]
    # 同ライン内は p_front が共有される
    assert sim.p_front[1] == sim.p_front[2] == sim.p_front[3]


def test_deterministic():
    a = simulate_race(_strong_vs_weak())
    b = simulate_race(_strong_vs_weak())
    assert a.p_win == b.p_win                  # v0 は厳密列挙 → 完全に決定的


def test_no_lines_returns_none():
    df = _race_df([{"car_no": i, "rating": 70 + i} for i in range(1, 8)])
    assert simulate_race(df) is None


def test_single_line_only_still_runs():
    """ライン1本 + 単騎複数でも動く (lines>=2 を満たす)。"""
    df = _race_df([
        {"car_no": 1, "line_id": 1, "line_pos": 1, "rating": 90, "b_count": 5},
        {"car_no": 2, "line_id": 1, "line_pos": 2, "rating": 85},
        {"car_no": 3, "rating": 80},
        {"car_no": 4, "rating": 75},
        {"car_no": 5, "rating": 70},
    ])
    sim = simulate_race(df)
    assert sim is not None
    assert abs(sum(sim.p_win.values()) - 1.0) < 1e-6


def test_too_few_cars_returns_none():
    df = _race_df([
        {"car_no": 1, "line_id": 1, "line_pos": 1},
        {"car_no": 2, "line_id": 1, "line_pos": 2},
        {"car_no": 3},
    ])
    assert simulate_race(df) is None


def test_pl_win_top3_closed_form():
    """PL閉形式 P(top3) をブルートフォース列挙と比較。"""
    import itertools
    w = np.array([4.0, 3.0, 2.0, 1.0, 0.5])
    p1, pt3 = _pl_win_top3(w)
    assert abs(p1.sum() - 1.0) < 1e-9
    assert abs(pt3.sum() - 3.0) < 1e-9

    n = len(w)
    brute = np.zeros(n)
    for perm in itertools.permutations(range(n), 3):
        a, b, c = perm
        W = w.sum()
        p = (w[a] / W) * (w[b] / (W - w[a])) * (w[c] / (W - w[a] - w[b]))
        for i in perm:
            brute[i] += p
    assert np.allclose(pt3, brute, atol=1e-9)
