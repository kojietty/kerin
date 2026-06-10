"""バックテスト指標のテスト."""
from __future__ import annotations

from collections import Counter

import pytest
from sqlalchemy import text

from keirin.backtest.metrics import (
    rank1_hit_rate, top3_set_hit_rate, tvd_car_bias, virtual_trifecta_roi,
)
from keirin.db.engine import get_engine, init_db


def test_rank1_hit_rate():
    preds = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    actuals = [[1, 3, 2], [5, 4, 6], [7, 9, 8]]
    # 1着一致: 1==1 ○, 4!=5 ×, 7==7 ○ → 2/3
    assert abs(rank1_hit_rate(preds, actuals) - 2 / 3) < 1e-9


def test_top3_set_hit_rate():
    preds = [[1, 2, 3, 4], [4, 5, 6]]
    actuals = [[3, 1, 2], [4, 5, 7]]
    # set{1,2,3}=={3,1,2} ○, {4,5,6}!={4,5,7} × → 1/2
    assert abs(top3_set_hit_rate(preds, actuals) - 0.5) < 1e-9


def test_metrics_empty():
    import math
    assert math.isnan(rank1_hit_rate([], []))
    assert math.isnan(top3_set_hit_rate([], []))


def test_tvd_car_bias():
    # 完全一致 → 0
    same = Counter({1: 10, 2: 10})
    assert tvd_car_bias(same, same) == 0.0
    # 完全に異なる分布 → 1.0
    p = Counter({1: 10})
    a = Counter({2: 10})
    assert abs(tvd_car_bias(p, a) - 1.0) < 1e-9


def test_virtual_trifecta_roi():
    eng = get_engine(":memory:")
    init_db(eng)
    with eng.begin() as conn:
        conn.execute(text("INSERT INTO venues VALUES ('81','小倉',400,NULL)"))
        conn.execute(text(
            "INSERT INTO races VALUES ('R1','2026-05-01','81',1,'F2',NULL,NULL,NULL,NULL,NULL)"
        ))
        conn.execute(text(
            "INSERT INTO races VALUES ('R2','2026-05-01','81',2,'F2',NULL,NULL,NULL,NULL,NULL)"
        ))
        conn.execute(text(
            "INSERT INTO payouts VALUES ('R1','trifecta','1-2-3',1500,1)"
        ))
        conn.execute(text(
            "INSERT INTO payouts VALUES ('R2','trifecta','4-5-6',800,2)"
        ))

    picks = {"R1": ["1-2-3"], "R2": ["1-2-3"]}   # R1 的中 / R2 ハズレ
    out = virtual_trifecta_roi(eng, picks, stake_yen=100)
    assert out["bets"] == 2
    assert out["stake"] == 200
    assert out["hits"] == 1
    assert out["payout"] == 1500     # 100円賭け → 払戻 1500円 (100円あたり配当)
    assert abs(out["roi"] - 7.5) < 1e-9
