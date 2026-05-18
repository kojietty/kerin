"""Safety invariant tests for stake sizing."""
from __future__ import annotations

from keirin.betting.ev import CandidatePick
from keirin.betting.stake import plan_stakes
from keirin.config import HARD_DAILY_BUDGET_YEN, HARD_EMERGENCY_STOP_YEN, BettingCfg


def _picks(n: int) -> list[CandidatePick]:
    return [CandidatePick(combo=f"{i}-{i+1}-{i+2}", prob=0.05, odds=25.0) for i in range(1, n + 1)]


def test_daily_cap_enforced():
    cfg = BettingCfg()
    plan = plan_stakes(_picks(10), cfg, month_to_date_pnl_yen=0)
    assert plan.total_yen <= HARD_DAILY_BUDGET_YEN


def test_emergency_stop_zeros_stakes():
    cfg = BettingCfg()
    plan = plan_stakes(_picks(3), cfg, month_to_date_pnl_yen=HARD_EMERGENCY_STOP_YEN)
    assert plan.emergency_stop_triggered
    assert plan.total_yen == 0
    assert all(a.stake_yen == 0 for a in plan.assignments)


def test_negative_edge_never_bets():
    cfg = BettingCfg()
    # Tiny probability × low odds = negative edge → Kelly zero
    picks = [CandidatePick(combo="1-2-3", prob=0.001, odds=2.0)]
    plan = plan_stakes(picks, cfg, month_to_date_pnl_yen=0)
    # No assignments above per_bet_min_yen
    assert plan.total_yen == 0


def test_hardcoded_caps_cannot_be_overridden():
    cfg = BettingCfg(daily_budget_yen=99999, emergency_stop_drawdown_yen=-99999)
    assert cfg.daily_budget_yen == HARD_DAILY_BUDGET_YEN
    assert cfg.emergency_stop_drawdown_yen == HARD_EMERGENCY_STOP_YEN
