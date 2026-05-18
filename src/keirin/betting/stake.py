"""Stake sizing: fractional Kelly with hard daily cap and emergency stop.

Important invariants enforced here:
  * Per-call stake never exceeds the hardcoded HARD_DAILY_BUDGET_YEN ceiling
    even if config or caller suggests otherwise.
  * If the month-to-date PnL has fallen below HARD_EMERGENCY_STOP_YEN, this
    module returns zero stake on all picks (a "paper trade only" output).
  * Per-bet stake is rounded down to the nearest 100 yen; sub-100 picks drop
    out entirely.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from keirin.betting.ev import CandidatePick
from keirin.config import HARD_DAILY_BUDGET_YEN, HARD_EMERGENCY_STOP_YEN, BettingCfg


@dataclass
class StakeAssignment:
    pick: CandidatePick
    stake_yen: int


@dataclass
class StakePlan:
    assignments: list[StakeAssignment]
    total_yen: int
    emergency_stop_triggered: bool
    reason: str | None = None


def plan_stakes(
    picks: list[CandidatePick],
    cfg: BettingCfg,
    *,
    month_to_date_pnl_yen: int,
) -> StakePlan:
    # Emergency stop overrides everything.
    if month_to_date_pnl_yen <= HARD_EMERGENCY_STOP_YEN:
        return StakePlan(
            assignments=[StakeAssignment(p, 0) for p in picks],
            total_yen=0,
            emergency_stop_triggered=True,
            reason=f"emergency stop at MTD PnL {month_to_date_pnl_yen} <= {HARD_EMERGENCY_STOP_YEN}",
        )
    if not picks:
        return StakePlan(assignments=[], total_yen=0, emergency_stop_triggered=False)

    cap = min(cfg.daily_budget_yen, HARD_DAILY_BUDGET_YEN)
    fractions = [_kelly_fraction(p, cfg.kelly_fraction) for p in picks]
    total_frac = sum(fractions) or 1.0
    target = [(cap * f / total_frac) for f in fractions]
    rounded = [_round_to_100(t) for t in target]

    # Clip to ensure sum <= cap (rounding shouldn't push us over, but be safe).
    while sum(rounded) > cap:
        idx = rounded.index(max(rounded))
        rounded[idx] = max(0, rounded[idx] - 100)

    assignments = [
        StakeAssignment(pick=p, stake_yen=s)
        for p, s in zip(picks, rounded, strict=True)
        if s >= cfg.per_bet_min_yen
    ]
    return StakePlan(
        assignments=assignments,
        total_yen=sum(a.stake_yen for a in assignments),
        emergency_stop_triggered=False,
    )


def _kelly_fraction(pick: CandidatePick, kelly_mult: float) -> float:
    """Compute fractional-Kelly bet fraction f = kelly_mult * (b*p - q) / b.

    Where b = odds - 1, p = prob, q = 1 - p. Negative or zero edges are clipped
    to 0 (we never bet at negative EV). This is then *relative*; absolute size
    is determined by the caller via the budget cap.
    """
    b = pick.odds - 1.0
    p = pick.prob
    q = 1.0 - p
    if b <= 0:
        return 0.0
    raw = (b * p - q) / b
    return max(0.0, kelly_mult * raw)


def _round_to_100(yen: float) -> int:
    return int(math.floor(yen / 100.0) * 100)
