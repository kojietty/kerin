"""Pick selection: EV / odds-band / prob-floor filters then top-N by EV."""
from __future__ import annotations

from dataclasses import dataclass

from keirin.betting.ev import CandidatePick
from keirin.config import BettingCfg


@dataclass
class SelectionResult:
    picks: list[CandidatePick]
    skipped: bool
    max_ev: float | None
    reason: str | None = None


def select_picks(candidates: list[CandidatePick], cfg: BettingCfg) -> SelectionResult:
    if not candidates:
        return SelectionResult(picks=[], skipped=True, max_ev=None, reason="no candidates")

    max_ev = max(c.ev for c in candidates)
    filtered = [
        c for c in candidates
        if cfg.odds_min <= c.odds <= cfg.odds_max
        and c.prob >= cfg.prob_min
        and c.ev >= cfg.ev_threshold
    ]
    if not filtered:
        return SelectionResult(picks=[], skipped=True, max_ev=max_ev, reason="below EV threshold")
    filtered.sort(key=lambda c: c.ev, reverse=True)
    chosen = filtered[: cfg.max_picks_per_race]
    return SelectionResult(picks=chosen, skipped=False, max_ev=max_ev)
