from __future__ import annotations

from keirin.betting.ev import CandidatePick
from keirin.betting.selector import select_picks
from keirin.config import BettingCfg


def test_filters_apply():
    cfg = BettingCfg()
    cands = [
        CandidatePick("1-2-3", prob=0.10, odds=15.0),   # EV 1.50 — keep
        CandidatePick("2-3-4", prob=0.01, odds=200.0),  # odds too high — drop
        CandidatePick("3-4-5", prob=0.50, odds=2.0),    # EV 1.0, odds too low — drop
        CandidatePick("4-5-6", prob=0.05, odds=21.0),   # EV 1.05 (<1.10) — drop
    ]
    res = select_picks(cands, cfg)
    assert not res.skipped
    combos = [c.combo for c in res.picks]
    assert combos == ["1-2-3"]


def test_skip_when_no_pick():
    cfg = BettingCfg()
    cands = [CandidatePick("1-2-3", prob=0.01, odds=2.0)]
    res = select_picks(cands, cfg)
    assert res.skipped
    assert res.max_ev is not None
