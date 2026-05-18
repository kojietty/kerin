"""P&L context builder for dashboard."""
from __future__ import annotations


def build_pnl_context(engine, ym: str) -> dict:
    from keirin.db.repository import month_pnl
    return month_pnl(engine, ym)
