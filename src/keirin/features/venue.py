"""Venue (バンク) specific player features.

The key insight: some players systematically outperform at specific venues
(e.g. short 333m banks favour explosive starts; 500m banks favour long-sprint
specialists). If the market hasn't priced in venue affinity, we get EV > 1.
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine


def _q(engine: Engine, sql: str, **params) -> pd.DataFrame:
    with engine.begin() as conn:
        return pd.read_sql(text(sql), conn, params=params)


def venue_stats(
    engine: Engine,
    venue_id: str,
    ref_date: str,
    lookback_days: int = 730,
) -> pd.DataFrame:
    """Per-player stats at a specific venue over the past N days.

    Returns: player_id, venue_top3_rate, venue_win_rate, venue_n_races, venue_id.
    """
    cutoff = _date_minus(ref_date, lookback_days)
    df = _q(
        engine,
        """
        SELECT
          player_id,
          :venue_id AS venue_id,
          AVG(CASE WHEN finish <= 3 AND scratched = 0 THEN 1.0 ELSE 0.0 END) AS venue_top3_rate,
          AVG(CASE WHEN finish  = 1 AND scratched = 0 THEN 1.0 ELSE 0.0 END) AS venue_win_rate,
          SUM(CASE WHEN scratched = 0 THEN 1 ELSE 0 END)                     AS venue_n_races
        FROM player_race_log
        WHERE venue_id  = :venue_id
          AND race_date >= :cutoff
          AND race_date <  :ref_date
        GROUP BY player_id
        """,
        venue_id=venue_id, ref_date=ref_date, cutoff=cutoff,
    )
    return df.set_index("player_id")


def bank_length_affinity(engine: Engine, ref_date: str, lookback_days: int = 730) -> pd.DataFrame:
    """Top3 rate split by bank length (333 / 400 / 500 m).

    Returns: player_id, top3_333, top3_400, top3_500, top3_other.
    """
    cutoff = _date_minus(ref_date, lookback_days)
    df = _q(
        engine,
        """
        SELECT
          l.player_id,
          AVG(CASE WHEN v.bank_length = 333 AND l.finish <= 3 AND l.scratched = 0 THEN 1.0
                   WHEN v.bank_length = 333 AND l.scratched = 0 THEN 0.0 END) AS top3_333,
          AVG(CASE WHEN v.bank_length = 400 AND l.finish <= 3 AND l.scratched = 0 THEN 1.0
                   WHEN v.bank_length = 400 AND l.scratched = 0 THEN 0.0 END) AS top3_400,
          AVG(CASE WHEN v.bank_length = 500 AND l.finish <= 3 AND l.scratched = 0 THEN 1.0
                   WHEN v.bank_length = 500 AND l.scratched = 0 THEN 0.0 END) AS top3_500,
          COUNT(*) AS bank_n
        FROM player_race_log l
        JOIN venues v ON l.venue_id = v.venue_id
        WHERE l.race_date >= :cutoff
          AND l.race_date <  :ref_date
        GROUP BY l.player_id
        """,
        ref_date=ref_date, cutoff=cutoff,
    )
    return df.set_index("player_id")


def venue_specialist_score(
    engine: Engine,
    venue_id: str,
    ref_date: str,
) -> pd.DataFrame:
    """Venue top3_rate minus the player's overall top3_rate — positive means venue specialist.

    This is the 「バンク適性サイン」. A strong positive value means:
    - The player excels here relative to their average form
    - The market may not have priced in this venue affinity → potential EV gap
    """
    from keirin.features.player_form import rolling_finish_stats

    vs = venue_stats(engine, venue_id, ref_date)
    overall = rolling_finish_stats(engine, ref_date, n_races=20)[["top3_rate"]]

    merged = vs.join(overall, how="left")
    merged["venue_specialist_score"] = (
        merged["venue_top3_rate"].fillna(0) - merged["top3_rate"].fillna(0)
    )
    return merged[["venue_specialist_score", "venue_top3_rate", "venue_n_races"]]


def _date_minus(iso_date: str, days: int) -> str:
    from datetime import date, timedelta
    return (date.fromisoformat(iso_date) - timedelta(days=days)).isoformat()
