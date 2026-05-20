"""Head-to-head matchup stats and grade-specific form features.

h2h_stats(): For each player in the upcoming race, find past races where
  they shared the track with at least one other player from THIS race field.
  This captures race-specific competitive dynamics that per-player form misses:
  a player whose wins come against weaker fields looks different from one who
  beats the same strong opponents repeatedly.

grade_form(): Per-player top-3 rate at the specific grade of this race.
  Some players are grade specialists (excel at F2 but underperform at G3).
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine


def h2h_stats(
    engine: Engine,
    player_ids: list[str],
    ref_date: str,
    *,
    n_races: int = 30,
) -> pd.DataFrame:
    """Win/top-3 rates in races against this specific field.

    For each player_id in player_ids, finds up to n_races most-recent races
    before ref_date where at least one other player from player_ids was also
    entered (not scratched). Computes:
      - h2h_win_rate:       fraction where finish == 1
      - h2h_top3_rate:      fraction where finish <= 3
      - h2h_n_encounters:   qualifying race count (data reliability)

    Returns DataFrame indexed by player_id.
    Players with no shared-field history return NaN rows (handled by LightGBM).
    """
    if not player_ids:
        return pd.DataFrame(
            columns=["h2h_win_rate", "h2h_top3_rate", "h2h_n_encounters"]
        )

    params: dict = {"ref_date": ref_date, "n": n_races}
    ph = ",".join(f":p{i}" for i in range(len(player_ids)))
    for i, pid in enumerate(player_ids):
        params[f"p{i}"] = pid

    sql = f"""
    WITH candidate_races AS (
      SELECT
        a.player_id,
        a.finish,
        ROW_NUMBER() OVER (
          PARTITION BY a.player_id
          ORDER BY a.race_date DESC
        ) AS rn
      FROM player_race_log a
      WHERE a.player_id IN ({ph})
        AND a.race_date < :ref_date
        AND a.scratched = 0
        AND a.finish IS NOT NULL
        AND EXISTS (
            SELECT 1 FROM player_race_log b
            WHERE b.race_id = a.race_id
              AND b.player_id IN ({ph})
              AND b.player_id != a.player_id
              AND b.scratched = 0
        )
    )
    SELECT
      player_id,
      AVG(CASE WHEN finish = 1  THEN 1.0 ELSE 0.0 END) AS h2h_win_rate,
      AVG(CASE WHEN finish <= 3 THEN 1.0 ELSE 0.0 END) AS h2h_top3_rate,
      COUNT(*) AS h2h_n_encounters
    FROM candidate_races
    WHERE rn <= :n
    GROUP BY player_id
    """

    with engine.begin() as conn:
        df = pd.read_sql(text(sql), conn, params=params)

    if df.empty:
        return pd.DataFrame(
            columns=["h2h_win_rate", "h2h_top3_rate", "h2h_n_encounters"]
        )

    return df.set_index("player_id")


def grade_form(
    engine: Engine,
    player_ids: list[str],
    ref_date: str,
    grade: str | None,
    *,
    n_races: int = 20,
) -> pd.DataFrame:
    """Top-3 rate at this specific race grade over the last N races at that grade.

    Returns DataFrame indexed by player_id with column: grade_top3_rate.
    If grade is None, returns an empty DataFrame (features become NaN via join).
    """
    if not player_ids or not grade:
        return pd.DataFrame(columns=["grade_top3_rate"])

    params: dict = {"ref_date": ref_date, "grade": grade, "n": n_races}
    ph = ",".join(f":p{i}" for i in range(len(player_ids)))
    for i, pid in enumerate(player_ids):
        params[f"p{i}"] = pid

    sql = f"""
    WITH ranked AS (
      SELECT
        player_id,
        finish,
        ROW_NUMBER() OVER (
          PARTITION BY player_id
          ORDER BY race_date DESC
        ) AS rn
      FROM player_race_log
      WHERE player_id IN ({ph})
        AND race_date < :ref_date
        AND grade = :grade
        AND scratched = 0
        AND finish IS NOT NULL
    )
    SELECT
      player_id,
      AVG(CASE WHEN finish <= 3 THEN 1.0 ELSE 0.0 END) AS grade_top3_rate
    FROM ranked
    WHERE rn <= :n
    GROUP BY player_id
    """

    with engine.begin() as conn:
        df = pd.read_sql(text(sql), conn, params=params)

    if df.empty:
        return pd.DataFrame(columns=["grade_top3_rate"])

    return df[["player_id", "grade_top3_rate"]].set_index("player_id")
