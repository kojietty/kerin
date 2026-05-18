"""Player form features computed from player_race_log.

Each function takes a SQLAlchemy engine + a reference race date and returns
a DataFrame indexed by player_id. These become per-entry features in the
training frame.

All computations are SQL-side for speed; they avoid loading full history
into memory.
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine


def _q(engine: Engine, sql: str, **params) -> pd.DataFrame:
    with engine.begin() as conn:
        return pd.read_sql(text(sql), conn, params=params)


# ---------------------------------------------------------------------------
# Rolling finish stats
# ---------------------------------------------------------------------------

def rolling_finish_stats(engine: Engine, ref_date: str, n_races: int = 10) -> pd.DataFrame:
    """Per-player average/top3/top2/win rates over the N races before ref_date.

    Returns DataFrame with columns:
      player_id, avg_finish, top3_rate, top2_rate, win_rate,
      n_races_actual (may be < n_races near start of dataset)
    """
    df = _q(
        engine,
        """
        WITH ranked AS (
          SELECT
            player_id,
            finish,
            ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY race_date DESC) AS rn
          FROM player_race_log
          WHERE race_date < :ref_date
            AND scratched = 0
            AND finish IS NOT NULL
        )
        SELECT
          player_id,
          AVG(finish)                           AS avg_finish,
          AVG(CASE WHEN finish <= 3 THEN 1.0 ELSE 0.0 END) AS top3_rate,
          AVG(CASE WHEN finish <= 2 THEN 1.0 ELSE 0.0 END) AS top2_rate,
          AVG(CASE WHEN finish  = 1 THEN 1.0 ELSE 0.0 END) AS win_rate,
          COUNT(*)                              AS n_races_actual
        FROM ranked
        WHERE rn <= :n
        GROUP BY player_id
        """,
        ref_date=ref_date, n=n_races,
    )
    return df.set_index("player_id")


def recent_trend(engine: Engine, ref_date: str, short_n: int = 5, long_n: int = 20) -> pd.DataFrame:
    """top3_rate(short) - top3_rate(long): positive = improving form."""
    short = rolling_finish_stats(engine, ref_date, n_races=short_n)[["top3_rate"]].rename(
        columns={"top3_rate": "top3_short"}
    )
    long = rolling_finish_stats(engine, ref_date, n_races=long_n)[["top3_rate"]].rename(
        columns={"top3_rate": "top3_long"}
    )
    merged = short.join(long, how="outer")
    merged["form_trend"] = merged["top3_short"].fillna(0) - merged["top3_long"].fillna(0)
    return merged


def rest_days(engine: Engine, ref_date: str) -> pd.DataFrame:
    """Days since last race (before ref_date). Capped at 90."""
    df = _q(
        engine,
        """
        SELECT player_id, MAX(race_date) AS last_race_date
        FROM player_race_log
        WHERE race_date < :ref_date
        GROUP BY player_id
        """,
        ref_date=ref_date,
    )
    if df.empty:
        return pd.DataFrame(columns=["player_id", "rest_days"]).set_index("player_id")
    df["rest_days"] = (
        pd.to_datetime(ref_date) - pd.to_datetime(df["last_race_date"])
    ).dt.days.clip(upper=90)
    return df[["player_id", "rest_days"]].set_index("player_id")


# ---------------------------------------------------------------------------
# Kimarite (決まり手) distribution
# ---------------------------------------------------------------------------

def kimarite_dist(engine: Engine, ref_date: str, n_races: int = 30) -> pd.DataFrame:
    """Fraction of wins by kimarite type over last N races.

    Returns columns: player_id, km_nige (逃げ), km_maki (捲り), km_sashi (差し),
                     km_mark (マーク). Always sums to ≤ 1.0 (some finishes have no kimarite).
    """
    df = _q(
        engine,
        """
        WITH ranked AS (
          SELECT player_id, kimarite,
                 ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY race_date DESC) AS rn
          FROM player_race_log
          WHERE race_date < :ref_date
            AND finish = 1
            AND kimarite IS NOT NULL
        )
        SELECT
          player_id,
          AVG(CASE WHEN kimarite = '逃' THEN 1.0 ELSE 0.0 END) AS km_nige,
          AVG(CASE WHEN kimarite = '捲' THEN 1.0 ELSE 0.0 END) AS km_maki,
          AVG(CASE WHEN kimarite = '差' THEN 1.0 ELSE 0.0 END) AS km_sashi,
          AVG(CASE WHEN kimarite = 'マーク' THEN 1.0 ELSE 0.0 END) AS km_mark,
          COUNT(*) AS km_n
        FROM ranked
        WHERE rn <= :n
        GROUP BY player_id
        """,
        ref_date=ref_date, n=n_races,
    )
    return df.set_index("player_id")


def scratch_rate(engine: Engine, ref_date: str, n_races: int = 30) -> pd.DataFrame:
    """欠車率 (fraction of races scratched) over last N races."""
    df = _q(
        engine,
        """
        WITH ranked AS (
          SELECT player_id, scratched,
                 ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY race_date DESC) AS rn
          FROM player_race_log
          WHERE race_date < :ref_date
        )
        SELECT
          player_id,
          AVG(CAST(scratched AS REAL)) AS scratch_rate,
          COUNT(*) AS sr_n
        FROM ranked
        WHERE rn <= :n
        GROUP BY player_id
        """,
        ref_date=ref_date, n=n_races,
    )
    return df.set_index("player_id")
