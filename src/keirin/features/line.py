"""Line (ライン) structure features.

A line is a group of riders working together. The lead (先行) rider does the
physical work of pacing; following riders (番手, 三番手) sit in their slipstream
and aim to sprint past near the finish. Key signals:

- Being a strong *self-starter* (逃げ・捲り) at the head of a long line is the
  strongest position: you get power AND you have allies blocking opponents.
- Being a strong rider stuck as番手 behind a weak 先行 is undervalued by the market
  — the 番手 absorbs the hard work without a reliable lead. These mismatches are a
  prime source of EV+ opportunities.
- Line length matters: a 3-person line is more stable than a 1-person solo.
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine


def line_features(engine: Engine, race_id: str) -> pd.DataFrame:
    """Compute line-based features for every entry in race_id.

    Returns DataFrame indexed by (race_id, car_no) with:
      line_id, line_pos, line_len, is_leader (pos==1), is_follower (pos>1),
      is_solo (line_len==1), leader_rating, leader_style,
      rating_vs_leader_gap (my rating - leader's rating; positive = stronger than my own lead).
    """
    with engine.begin() as conn:
        entries = pd.read_sql(
            text(
                """
                SELECT e.race_id, e.car_no, e.player_id, e.line_id, e.line_pos,
                       e.rating, e.style, e.scratched
                FROM entries e
                WHERE e.race_id = :race_id
                ORDER BY e.line_id, e.line_pos
                """
            ),
            conn,
            params={"race_id": race_id},
        )

    if entries.empty:
        return entries

    # Line size per line_id
    line_sizes = (
        entries[entries["line_id"].notna()]
        .groupby("line_id")["car_no"]
        .count()
        .rename("line_len")
    )
    entries = entries.join(line_sizes, on="line_id")

    # Leader per line (line_pos == 1)
    leaders = (
        entries[entries["line_pos"] == 1]
        .set_index("line_id")[["rating", "style"]]
        .rename(columns={"rating": "leader_rating", "style": "leader_style"})
    )
    entries = entries.join(leaders, on="line_id")

    entries["is_leader"] = (entries["line_pos"] == 1).astype(int)
    entries["is_follower"] = (entries["line_pos"] > 1).astype(int)
    entries["is_solo"] = (entries["line_len"].isna() | (entries["line_len"] == 1)).astype(int)
    entries["line_len"] = entries["line_len"].fillna(1).astype(int)
    entries["rating_vs_leader_gap"] = entries["rating"] - entries["leader_rating"].fillna(entries["rating"])

    # has_line distinguishes "this race has 並び data" from "everyone is solo".
    # Without it, the 97.5% of historical rows lacking line data look identical
    # to genuine solo riders, so LightGBM never learns to use the line columns.
    entries["has_line"] = int(entries["line_id"].notna().any())

    return entries.set_index(["race_id", "car_no"])


def mismatch_score(entries_df: pd.DataFrame) -> pd.Series:
    """「実力番手」検出 — strong player in follower position behind a weaker leader.

    Positive score = potential undervaluation by the market:
    - High positive: good rider stuck behind weak lead  → often underpriced
    - Zero / negative: follower correctly priced or not applicable

    Formula: is_follower × max(0, rating_vs_leader_gap).
    A large positive value is a soft signal to investigate further.
    """
    return (
        entries_df["is_follower"]
        * entries_df["rating_vs_leader_gap"].clip(lower=0)
    )
