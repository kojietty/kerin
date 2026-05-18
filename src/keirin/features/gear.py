"""Gear ratio features.

In keirin, riders choose their gear ratio. A higher gear requires more force
to accelerate but allows higher top speed. The main signals:

- gear_ratio itself: absolute value matters in context of the bank (short banks
  vs long banks have different optimal ratios)
- gear_z: z-score within the race field — being the *outlier* (very high or
  very low) is itself a signal
- High gear + 逃げ style: possibly overextended if the bank doesn't support it
- High gear + 捲り style: extremely dangerous for rivals, often underpriced in odds

All are simple scalar features with no domain-specific controversy.
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine


def gear_features(engine: Engine, race_id: str) -> pd.DataFrame:
    """Gear ratio features for every entry in race_id.

    Returns DataFrame indexed by (race_id, car_no) with:
      gear_ratio, gear_z (within-race z-score), gear_rank (1=highest in race),
      gear_above_field (1 if ratio > race mean else 0).
    """
    with engine.begin() as conn:
        df = pd.read_sql(
            text(
                """
                SELECT race_id, car_no, gear_ratio
                FROM entries
                WHERE race_id = :race_id AND scratched = 0
                """
            ),
            conn,
            params={"race_id": race_id},
        )

    if df.empty:
        return df

    mean = df["gear_ratio"].mean()
    std = df["gear_ratio"].std()
    df["gear_z"] = ((df["gear_ratio"] - mean) / std) if std > 0 else 0.0
    df["gear_rank"] = df["gear_ratio"].rank(ascending=False, method="min").astype(int)
    df["gear_above_field"] = (df["gear_ratio"] > mean).astype(int)

    return df.set_index(["race_id", "car_no"])
