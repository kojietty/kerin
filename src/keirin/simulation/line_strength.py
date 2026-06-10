"""ライン単位の強度評価.

各ラインを「先行力 (front_power)」「援護力 (guard_power)」で定量化する。
z 正規化は必ずレース内で行う — 車番や絶対スケールへの依存を排除し、
「この面子の中で誰が強いか」だけを見る (思想②: 実力主義)。

- front_power: リーダー (line_pos=1) の B回数 z + 自力決まり手 (逃+捲) z + 得点 z
- guard_power: 番手以降の 差し+マーク z + 得点 z (三番手は 0.5 減衰)
- 単騎は line_len=1 のラインとして扱う (front=本人, guard=0)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# front_power の重み: B回数 / 自力決まり手 / 得点
_W_FRONT = (0.5, 0.3, 0.2)
# guard_power の重み: 差し+マーク / 得点
_W_GUARD = (0.6, 0.4)
# 三番手以降の減衰
_POS3_DECAY = 0.5


@dataclass
class LineStrength:
    line_id: int                       # 正のID (単騎は負のIDを合成)
    cars: list[int] = field(default_factory=list)   # line_pos 順
    front_power: float = 0.0
    guard_power: float = 0.0
    total_power: float = 0.0
    is_solo: bool = False


def _z(series: pd.Series) -> pd.Series:
    """レース内 z-score。観測が1つ以下/分散ゼロ/NaN は 0 扱い。"""
    s = pd.to_numeric(series, errors="coerce")
    m = s.mean()
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=series.index)
    return ((s - m) / sd).fillna(0.0)


def compute_line_strengths(race_df: pd.DataFrame) -> list[LineStrength]:
    """race_df: 1行=1車。必要列: car_no, line_id, line_pos, rating,
    b_count, nige_cnt, makuri_cnt, sashi_cnt, mark_cnt (欠損可)。

    Returns: LineStrength のリスト (単騎を含む全ライン)。
    """
    df = race_df.reset_index(drop=True).copy()

    z_b = _z(df.get("b_count", pd.Series(dtype=float)).reindex(df.index))
    z_self = _z(
        pd.to_numeric(df.get("nige_cnt"), errors="coerce").fillna(0)
        + pd.to_numeric(df.get("makuri_cnt"), errors="coerce").fillna(0)
    )
    z_chase = _z(
        pd.to_numeric(df.get("sashi_cnt"), errors="coerce").fillna(0)
        + pd.to_numeric(df.get("mark_cnt"), errors="coerce").fillna(0)
    )
    z_rating = _z(df.get("rating", pd.Series(dtype=float)).reindex(df.index))

    df["_z_front"] = _W_FRONT[0] * z_b + _W_FRONT[1] * z_self + _W_FRONT[2] * z_rating
    df["_z_guard"] = _W_GUARD[0] * z_chase + _W_GUARD[1] * z_rating
    df["_z_rating"] = z_rating

    lines: list[LineStrength] = []
    solo_id = -1

    lined = df[df["line_id"].notna()]
    for lid, grp in lined.groupby("line_id"):
        grp = grp.sort_values("line_pos")
        cars = grp["car_no"].astype(int).tolist()
        leader = grp.iloc[0]
        front = float(leader["_z_front"])
        guard = 0.0
        for k, (_, follower) in enumerate(grp.iloc[1:].iterrows()):
            w = 1.0 if k == 0 else _POS3_DECAY
            guard += w * float(follower["_z_guard"])
        lines.append(LineStrength(
            line_id=int(lid), cars=cars,
            front_power=front, guard_power=guard,
            total_power=front + guard,
            is_solo=len(cars) == 1,
        ))

    for _, row in df[df["line_id"].isna()].iterrows():
        lines.append(LineStrength(
            line_id=solo_id, cars=[int(row["car_no"])],
            front_power=float(row["_z_front"]), guard_power=0.0,
            total_power=float(row["_z_front"]),
            is_solo=True,
        ))
        solo_id -= 1

    return lines
