"""レース内の脚質構成 (style matchup) 特徴量.

競輪の定石: レースの性格は「先行型が何人いるか」でほぼ決まる。
- 逃げ屋が1人だけ → 主導権独占でそのラインが圧倒的有利 (is_lone_nige)
- 逃げ屋が多い → 先行争いでハイペース → 差し・追込が有利 (n_nige_in_race)
- B回数 (バック先頭回数) は「実際に先行できているか」の直接指標。
  レース内で相対化することで車番や絶対スケールに依存しない (思想②)。

入力: builder の base entries DataFrame (style 正規化済み、
      nige_cnt/makuri_cnt/sashi_cnt/mark_cnt/b_count, bank_length を含む)
出力: car_no を index とする特徴量 DataFrame
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from keirin.features.style_norm import normalize_style

STYLE_CONTEXT_COLUMNS = [
    "style_nige", "style_ryo", "style_oi",
    "n_nige_in_race", "nige_ratio", "is_lone_nige",
    "b_count_rank", "b_count_share",
    "nige_makuri_rate", "sashi_mark_rate",
    "style_x_short_bank", "style_x_long_bank",
]


def race_style_context(entries: pd.DataFrame) -> pd.DataFrame:
    """Compute within-race style composition features. Index: car_no."""
    df = entries.copy()
    out = pd.DataFrame(index=df["car_no"].values)
    out.index.name = "car_no"

    style = df["style"].map(normalize_style) if "style" in df.columns else pd.Series(None, index=df.index)
    style = style.reset_index(drop=True)

    out["style_nige"] = (style == "逃").astype(int).values
    out["style_ryo"] = (style == "両").astype(int).values
    out["style_oi"] = (style == "追").astype(int).values

    n_active = max(len(df), 1)
    n_nige = int((style == "逃").sum())
    out["n_nige_in_race"] = n_nige
    out["nige_ratio"] = n_nige / n_active
    out["is_lone_nige"] = ((style == "逃") & (n_nige == 1)).astype(int).values

    # --- B回数のレース内相対化 (先行力) ---------------------------------------
    b = pd.to_numeric(df.get("b_count"), errors="coerce").reset_index(drop=True)
    if b.notna().any():
        # rank 1 = 最多B回数。NaN は NaN のまま (LightGBM ネイティブ処理)
        out["b_count_rank"] = b.rank(ascending=False, method="min").values
        total_b = b.sum()
        out["b_count_share"] = (b / total_b).values if total_b > 0 else np.where(b.notna(), 0.0, np.nan)
    else:
        out["b_count_rank"] = np.nan
        out["b_count_share"] = np.nan

    # --- 決まり手回数の構成比 (自力型 vs 追込型) -------------------------------
    cnt_cols = ["nige_cnt", "makuri_cnt", "sashi_cnt", "mark_cnt"]
    cnts = {c: pd.to_numeric(df.get(c), errors="coerce").reset_index(drop=True) for c in cnt_cols}
    total = sum(cnts.values())
    with np.errstate(invalid="ignore", divide="ignore"):
        nm = (cnts["nige_cnt"] + cnts["makuri_cnt"]) / total
        sm = (cnts["sashi_cnt"] + cnts["mark_cnt"]) / total
    out["nige_makuri_rate"] = nm.where(total > 0).values
    out["sashi_mark_rate"] = sm.where(total > 0).values

    # --- バンク周長との交互作用 -------------------------------------------------
    bank = pd.to_numeric(df.get("bank_length"), errors="coerce").reset_index(drop=True)
    short_bank = (bank <= 350).fillna(False)
    long_bank = (bank >= 500).fillna(False)
    out["style_x_short_bank"] = (out["style_nige"].values & short_bank.values).astype(int)
    out["style_x_long_bank"] = (out["style_oi"].values & long_bank.values).astype(int)

    return out
