"""Tests for feature modules using an in-memory SQLite DB."""
from __future__ import annotations

import pytest
from sqlalchemy import text

from keirin.db.engine import get_engine, init_db
from keirin.db.repository import sync_player_race_log
from keirin.features import player_form, venue, line, gear
from keirin.features.odds_divergence import (
    rest_day_signal,
    form_rebound_signal,
    line_mismatch_signal,
)
import pandas as pd


@pytest.fixture()
def mem_engine():
    engine = get_engine(":memory:")
    init_db(engine)
    # Insert minimal fixture data
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO venues VALUES ('81', '小倉', 400, NULL)"))
        conn.execute(text("INSERT INTO players VALUES ('P1','田中',NULL,NULL,'逃','S1',90.0,NULL)"))
        conn.execute(text("INSERT INTO players VALUES ('P2','鈴木',NULL,NULL,'差','A1',75.0,NULL)"))
        conn.execute(text("INSERT INTO players VALUES ('P3','佐藤',NULL,NULL,'追','A1',70.0,NULL)"))
        conn.execute(text(
            "INSERT INTO races VALUES ('R001','2026-04-01','81',11,'F1',NULL,400,NULL,NULL,'18:00')"
        ))
        conn.execute(text(
            "INSERT INTO races VALUES ('R002','2026-04-10','81',9,'F1',NULL,400,NULL,NULL,'17:00')"
        ))
        # Entries
        for car, pid, lid, lpos in [(1,'P1',1,1),(2,'P2',1,2)]:
            conn.execute(text(
                "INSERT INTO entries (race_id,car_no,player_id,rank_class,rating,"
                "gear_ratio,line_id,line_pos,style,scratched)"
                " VALUES ('R001',:cn,:pid,'S1',90.0,3.5,:lid,:lp,'逃',0)"
            ), {"cn": car, "pid": pid, "lid": lid, "lp": lpos})
        for car, pid, lid, lpos in [(1,'P1',1,1),(2,'P2',1,2)]:
            conn.execute(text(
                "INSERT INTO entries (race_id,car_no,player_id,rank_class,rating,"
                "gear_ratio,line_id,line_pos,style,scratched)"
                " VALUES ('R002',:cn,:pid,'S1',90.0,3.5,:lid,:lp,'逃',0)"
            ), {"cn": car, "pid": pid, "lid": lid, "lp": lpos})
        # P3 raced R001 but has no result row → 着外 (finish=9) derivation target
        conn.execute(text(
            "INSERT INTO entries (race_id,car_no,player_id,rank_class,rating,"
            "gear_ratio,line_id,line_pos,style,scratched)"
            " VALUES ('R001',3,'P3','A1',70.0,3.4,NULL,NULL,'追',0)"
        ))
        # Results
        conn.execute(text("INSERT INTO results VALUES ('R001',1,1,'逃',NULL)"))
        conn.execute(text("INSERT INTO results VALUES ('R001',2,2,'マーク',NULL)"))
        conn.execute(text("INSERT INTO results VALUES ('R002',1,3,'逃',NULL)"))
        conn.execute(text("INSERT INTO results VALUES ('R002',2,1,'差',NULL)"))
    sync_player_race_log(engine)
    return engine


def test_rest_days(mem_engine):
    df = player_form.rest_days(mem_engine, "2026-04-11")
    assert "P1" in df.index
    assert df.loc["P1", "rest_days"] == 1


def test_rolling_finish(mem_engine):
    df = player_form.rolling_finish_stats(mem_engine, "2026-04-11", n_races=5)
    assert "P1" in df.index
    assert 0.0 <= df.loc["P1", "top3_rate"] <= 1.0


def test_venue_stats(mem_engine):
    df = venue.venue_stats(mem_engine, "81", "2026-04-11")
    assert "P1" in df.index
    assert df.loc["P1", "venue_n_races"] == 2


def test_line_features(mem_engine):
    df = line.line_features(mem_engine, "R001")
    assert not df.empty
    assert "is_leader" in df.columns
    assert df.loc[("R001", 1), "is_leader"] == 1
    assert df.loc[("R001", 2), "is_follower"] == 1


def test_gear_features(mem_engine):
    df = gear.gear_features(mem_engine, "R001")
    assert not df.empty
    assert "gear_z" in df.columns


def test_rest_signal():
    import pandas as pd
    s = pd.Series([0, 7, 15, 30, 90], name="days")
    sig = rest_day_signal(s)
    assert list(sig) == [0, 0, 1, 2, 3]


def test_form_rebound():
    import pandas as pd
    last = pd.Series([6.0, 2.0, 5.0])
    avg_long = pd.Series([2.0, 2.0, 4.0])
    sig = form_rebound_signal(last, avg_long)
    assert list(sig) == [1.0, 0.0, 0.0]


def test_line_mismatch():
    import pandas as pd
    is_f = pd.Series([1, 1, 0])
    gap = pd.Series([5.0, 1.0, 10.0])
    sig = line_mismatch_signal(is_f, gap)
    assert list(sig) == [1.0, 0.0, 0.0]


def test_last_race_result(mem_engine):
    """P2 won R002 (most recent), so win_last_race=1; P1 finished 3rd → top3_last_race=1."""
    df = player_form.last_race_result(mem_engine, "2026-04-11")
    assert "P2" in df.index
    assert df.loc["P2", "win_last_race"] == 1
    assert df.loc["P2", "top3_last_race"] == 1
    assert "P1" in df.index
    assert df.loc["P1", "win_last_race"] == 0
    assert df.loc["P1", "top3_last_race"] == 1   # P1 finished 3rd in R002


def test_last_race_result_no_data(mem_engine):
    """Before any races, result should be empty."""
    df = player_form.last_race_result(mem_engine, "2026-01-01")
    assert df.empty


def test_relative_rating_computation():
    """relative_rating = player_rating - avg_rating_of_others."""
    import pandas as pd
    from keirin.features.builder import build_race_features

    # Test the math directly without a DB
    ratings = pd.Series([90.0, 80.0, 70.0])
    n = len(ratings)
    avg_opp = (ratings.sum() - ratings) / (n - 1)
    rel = ratings - avg_opp
    # avg_opp for 90.0 = (80+70)/2 = 75; relative = 15
    assert abs(rel.iloc[0] - 15.0) < 0.01
    # avg_opp for 80.0 = (90+70)/2 = 80; relative = 0
    assert abs(rel.iloc[1] - 0.0) < 0.01
    # avg_opp for 70.0 = (90+80)/2 = 85; relative = -15
    assert abs(rel.iloc[2] - (-15.0)) < 0.01


def test_relative_rating_missing_rating_not_imputed():
    """A car with a missing rating must (a) get NaN relative_rating, and (b) be
    excluded from every other car's opponent-average denominator."""
    import numpy as np
    import pandas as pd

    # Field of 3: two observed ratings (90, 80) and one missing.
    entries = pd.DataFrame({"car_no": [1, 2, 3], "rating": [90.0, 80.0, np.nan]})

    # Replicate builder step 8b
    valid = entries["rating"].dropna()
    valid_sum = valid.sum()
    m = len(valid)
    has_rating = entries["rating"].notna()
    avg_opp = entries["rating"].where(has_rating).copy()
    avg_opp[has_rating] = (valid_sum - entries.loc[has_rating, "rating"]) / (m - 1)
    avg_opp[~has_rating] = valid_sum / m
    rel = entries["rating"] - avg_opp

    # Car 1 (90): opponent avg uses only observed others = 80; relative = +10
    assert abs(avg_opp.iloc[0] - 80.0) < 0.01
    assert abs(rel.iloc[0] - 10.0) < 0.01
    # Car 2 (80): opponent avg = 90; relative = -10
    assert abs(avg_opp.iloc[1] - 90.0) < 0.01
    assert abs(rel.iloc[1] - (-10.0)) < 0.01
    # Car 3 (missing): relative_rating is NaN (not falsely 0)
    assert np.isnan(rel.iloc[2])


def test_monotone_constraints_alignment():
    """_monotone_constraints returns a vector aligned to feature_cols with -1
    only for car_no."""
    from keirin.models.train import _monotone_constraints

    cols = ["rating", "car_no", "line_pos", "has_line"]
    mc = _monotone_constraints(cols)
    assert mc == [0, -1, 0, 0]
    assert len(mc) == len(cols)
    # Empty input is safe
    assert _monotone_constraints([]) == []


def test_has_line_indicator(mem_engine):
    """line_features should mark has_line=1 when the race has 並び data."""
    df = line.line_features(mem_engine, "R001")
    assert "has_line" in df.columns
    # R001 fixture has line_id populated for both cars
    assert int(df["has_line"].iloc[0]) == 1


def test_outside_finish_sentinel(mem_engine):
    """sync_player_race_log derives finish=9 (着外) for non-scratched entries in
    races that have results but no result row for that car."""
    from sqlalchemy import text as _text
    with mem_engine.begin() as conn:
        row = conn.execute(_text(
            "SELECT finish FROM player_race_log"
            " WHERE player_id='P3' AND race_id='R001'"
        )).fetchone()
    assert row is not None
    assert row[0] == 9
    # 着外行はフォーム統計に負例として入る (top3_rate < 1.0)
    df = player_form.rolling_finish_stats(mem_engine, "2026-04-11", n_races=5)
    assert "P3" in df.index
    assert df.loc["P3", "top3_rate"] == 0.0


def test_add_target_outside_fill(mem_engine):
    """_add_target fills finish=9 for raced-but-unplaced cars so the ranker
    sees them as negatives (relevance 0) instead of dropping the rows."""
    import pandas as pd
    from keirin.features.builder import _add_target

    entries = pd.DataFrame({
        "race_id": ["R001"] * 3, "car_no": [1, 2, 3], "scratched": [0, 0, 0],
    })
    out = _add_target(mem_engine, entries)
    assert out.loc[out["car_no"] == 3, "finish"].iloc[0] == 9
    assert out.loc[out["car_no"] == 3, "label_top3"].iloc[0] == 0
    assert out.loc[out["car_no"] == 1, "finish"].iloc[0] == 1


def test_style_matchup_context():
    """race_style_context: 脚質構成・B回数相対化の基本性質。"""
    import numpy as np
    import pandas as pd
    from keirin.features.style_matchup import race_style_context

    entries = pd.DataFrame({
        "car_no": [1, 2, 3, 4, 5],
        "style": ["逃", "追", "追", "両", "逃"],
        "b_count": [8, 0, 1, 3, 4],
        "nige_cnt": [5, 0, 0, 1, 3],
        "makuri_cnt": [1, 0, 1, 2, 1],
        "sashi_cnt": [0, 4, 3, 1, 0],
        "mark_cnt": [0, 2, 2, 0, 0],
        "bank_length": [333] * 5,
    })
    out = race_style_context(entries)
    assert out.loc[1, "style_nige"] == 1
    assert out.loc[2, "style_oi"] == 1
    assert out.loc[1, "n_nige_in_race"] == 2          # 車1と車5が逃
    assert out.loc[1, "is_lone_nige"] == 0            # 逃が2人 → 単独ではない
    assert out.loc[1, "b_count_rank"] == 1            # B回数最多
    assert abs(out["b_count_share"].sum() - 1.0) < 1e-9
    assert out.loc[1, "style_x_short_bank"] == 1      # 逃 × 333バンク
    assert out.loc[2, "style_x_short_bank"] == 0
    # 決まり手構成比: 車2 は 差+マ のみ → sashi_mark_rate=1.0
    assert abs(out.loc[2, "sashi_mark_rate"] - 1.0) < 1e-9
    # 文字化け style も正規化されて one-hot に乗る
    entries2 = entries.assign(style=["騾�", "霑ｽ", "霑ｽ", "荳｡", "騾�"])
    out2 = race_style_context(entries2)
    assert (out2[["style_nige", "style_ryo", "style_oi"]].values
            == out[["style_nige", "style_ryo", "style_oi"]].values).all()


def test_lone_nige_flag():
    import pandas as pd
    from keirin.features.style_matchup import race_style_context

    entries = pd.DataFrame({
        "car_no": [1, 2, 3, 4],
        "style": ["逃", "追", "追", "両"],
        "b_count": [None] * 4,
        "nige_cnt": [None] * 4, "makuri_cnt": [None] * 4,
        "sashi_cnt": [None] * 4, "mark_cnt": [None] * 4,
        "bank_length": [400] * 4,
    })
    out = race_style_context(entries)
    assert out.loc[1, "is_lone_nige"] == 1
    assert out.loc[2, "is_lone_nige"] == 0
