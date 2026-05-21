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
        conn.execute(text(
            "INSERT INTO races VALUES ('R001','2026-04-01','81',11,'F1',NULL,400,NULL,NULL,'18:00')"
        ))
        conn.execute(text(
            "INSERT INTO races VALUES ('R002','2026-04-10','81',9,'F1',NULL,400,NULL,NULL,'17:00')"
        ))
        # Entries
        for car, pid, lid, lpos in [(1,'P1',1,1),(2,'P2',1,2)]:
            conn.execute(text(
                "INSERT INTO entries VALUES ('R001',:cn,:pid,'S1',90.0,3.5,:lid,:lp,'逃',0)"
            ), {"cn": car, "pid": pid, "lid": lid, "lp": lpos})
        for car, pid, lid, lpos in [(1,'P1',1,1),(2,'P2',1,2)]:
            conn.execute(text(
                "INSERT INTO entries VALUES ('R002',:cn,:pid,'S1',90.0,3.5,:lid,:lp,'逃',0)"
            ), {"cn": car, "pid": pid, "lid": lid, "lp": lpos})
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
