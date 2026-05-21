"""Tests for matchup features: head-to-head stats and grade-specific form."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sqlalchemy import create_engine, text


def _make_engine():
    """In-memory SQLite with player_race_log data for testing."""
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE player_race_log (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id TEXT NOT NULL,
                race_id   TEXT NOT NULL,
                race_date TEXT NOT NULL,
                venue_id  TEXT,
                grade     TEXT,
                finish    INTEGER,
                scratched INTEGER DEFAULT 0,
                UNIQUE (player_id, race_id)
            )
        """))
        rows = [
            # r1: p1 wins, p2 2nd, p3 3rd, p4 4th  (grade F1)
            ("p1", "r1", "2026-01-01", "F1", 1, 0),
            ("p2", "r1", "2026-01-01", "F1", 2, 0),
            ("p3", "r1", "2026-01-01", "F1", 3, 0),
            ("p4", "r1", "2026-01-01", "F1", 4, 0),
            # r2: p2 wins, p1 2nd, p3 4th (grade F1)
            ("p1", "r2", "2026-01-10", "F1", 2, 0),
            ("p2", "r2", "2026-01-10", "F1", 1, 0),
            ("p3", "r2", "2026-01-10", "F1", 4, 0),
            # r3: p1 wins alone against p5 (different opponent — grade G3)
            ("p1", "r3", "2026-02-01", "G3", 1, 0),
            ("p5", "r3", "2026-02-01", "G3", 2, 0),
            # r4: p3 wins with p6 — different field, no p1/p2
            ("p3", "r4", "2026-02-15", "F2", 1, 0),
            ("p6", "r4", "2026-02-15", "F2", 2, 0),
            # r5: scratched race — should be ignored
            ("p1", "r5", "2026-03-01", "F1", None, 1),
            ("p2", "r5", "2026-03-01", "F1", None, 1),
        ]
        for r in rows:
            conn.execute(
                text("""
                    INSERT OR IGNORE INTO player_race_log
                    (player_id, race_id, race_date, grade, finish, scratched)
                    VALUES (:pid, :rid, :rd, :gr, :fi, :sc)
                """),
                {"pid": r[0], "rid": r[1], "rd": r[2],
                 "gr": r[3], "fi": r[4], "sc": r[5]},
            )
    return engine


# ---------------------------------------------------------------------------
# h2h_stats tests
# ---------------------------------------------------------------------------

def test_h2h_stats_basic():
    """p1 and p2 have two shared races (r1, r2). p1 won r1, placed 2nd in r2."""
    from keirin.features.matchup import h2h_stats

    engine = _make_engine()
    result = h2h_stats(engine, ["p1", "p2", "p3"], "2026-05-01")

    assert "p1" in result.index, "p1 should appear"
    p1 = result.loc["p1"]
    # p1: r1 finish=1, r2 finish=2 → win_rate=0.5, top3_rate=1.0
    assert abs(p1["h2h_win_rate"] - 0.5) < 0.01
    assert abs(p1["h2h_top3_rate"] - 1.0) < 0.01
    assert int(p1["h2h_n_encounters"]) == 2


def test_h2h_stats_loser():
    """p3 finished 3rd and 4th in shared races → win_rate=0, top3_rate=0.5."""
    from keirin.features.matchup import h2h_stats

    engine = _make_engine()
    result = h2h_stats(engine, ["p1", "p2", "p3"], "2026-05-01")

    p3 = result.loc["p3"]
    assert abs(p3["h2h_win_rate"] - 0.0) < 0.01
    assert abs(p3["h2h_top3_rate"] - 0.5) < 0.01  # finish=3 in r1 (top3), finish=4 in r2 (not top3)


def test_h2h_stats_no_shared_races():
    """p5 and p6 never raced together — h2h result should be empty for them."""
    from keirin.features.matchup import h2h_stats

    engine = _make_engine()
    result = h2h_stats(engine, ["p5", "p6"], "2026-05-01")
    # p5 only raced with p1/p2 (r3) — not in this field; p6 only raced with p3 (r4)
    assert result.empty or (len(result) == 0), "no shared races → empty result"


def test_h2h_stats_empty_player_list():
    from keirin.features.matchup import h2h_stats

    engine = _make_engine()
    result = h2h_stats(engine, [], "2026-05-01")
    assert result.empty


def test_h2h_stats_ignores_scratched():
    """Scratched race r5 should not appear in h2h encounters."""
    from keirin.features.matchup import h2h_stats

    engine = _make_engine()
    result = h2h_stats(engine, ["p1", "p2"], "2026-05-01")
    # Only r1 and r2 are valid (not scratched) → 2 encounters each
    assert int(result.loc["p1"]["h2h_n_encounters"]) == 2
    assert int(result.loc["p2"]["h2h_n_encounters"]) == 2


def test_h2h_stats_ref_date_cutoff():
    """Races on or after ref_date must be excluded."""
    from keirin.features.matchup import h2h_stats

    engine = _make_engine()
    # ref_date = 2026-01-05: only r1 (2026-01-01) qualifies; r2 (2026-01-10) is after
    result = h2h_stats(engine, ["p1", "p2", "p3"], "2026-01-05")
    assert int(result.loc["p1"]["h2h_n_encounters"]) == 1
    assert abs(result.loc["p1"]["h2h_win_rate"] - 1.0) < 0.01  # p1 won r1


# ---------------------------------------------------------------------------
# grade_form tests
# ---------------------------------------------------------------------------

def test_grade_form_basic():
    """p1 has two F1 races (r1, r2): finish 1 and 2 → grade_top3_rate=1.0."""
    from keirin.features.matchup import grade_form

    engine = _make_engine()
    result = grade_form(engine, ["p1", "p2", "p3"], "2026-05-01", "F1")

    assert "p1" in result.index
    assert abs(result.loc["p1"]["grade_top3_rate"] - 1.0) < 0.01


def test_grade_form_p3_f1_and_f2():
    """p3 has F1 races with finish 3 and 4 → top3_rate=0.5."""
    from keirin.features.matchup import grade_form

    engine = _make_engine()
    result = grade_form(engine, ["p3"], "2026-05-01", "F1")

    assert "p3" in result.index
    assert abs(result.loc["p3"]["grade_top3_rate"] - 0.5) < 0.01


def test_grade_form_no_history():
    """If player has no races at this grade, they won't appear (NaN in join)."""
    from keirin.features.matchup import grade_form

    engine = _make_engine()
    # p4 only raced in F1/r1; asking for G3
    result = grade_form(engine, ["p4"], "2026-05-01", "G3")
    assert result.empty or "p4" not in result.index


def test_grade_form_none_grade():
    """None grade should return empty DataFrame."""
    from keirin.features.matchup import grade_form

    engine = _make_engine()
    result = grade_form(engine, ["p1", "p2"], "2026-05-01", None)
    assert result.empty


def test_grade_form_empty_players():
    from keirin.features.matchup import grade_form

    engine = _make_engine()
    result = grade_form(engine, [], "2026-05-01", "F1")
    assert result.empty


# ---------------------------------------------------------------------------
# _ndcg_at_k tests (from train module)
# ---------------------------------------------------------------------------

def test_ndcg_perfect_ranking():
    """Perfect ranking should give NDCG=1.0."""
    import numpy as np
    from keirin.models.train import _ndcg_at_k

    y_true = np.array([10.0, 7.0, 5.0, 1.0, 0.0])
    y_score = np.array([4.0, 3.0, 2.0, 1.0, 0.0])  # same order
    assert abs(_ndcg_at_k(y_true, y_score, k=3) - 1.0) < 1e-9


def test_ndcg_worst_ranking():
    """Reversed ranking should give NDCG < 1.0 (worst possible)."""
    import numpy as np
    from keirin.models.train import _ndcg_at_k

    y_true = np.array([10.0, 7.0, 5.0, 1.0, 0.0])
    y_score = np.array([0.0, 1.0, 2.0, 3.0, 4.0])  # reversed order
    ndcg = _ndcg_at_k(y_true, y_score, k=3)
    assert ndcg < 1.0
    assert ndcg >= 0.0


def test_ndcg_at_1_correct_winner():
    """NDCG@1 = 1.0 when the predicted top car is the true winner."""
    import numpy as np
    from keirin.models.train import _ndcg_at_k

    # Car with relevance=10 (1st place) is predicted first
    y_true = np.array([10.0, 7.0, 5.0])
    y_score = np.array([9.0, 5.0, 3.0])
    assert abs(_ndcg_at_k(y_true, y_score, k=1) - 1.0) < 1e-9


def test_ndcg_at_1_wrong_winner():
    """NDCG@1 < 1.0 when model predicts the wrong winner."""
    import numpy as np
    from keirin.models.train import _ndcg_at_k

    # Car with relevance=7 (2nd place) is predicted first
    y_true = np.array([10.0, 7.0, 5.0])
    y_score = np.array([3.0, 9.0, 5.0])
    ndcg = _ndcg_at_k(y_true, y_score, k=1)
    assert ndcg < 1.0


def test_ndcg_empty():
    import numpy as np
    from keirin.models.train import _ndcg_at_k

    assert _ndcg_at_k(np.array([]), np.array([]), k=1) == 0.0


def test_ndcg_all_zero_relevance():
    """All-zero relevance (no known results) → NDCG=0.0 (idcg=0)."""
    import numpy as np
    from keirin.models.train import _ndcg_at_k

    y_true = np.zeros(5)
    y_score = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert _ndcg_at_k(y_true, y_score, k=3) == 0.0
