"""Smoke tests for rank_prediction (ranking from p_top3 or ranker scores)."""
from __future__ import annotations

from keirin.models.predict import rank_prediction


def test_rank_prediction_basic_ordering():
    """Without ranker_scores, car with highest p_top3 should rank 1st."""
    p_top3 = {1: 0.7, 2: 0.5, 3: 0.3, 4: 0.2}
    result = rank_prediction(p_top3)
    assert result[0]["car_no"] == 1
    assert result[0]["pred_prob_1st"] > result[-1]["pred_prob_1st"]


def test_rank_prediction_probs_in_range():
    """All probabilities should be in [0, 1]."""
    p_top3 = {1: 0.6, 2: 0.4, 3: 0.3, 4: 0.2, 5: 0.5}
    result = rank_prediction(p_top3)
    for r in result:
        for k in ("pred_prob_1st", "pred_prob_2nd", "pred_prob_3rd"):
            assert 0.0 <= r[k] <= 1.0


def test_rank_prediction_total_1st_sums_to_1():
    """Sum of P(1st) across all cars should equal 1 (someone wins)."""
    p_top3 = {1: 0.7, 2: 0.5, 3: 0.3, 4: 0.2, 5: 0.4, 6: 0.6, 7: 0.35}
    result = rank_prediction(p_top3)
    total = sum(r["pred_prob_1st"] for r in result)
    # Allow small slack due to floating point
    assert abs(total - 1.0) < 0.01


def test_rank_prediction_uses_ranker_scores_when_provided():
    """When rank_scores is given, ordering should follow scores, not p_top3."""
    p_top3 = {1: 0.4, 2: 0.5, 3: 0.3, 4: 0.6}
    # Override: car 1 should win according to ranker, even though p_top3 has car 4 highest
    rank_scores = {1: 2.0, 2: 0.5, 3: -0.1, 4: 0.3}
    result = rank_prediction(p_top3, rank_scores=rank_scores)
    assert result[0]["car_no"] == 1, f"ranker says car 1 wins, got {result[0]['car_no']}"


def test_rank_prediction_fallback_when_no_ranker():
    """Without rank_scores, falls back to PL strengths from p_top3."""
    p_top3 = {1: 0.4, 2: 0.5, 3: 0.3, 4: 0.6}
    result = rank_prediction(p_top3)
    assert result[0]["car_no"] == 4, "highest p_top3 should win in fallback"


def test_rank_prediction_empty_input():
    assert rank_prediction({}) == []


def test_rank_prediction_expected_rank_ordering():
    """Result should be sorted by expected_rank ascending."""
    p_top3 = {1: 0.5, 2: 0.3, 3: 0.4, 4: 0.6, 5: 0.2}
    result = rank_prediction(p_top3)
    expected_ranks = [r["expected_rank"] for r in result]
    assert expected_ranks == sorted(expected_ranks), "result not sorted by expected_rank"
