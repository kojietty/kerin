"""オッズ乖離シグナル — the core of value-bet detection.

The central idea: keirin trifecta odds are set by the public's collective bet
(parimutuel). The public tends to:
  1. Over-weight recent bad results (recency bias)
  2. Under-weight venue specialist records
  3. Over-react to rest periods (returning from injury / suspension)
  4. Miss intra-line structural advantages
  5. Follow simple heuristics (favour the highest 競走得点)

When any of these biases create a systematic gap between a player's *actual
probability* and the *implied probability from the odds*, we get EV > 1.

This module computes divergence signals at the (race_id, car_no) level.
They are used both as model features *and* as human-readable explanations
in the prediction report.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine


# ---------------------------------------------------------------------------
# 1. Score-odds divergence (得点オッズ乖離)
# ---------------------------------------------------------------------------

def score_implied_rank(df_entries: pd.DataFrame) -> pd.DataFrame:
    """Rank players by 競走得点 vs implied rank from odds market.

    A player ranked #2 by rating but priced as #4 by odds has positive
    「score_odds_gap」— the market undervalues them.

    Inputs: df_entries must have columns: car_no, rating, implied_prob
      (implied_prob = 1 / fair_odds, normalised to sum to 1).
    Returns df_entries with added columns: score_rank, odds_rank, score_odds_gap.
    """
    df = df_entries.copy()
    df["score_rank"] = df["rating"].rank(ascending=False, method="min")
    df["odds_rank"] = df["implied_prob"].rank(ascending=False, method="min")
    # Positive = rated stronger than market implies → undervalued
    df["score_odds_gap"] = df["odds_rank"] - df["score_rank"]
    return df


# ---------------------------------------------------------------------------
# 2. Rest-day discount signal (休養明け)
# ---------------------------------------------------------------------------

def rest_day_signal(rest_days_series: pd.Series) -> pd.Series:
    """公共の過剰反応サイン: longer rest → market over-penalises the player.

    Observation from Japanese keirin research: rest ≥ 15 days does NOT
    predict significantly worse performance, but odds are often inflated
    because bettors assume the player is out of form.

    Returns a Series of signal strengths:
      0  = short rest (≤ 7 d): no mispricing expected
      1  = medium rest (8–21 d): mild inflation signal
      2  = long rest (22–60 d): strong inflation signal — investigate
      3  = very long rest (> 60 d): maximum discount signal
    (Capped at 3 to avoid extreme weights from arbitrarily long absences.)
    """
    def _signal(d: float) -> int:
        if pd.isna(d) or d <= 7:
            return 0
        if d <= 21:
            return 1
        if d <= 60:
            return 2
        return 3

    return rest_days_series.map(_signal)


# ---------------------------------------------------------------------------
# 3. Venue specialist discount (バンク適性未認知)
# ---------------------------------------------------------------------------

def venue_specialist_discount(
    venue_specialist_score: pd.Series,
    *,
    threshold: float = 0.15,
) -> pd.Series:
    """Binary flag: player has significantly higher venue top3 rate than overall.

    When specialist_score ≥ threshold AND venue_n_races ≥ 5 (enough sample),
    the market may not have priced in the venue edge.
    """
    return (venue_specialist_score >= threshold).astype(int)


# ---------------------------------------------------------------------------
# 4. Form-rebound signal (前走着外れ + 地力高)
# ---------------------------------------------------------------------------

def form_rebound_signal(
    last_finish: pd.Series,
    avg_finish_long: pd.Series,
    *,
    recent_bad_threshold: float = 5.0,
    long_avg_good_threshold: float = 3.0,
) -> pd.Series:
    """Recency-bias detector: last race was bad, but long-term form is good.

    A player who finished 5th last time but averages 2.5 over 20 races is
    a classic「見た目より強い」player — the public over-penalises the recent poor
    finish (recency bias). This signal can provide meaningful EV.

    Returns 1.0 if (last_finish ≥ recent_bad_threshold AND avg_finish_long
    ≤ long_avg_good_threshold), else 0.0.
    """
    bad_recent = last_finish >= recent_bad_threshold
    good_long = avg_finish_long <= long_avg_good_threshold
    return (bad_recent & good_long).astype(float)


def last_finish(engine: Engine, ref_date: str) -> pd.Series:
    """Most recent finish position per player (before ref_date)."""
    with engine.begin() as conn:
        df = pd.read_sql(
            text(
                """
                SELECT player_id, finish
                FROM player_race_log
                WHERE race_date = (
                  SELECT MAX(race_date)
                  FROM player_race_log prl2
                  WHERE prl2.player_id = player_race_log.player_id
                    AND prl2.race_date < :ref_date
                    AND prl2.scratched = 0
                )
                  AND scratched = 0
                """
            ),
            conn,
            params={"ref_date": ref_date},
        )
    return df.set_index("player_id")["finish"]


# ---------------------------------------------------------------------------
# 5. Line-mismatch discount (ライン構造ミスマッチ)
# ---------------------------------------------------------------------------

def line_mismatch_signal(
    is_follower: pd.Series,
    rating_vs_leader_gap: pd.Series,
    *,
    gap_threshold: float = 3.0,
) -> pd.Series:
    """「実力番手」signal — strong player stuck behind a weaker leader.

    These riders are often undervalued because:
    - Bettors treat followers as pure beneficiaries of the lead
    - But if the leader is weaker, the follower may need to take a risky
      outside line to win, increasing variance but not changing underlying ability

    Returns 1.0 when: is_follower AND (rating_vs_leader_gap ≥ gap_threshold).
    """
    return (is_follower.astype(bool) & (rating_vs_leader_gap >= gap_threshold)).astype(float)


# ---------------------------------------------------------------------------
# Composite divergence score (all signals combined, weighted)
# ---------------------------------------------------------------------------

@dataclass
class DivergenceWeights:
    score_odds_gap: float = 0.30
    rest_day_signal: float = 0.20
    venue_specialist: float = 0.25
    form_rebound: float = 0.15
    line_mismatch: float = 0.10


def composite_divergence(
    df: pd.DataFrame,
    weights: DivergenceWeights | None = None,
) -> pd.Series:
    """Weighted sum of all divergence signals. Range roughly [−3, +6].

    Positive = market likely undervalues this player → higher EV potential.
    This is used as a single feature in LightGBM, *and* as the human-readable
    「乖離スコア」shown in the prediction report.

    df must have columns: score_odds_gap, rest_day_signal (0-3),
    venue_specialist_discount (0/1), form_rebound_signal (0/1),
    line_mismatch_signal (0/1).
    """
    if weights is None:
        weights = DivergenceWeights()

    # Normalise score_odds_gap to [-1, +1] range (clip extremes).
    norm_sog = df["score_odds_gap"].clip(-4, 4) / 4.0
    # Normalise rest signal to [0, 1]
    norm_rest = df["rest_day_signal"].clip(0, 3) / 3.0

    score = (
        weights.score_odds_gap * norm_sog
        + weights.rest_day_signal * norm_rest
        + weights.venue_specialist * df.get("venue_specialist_discount", 0)
        + weights.form_rebound * df.get("form_rebound_signal", 0)
        + weights.line_mismatch * df.get("line_mismatch_signal", 0)
    )
    return score.rename("divergence_score")
