"""Master feature builder.

build_race_features(engine, race_id, ref_date, latest_odds)
  → DataFrame with one row per (race_id, car_no), all features joined.

This is the single entry point for both training (iterating over historical
races) and inference (building features for today's races right before predict).
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from keirin.features import gear, line, matchup, odds_divergence, player_form, venue
from keirin.features.style_matchup import STYLE_CONTEXT_COLUMNS, race_style_context

log = logging.getLogger(__name__)

# Master list of feature columns (for documentation + train/predict alignment)
FEATURE_COLUMNS = [
    # --- Player ability ---
    "rating",               # 競走得点
    "rank_s1", "rank_s2", "rank_a1", "rank_a2", "rank_a3",  # rank class one-hot
    # --- Recent form ---
    "avg_finish_5", "top3_rate_5", "top3_rate_10", "top3_rate_20",
    "win_rate_10", "form_trend", "rest_days",
    "win_last_race", "top3_last_race",
    # --- Field context ---
    "relative_rating", "avg_opponent_rating",
    # --- Kimarite style ---
    "km_nige", "km_maki", "km_sashi", "km_mark",
    # --- Venue ---
    "bank_length", "bank_angle",
    "venue_top3_rate", "venue_n_races", "venue_specialist_score",
    "top3_333", "top3_400", "top3_500",
    # --- Line structure ---
    "line_pos", "line_len", "is_leader", "is_follower", "is_solo",
    "rating_vs_leader_gap", "has_line",
    # --- Style matchup (レース内の脚質構成) ---
    *STYLE_CONTEXT_COLUMNS,
    # --- Simulation (ライン力学・展開シミュレーション) ---
    "p_sim_win", "p_sim_top3", "p_front", "sim_used",
    # --- Gear ---
    "gear_ratio", "gear_z", "gear_rank", "gear_above_field",
    # --- Race context ---
    "grade_gp", "grade_g1", "grade_g2", "grade_g3", "grade_f1", "grade_f2",
    "car_no",
    # --- Divergence (odds-ability gap) ---
    "score_odds_gap", "rest_day_signal", "venue_specialist_discount",
    "form_rebound_signal", "line_mismatch_signal",
    "divergence_score",
    # --- Matchup (head-to-head vs. this specific field) ---
    "h2h_win_rate", "h2h_top3_rate", "h2h_n_encounters",
    "grade_top3_rate",
]

# Target column
TARGET = "label_top3"  # 1 if finish <= 3 else 0


def build_race_features(
    engine: Engine,
    race_id: str,
    ref_date: str,
    *,
    latest_odds: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Assemble all features for a single race.

    Args:
        race_id:  The race identifier.
        ref_date: ISO date string. All look-back queries use race_date < ref_date.
        latest_odds: Optional dict {combo_str: odds} for divergence computation.
                     If None, divergence features that require odds are set to NaN.

    Returns a DataFrame with one row per (race_id, car_no).
    """
    # --- 1. Base entries -------------------------------------------------------
    with engine.begin() as conn:
        entries = pd.read_sql(
            text(
                """
                SELECT e.race_id, e.car_no, e.player_id,
                       e.rank_class, e.rating, e.gear_ratio, e.style,
                       e.line_id, e.line_pos, e.scratched,
                       e.nige_cnt, e.makuri_cnt, e.sashi_cnt, e.mark_cnt,
                       e.s_count, e.b_count,
                       r.grade, r.distance_m, r.weather,
                       v.bank_length, v.bank_angle
                FROM entries e
                JOIN races r ON e.race_id = r.race_id
                LEFT JOIN venues v ON r.venue_id = v.venue_id
                WHERE e.race_id = :race_id
                ORDER BY e.car_no
                """
            ),
            conn,
            params={"race_id": race_id},
        )

    if entries.empty:
        log.warning("build_race_features: no entries for race_id=%s", race_id)
        return pd.DataFrame()

    venue_id = _venue_id_for_race(engine, race_id)

    # --- 2. Rank class one-hot -------------------------------------------------
    rank_dummies = pd.get_dummies(entries["rank_class"], prefix="rank").reindex(
        columns=["rank_S1", "rank_S2", "rank_A1", "rank_A2", "rank_A3"], fill_value=0
    )
    rank_dummies.columns = [c.lower() for c in rank_dummies.columns]
    entries = pd.concat([entries.reset_index(drop=True), rank_dummies.reset_index(drop=True)], axis=1)

    # --- 3. Grade one-hot ------------------------------------------------------
    grade_map = {"GP": "gp", "GⅠ": "g1", "GI": "g1", "GⅡ": "g2", "GII": "g2",
                 "GⅢ": "g3", "GIII": "g3", "F1": "f1", "F2": "f2"}
    for key, col in [("grade_gp", "gp"), ("grade_g1", "g1"), ("grade_g2", "g2"),
                     ("grade_g3", "g3"), ("grade_f1", "f1"), ("grade_f2", "f2")]:
        entries[key] = (entries["grade"].map(grade_map) == col).astype(int)

    # --- 4. Player form --------------------------------------------------------
    try:
        pf5 = player_form.rolling_finish_stats(engine, ref_date, n_races=5)[
            ["avg_finish", "top3_rate"]
        ].rename(columns={"avg_finish": "avg_finish_5", "top3_rate": "top3_rate_5"})
        pf10 = player_form.rolling_finish_stats(engine, ref_date, n_races=10)[
            ["top3_rate", "win_rate"]
        ].rename(columns={"top3_rate": "top3_rate_10", "win_rate": "win_rate_10"})
        pf20 = player_form.rolling_finish_stats(engine, ref_date, n_races=20)[
            ["top3_rate"]
        ].rename(columns={"top3_rate": "top3_rate_20"})
        trend = player_form.recent_trend(engine, ref_date)[["form_trend"]]
        rest = player_form.rest_days(engine, ref_date)
        km = player_form.kimarite_dist(engine, ref_date)
        lrr = player_form.last_race_result(engine, ref_date)
    except Exception as e:
        log.warning("player_form features failed: %s", e)
        pf5 = pf10 = pf20 = trend = rest = km = lrr = pd.DataFrame()

    # --- 5. Venue ---------------------------------------------------------------
    try:
        vs = venue.venue_specialist_score(engine, venue_id, ref_date) if venue_id else pd.DataFrame()
        bl = venue.bank_length_affinity(engine, ref_date)[["top3_333", "top3_400", "top3_500"]]
    except Exception as e:
        log.warning("venue features failed: %s", e)
        vs = bl = pd.DataFrame()

    # --- 6a. Matchup features (head-to-head vs. this field + grade form) -------
    try:
        pids = entries["player_id"].dropna().tolist()
        grade_val = (
            entries["grade"].dropna().iloc[0]
            if "grade" in entries.columns and not entries["grade"].dropna().empty
            else None
        )
        h2h = matchup.h2h_stats(engine, pids, ref_date)
        grd = matchup.grade_form(engine, pids, ref_date, grade_val)
    except Exception as e:
        log.warning("matchup features failed: %s", e)
        h2h = grd = pd.DataFrame()

    # --- 6. Line ----------------------------------------------------------------
    try:
        lf = line.line_features(engine, race_id)
        if not lf.empty:
            lf = lf.reset_index().rename(columns={"race_id": "_", "car_no": "car_no_line"})
            lf = lf.drop(columns=["_", "player_id", "rating", "style", "scratched",
                                   "line_id", "line_pos"], errors="ignore")
            lf = lf.set_index("car_no_line")
    except Exception as e:
        log.warning("line features failed: %s", e)
        lf = pd.DataFrame()

    # --- 7. Gear ----------------------------------------------------------------
    try:
        gf = gear.gear_features(engine, race_id).reset_index()[["car_no", "gear_z", "gear_rank", "gear_above_field"]]
        gf = gf.set_index("car_no")
    except Exception as e:
        log.warning("gear features failed: %s", e)
        gf = pd.DataFrame()

    # --- 8. Join all player-level frames onto entries --------------------------
    entries = entries.set_index("player_id")
    for frame in [pf5, pf10, pf20, trend, rest, km, vs, bl, h2h, grd, lrr]:
        if not frame.empty:
            entries = entries.join(frame, how="left")
    entries = entries.reset_index().rename(columns={"index": "player_id"})

    # Car-no level joins
    entries = entries.set_index("car_no")
    for frame in [lf, gf]:
        if not frame.empty:
            entries = entries.join(frame, how="left")
    entries = entries.reset_index().rename(columns={"index": "car_no"})

    # If no line frame was joined at all, this race has no 並び data.
    if "has_line" not in entries.columns:
        entries["has_line"] = 0
    else:
        entries["has_line"] = entries["has_line"].fillna(0).astype(int)

    # --- 6b. Style matchup (レース内の脚質構成) ---------------------------------
    try:
        sc = race_style_context(entries)
        entries = entries.set_index("car_no").join(sc, how="left")
        entries = entries.reset_index().rename(columns={"index": "car_no"})
    except Exception as e:
        log.warning("style matchup features failed: %s", e)
        for col in STYLE_CONTEXT_COLUMNS:
            if col not in entries.columns:
                entries[col] = float("nan")

    # --- 6c. Simulation features (ライン力学・展開シミュレーション) -------------
    entries = _join_sim_features(engine, race_id, entries)

    # --- 8b. Field-relative rating (computed from entries, no SQL needed) ------
    # Use ONLY observed ratings for the field statistics. A car with a missing
    # rating must not be imputed to the field mean (that would distort both its
    # own relative_rating and every opponent's denominator). Its relative_rating
    # is left NaN — LightGBM handles NaN natively.
    if "rating" in entries.columns and entries["rating"].notna().sum() > 1:
        valid = entries["rating"].dropna()
        valid_sum = valid.sum()
        m = len(valid)
        has_rating = entries["rating"].notna()
        # Cars with a rating: average of the OTHER observed ratings (exclude self).
        # Cars without a rating: average of all observed ratings (nothing to exclude).
        avg_opp = entries["rating"].where(has_rating).copy()
        avg_opp[has_rating] = (valid_sum - entries.loc[has_rating, "rating"]) / (m - 1)
        avg_opp[~has_rating] = valid_sum / m
        entries["avg_opponent_rating"] = avg_opp
        entries["relative_rating"] = entries["rating"] - avg_opp
    else:
        entries["avg_opponent_rating"] = float("nan")
        entries["relative_rating"] = float("nan")

    # --- 9. Divergence signals --------------------------------------------------
    if latest_odds is not None and not entries.empty:
        implied = _implied_probs(latest_odds)
        entries = _add_divergence(entries, implied, engine, ref_date, venue_id)
    else:
        for col in ["score_odds_gap", "rest_day_signal", "venue_specialist_discount",
                    "form_rebound_signal", "line_mismatch_signal", "divergence_score"]:
            entries[col] = float("nan")

    # --- 10. Add target if results exist ----------------------------------------
    try:
        entries = _add_target(engine, entries)
    except Exception:
        entries[TARGET] = None

    entries["race_id"] = race_id
    return entries


# ---------------------------------------------------------------------------
# Training frame builder (iterate over date range)
# ---------------------------------------------------------------------------

def build_training_frame(
    engine: Engine,
    from_date: str,
    to_date: str,
) -> pd.DataFrame:
    """Build training data by iterating over all races in the date range."""
    with engine.begin() as conn:
        race_rows = pd.read_sql(
            text(
                """
                SELECT race_id, date FROM races
                WHERE date >= :from_date AND date <= :to_date
                ORDER BY date
                """
            ),
            conn,
            params={"from_date": from_date, "to_date": to_date},
        ).to_dict("records")

    chunks: list[pd.DataFrame] = []
    n = len(race_rows)
    for i, row in enumerate(race_rows, 1):
        if i % 100 == 0:
            log.info("build_training_frame: %d / %d races", i, n)
        df = build_race_features(engine, row["race_id"], ref_date=row["date"])
        if not df.empty:
            chunks.append(df)

    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SIM_COLS = ["p_sim_win", "p_sim_top3", "p_front"]


def _join_sim_features(engine: Engine, race_id: str, entries: pd.DataFrame) -> pd.DataFrame:
    """sim_features キャッシュを結合。無ければライン付きレースに限りその場で計算。

    ライン情報がない (シミュレーション不可) レースは NaN + sim_used=0 のままにし、
    LightGBM のネイティブ NaN 処理でフォールバックさせる。
    """
    sim_df = pd.DataFrame()
    try:
        with engine.begin() as conn:
            sim_df = pd.read_sql(
                text(
                    "SELECT car_no, p_sim_win, p_sim_top3, p_front"
                    " FROM sim_features WHERE race_id = :rid"
                ),
                conn,
                params={"rid": race_id},
            )
    except Exception as e:
        log.warning("sim_features cache read failed: %s", e)

    if sim_df.empty and entries["line_id"].notna().sum() >= 2:
        try:
            from keirin.simulation import simulate_race
            active = entries[entries["scratched"].fillna(0) == 0]
            sim = simulate_race(active)
            if sim is not None:
                sim_df = pd.DataFrame({
                    "car_no": list(sim.p_win.keys()),
                    "p_sim_win": list(sim.p_win.values()),
                    "p_sim_top3": [sim.p_top3.get(c) for c in sim.p_win],
                    "p_front": [sim.p_front.get(c) for c in sim.p_win],
                })
        except Exception as e:
            log.warning("on-the-fly simulation failed for %s: %s", race_id, e)

    if not sim_df.empty:
        entries = entries.merge(sim_df, on="car_no", how="left")
        entries["sim_used"] = entries["p_sim_win"].notna().astype(int)
    else:
        for col in _SIM_COLS:
            entries[col] = float("nan")
        entries["sim_used"] = 0
    return entries


def _venue_id_for_race(engine: Engine, race_id: str) -> str | None:
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT venue_id FROM races WHERE race_id = :rid"),
            {"rid": race_id},
        ).fetchone()
    return row[0] if row else None


def _implied_probs(trifecta_odds: dict[str, float]) -> dict[int, float]:
    """Estimate per-car implied probability from trifecta odds.

    Simple heuristic: for each car, sum 1/odds over all combos where it
    appears in position 1 or 2, normalised to reflect top-2 probability,
    then translate to top-3.
    """
    car_inv: dict[int, float] = {}
    for combo, odds in trifecta_odds.items():
        parts = combo.split("-")
        if len(parts) != 3:
            continue
        for pos, c_str in enumerate(parts):
            try:
                c = int(c_str)
            except ValueError:
                continue
            weight = 1.0 / (pos + 1)  # position 1 gets full weight, 2 half, 3 third
            car_inv[c] = car_inv.get(c, 0.0) + (1.0 / odds) * weight

    total = sum(car_inv.values()) or 1.0
    return {c: v / total for c, v in car_inv.items()}


def _add_divergence(
    entries: pd.DataFrame,
    implied: dict[int, float],
    engine: Engine,
    ref_date: str,
    venue_id: str | None,
) -> pd.DataFrame:
    entries = entries.copy()
    entries["implied_prob"] = entries["car_no"].map(implied).fillna(0.0)
    entries = odds_divergence.score_implied_rank(entries)

    rest = player_form.rest_days(engine, ref_date)
    if not rest.empty:
        entries = entries.set_index("player_id")
        entries = entries.join(rest, how="left")
        entries = entries.reset_index().rename(columns={"index": "player_id"})
    if "rest_days" not in entries.columns:
        entries["rest_days"] = float("nan")

    entries["rest_day_signal"] = odds_divergence.rest_day_signal(
        entries["rest_days"].fillna(0)
    )

    if venue_id:
        vspec = venue.venue_specialist_score(engine, venue_id, ref_date)
        if not vspec.empty:
            entries = entries.set_index("player_id")
            entries = entries.join(vspec[["venue_specialist_score"]], how="left")
            entries = entries.reset_index().rename(columns={"index": "player_id"})
    if "venue_specialist_score" not in entries.columns:
        entries["venue_specialist_score"] = 0.0
    entries["venue_specialist_discount"] = odds_divergence.venue_specialist_discount(
        entries["venue_specialist_score"].fillna(0)
    )

    lf_col = entries.get("is_follower", pd.Series(0, index=entries.index))
    gap_col = entries.get("rating_vs_leader_gap", pd.Series(0, index=entries.index))
    entries["line_mismatch_signal"] = odds_divergence.line_mismatch_signal(
        lf_col.fillna(0), gap_col.fillna(0)
    )

    last = odds_divergence.last_finish(engine, ref_date)
    if not last.empty:
        entries = entries.set_index("player_id")
        entries = entries.join(last.rename("last_finish"), how="left")
        entries = entries.reset_index().rename(columns={"index": "player_id"})
    if "last_finish" not in entries.columns:
        entries["last_finish"] = float("nan")
    entries["form_rebound_signal"] = odds_divergence.form_rebound_signal(
        entries["last_finish"].fillna(99),
        entries.get("avg_finish_5", pd.Series(99, index=entries.index)).fillna(99),
    )

    entries["divergence_score"] = odds_divergence.composite_divergence(entries)

    return entries


def _add_target(engine: Engine, entries: pd.DataFrame) -> pd.DataFrame:
    race_id = entries["race_id"].iloc[0]
    with engine.begin() as conn:
        results = pd.read_sql(
            text("SELECT car_no, finish FROM results WHERE race_id = :rid"),
            conn,
            params={"rid": race_id},
        )
    if results.empty:
        entries[TARGET] = None
        entries["finish"] = None
        return entries
    # top3=1 for finishing cars, 0 for everyone else (all scratched cars too = 0).
    # Merge so unmatched entries (finish > 3 or no result row) get NaN, then fill 0.
    results[TARGET] = (results["finish"] <= 3).astype(int)
    merged = entries.merge(results[["car_no", TARGET, "finish"]], on="car_no", how="left")
    # Cars that raced but didn't place top-3 get 0; truly missing data stays NaN.
    has_any_result = not results.empty
    if has_any_result:
        merged[TARGET] = merged[TARGET].fillna(0).astype(int)
        # 着外導出: the race has results, so a non-scratched car with no result
        # row finished 4th or worse. Fill the sentinel so the ranker sees these
        # rows as negatives instead of dropping them (finish=9 → relevance 0).
        # Scratched cars keep NaN finish and stay out of ranker training.
        if "scratched" in merged.columns:
            outside = merged["finish"].isna() & (merged["scratched"].fillna(0) == 0)
        else:
            outside = merged["finish"].isna()
        merged.loc[outside, "finish"] = 9
    return merged
