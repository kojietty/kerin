"""Inference: load model+calibrator, output p_top3 per car for a race."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from keirin.models.calibrate import apply_calibration

log = logging.getLogger(__name__)


def predict_race(
    race_features_df: pd.DataFrame,
    model,
    feature_cols: list[str],
    calibrator,
) -> dict[int, float]:
    """Return {car_no: calibrated_p_top3} for every car in the race."""
    if race_features_df.empty:
        return {}

    X = race_features_df.reindex(columns=feature_cols, fill_value=np.nan).values.astype(np.float32)
    raw = model.predict_proba(X)[:, 1]
    cal = apply_calibration(raw, calibrator)

    car_nos = race_features_df["car_no"].tolist()
    return dict(zip(car_nos, cal.tolist()))


def predict_today(
    engine,
    race_ids: list[str],
    models_dir: Path,
    *,
    latest_odds_by_race: dict[str, dict[str, float]] | None = None,
) -> list[dict]:
    """Run full prediction pipeline for a list of race_ids.

    Returns list of prediction dicts (one per race) suitable for the dashboard renderer.
    """
    from keirin.models.train import load_latest_model
    from keirin.models.combo import expand_trifecta
    from keirin.betting.ev import build_candidates
    from keirin.betting.selector import select_picks
    from keirin.betting.stake import plan_stakes
    from keirin.features.builder import build_race_features
    from keirin.config import load_betting_config, load_app_config
    from keirin.db.repository import month_pnl
    from datetime import datetime

    bundle = load_latest_model(models_dir)
    if bundle is None:
        raise RuntimeError("No trained model found — run `python -m keirin retrain` first")
    model, feature_cols, calibrator = bundle

    bcfg = load_betting_config()

    # Month-to-date PnL for emergency stop check
    ym = datetime.now().strftime("%Y-%m")
    pnl_data = month_pnl(engine, ym)
    mtd_pnl = pnl_data.get("pnl", 0)

    recommended: list[dict] = []
    skipped: list[dict] = []

    for race_id in race_ids:
        date_iso = f"{race_id[0:4]}-{race_id[4:6]}-{race_id[6:8]}"
        venue_id = race_id[8:10]
        race_no = int(race_id[10:12])

        odds = (latest_odds_by_race or {}).get(race_id)

        df = build_race_features(engine, race_id, ref_date=date_iso, latest_odds=odds)
        if df.empty:
            log.warning("predict: no features for race_id=%s", race_id)
            continue

        p_top3 = predict_race(df, model, feature_cols, calibrator)
        if not p_top3:
            continue

        combo_probs = expand_trifecta(p_top3, method=bcfg.combo_method)

        if odds:
            from keirin.betting.ev import build_candidates
            candidates = build_candidates(combo_probs, odds)
            result = select_picks(candidates, bcfg)
            plan = plan_stakes(result.picks, bcfg, month_to_date_pnl_yen=mtd_pnl)

            # Venue info
            from keirin.data.import_cache import VENUE_TABLE
            venue_name = VENUE_TABLE.get(venue_id, ("?", 400))[0]

            if result.skipped:
                skipped.append({
                    "venue_name": venue_name,
                    "race_no": race_no,
                    "max_ev": result.max_ev,
                })
            else:
                picks_out = []
                for assign in plan.assignments:
                    p = assign.pick
                    cars = [int(x) for x in p.combo.split("-")]
                    picks_out.append({
                        "cars": cars,
                        "odds": p.odds,
                        "prob": p.prob,
                        "ev": p.ev,
                        "stake_yen": assign.stake_yen if not plan.emergency_stop_triggered else 0,
                        "hit": None,
                    })
                # Line info from features df
                line_info = _build_line_display(df)
                axis = _best_axis(df)
                recommended.append({
                    "venue_name": venue_name,
                    "race_no": race_no,
                    "grade": df.get("grade", pd.Series([None])).iloc[0] if "grade" in df.columns else None,
                    "race_class": None,
                    "post_time": None,
                    "distance_m": int(df["bank_length"].iloc[0]) if "bank_length" in df.columns and pd.notna(df["bank_length"].iloc[0]) else None,
                    "weather": None,
                    "lines": line_info,
                    "axis": axis,
                    "picks": picks_out,
                    "max_ev": result.max_ev,
                    "total_stake": plan.total_yen,
                })
        else:
            # No odds yet — just compute top-probability combos for preview
            from keirin.data.import_cache import VENUE_TABLE
            venue_name = VENUE_TABLE.get(venue_id, ("?", 400))[0]
            top_combos = sorted(combo_probs.items(), key=lambda x: -x[1])[:5]
            skipped.append({
                "venue_name": venue_name,
                "race_no": race_no,
                "max_ev": None,
                "top_combos": top_combos,
            })

    return {
        "recommended": recommended,
        "skipped": skipped,
        "mtd_pnl": mtd_pnl,
        "mtd_roi": pnl_data.get("roi"),
        "mtd_hit_rate": pnl_data.get("hit_rate"),
    }


def rank_prediction(
    p_top3: dict[int, float],
    *,
    rank_scores: dict[int, float] | None = None,
) -> list[dict]:
    """Derive predicted finish order with Plackett-Luce probabilities.

    If `rank_scores` (LambdaRank ranker output) is provided, those scores
    are converted to PL strengths — this directly optimises ranking quality.
    Otherwise falls back to deriving strengths from p_top3 via -log(1-p).
    Returns list sorted by expected rank ascending (best prediction first).
    """
    import math
    from itertools import permutations

    cars = list(p_top3.keys()) if not rank_scores else list(rank_scores.keys())
    if not cars:
        return []

    if rank_scores:
        # Ranker outputs continuous scores; convert to PL strengths via exp.
        # Subtract max for numerical stability (softmax-style).
        s_max = max(rank_scores.get(c, 0.0) for c in cars)
        strengths = {c: math.exp(rank_scores.get(c, 0.0) - s_max) for c in cars}
    else:
        probs = {c: max(min(p_top3[c], 0.9999), 0.0001) for c in cars}
        strengths = {c: -math.log(1.0 - probs[c]) for c in cars}
    total = sum(strengths.values())

    p1: dict[int, float] = {c: 0.0 for c in cars}
    p2: dict[int, float] = {c: 0.0 for c in cars}
    p3: dict[int, float] = {c: 0.0 for c in cars}

    for i, j, k in permutations(cars, 3):
        s_i, s_j, s_k = strengths[i], strengths[j], strengths[k]
        d2 = total - s_i
        d3 = d2 - s_j
        if d2 <= 0 or d3 <= 0:
            continue
        prob = (s_i / total) * (s_j / d2) * (s_k / d3)
        p1[i] += prob
        p2[j] += prob
        p3[k] += prob

    n = len(cars)
    staging = []
    for c in cars:
        remainder = max(0.0, 1.0 - p1[c] - p2[c] - p3[c])
        avg_rest = (4 + n) / 2 if n > 3 else 4.0
        expected = 1 * p1[c] + 2 * p2[c] + 3 * p3[c] + avg_rest * remainder
        display = {
            "car_no": c,
            "pred_prob_1st": round(p1[c], 4),
            "pred_prob_2nd": round(p2[c], 4),
            "pred_prob_3rd": round(p3[c], 4),
            "expected_rank": round(expected, 3),
        }
        # Sort on FULL-precision expected rank, then break ties by descending
        # 1st/2nd/3rd probability — never by insertion order (which is car_no
        # ascending and would systematically favour low car numbers).
        sort_key = (expected, -p1[c], -p2[c], -p3[c], c)
        staging.append((sort_key, display))

    staging.sort(key=lambda t: t[0])
    return [display for _, display in staging]


def analyze_race(
    engine,
    race_id: str,
    df: pd.DataFrame,
    model,
    feature_cols: list[str],
    calibrator,
    *,
    ranker=None,
    ranker_features: list[str] | None = None,
    similar_n: int = 5,
    lookback_days: int = 365,
    model_version: str = "unknown",
) -> dict:
    """Full single-race analysis focused on ranking prediction.

    Pure prediction pipeline — no betting/EV/Kelly logic. The user decides
    whether to bet based on the ranking output.

    If `ranker` (LambdaRank model) is provided, ranking probabilities are
    derived from its scores. Otherwise falls back to Plackett-Luce expansion
    of the binary top-3 classifier output.

    Returns a structured dict with participants, lines, similar races,
    and ranking prediction.
    """
    from sqlalchemy import text as _text
    from keirin.features.similar_races import race_fingerprint, find_similar_races
    from keirin.features.line import mismatch_score

    date_iso = f"{race_id[0:4]}-{race_id[4:6]}-{race_id[6:8]}"

    # --- Race metadata ---
    with engine.begin() as conn:
        meta_row = conn.execute(
            _text(
                """
                SELECT r.date, r.race_no, r.grade, r.distance_m,
                       v.name AS venue_name, v.bank_length
                FROM races r
                JOIN venues v ON r.venue_id = v.venue_id
                WHERE r.race_id = :rid
                """
            ),
            {"rid": race_id},
        ).fetchone()

    meta: dict = {}
    if meta_row:
        meta = {
            "date": meta_row[0],
            "race_no": meta_row[1],
            "grade": meta_row[2],
            "distance_m": meta_row[3],
            "venue_name": meta_row[4],
            "bank_length": meta_row[5],
        }

    # --- Player names ---
    player_ids = df["player_id"].dropna().tolist() if "player_id" in df.columns else []
    player_names: dict[str, str] = {}
    if player_ids:
        in_clause = ",".join([f":pn{i}" for i in range(len(player_ids))])
        with engine.begin() as conn:
            rows = conn.execute(
                _text(f"SELECT player_id, name FROM players WHERE player_id IN ({in_clause})"),
                {f"pn{i}": pid for i, pid in enumerate(player_ids)},
            ).fetchall()
        player_names = {r[0]: r[1] for r in rows}

    # --- Prediction ---
    p_top3 = predict_race(df, model, feature_cols, calibrator)

    # If a LambdaRank ranker is available, use its scores as PL strengths
    # (directly optimises ranking metrics, more accurate than PL-from-binary).
    rank_scores: dict[int, float] | None = None
    if ranker is not None and ranker_features and not df.empty:
        try:
            X_rank = df.reindex(columns=ranker_features, fill_value=np.nan).values.astype(np.float32)
            scores = ranker.predict(X_rank)
            rank_scores = dict(zip(df["car_no"].tolist(), [float(s) for s in scores]))
        except Exception:
            log.debug("ranker.predict failed", exc_info=True)

    ranking = rank_prediction(p_top3, rank_scores=rank_scores)

    # Attach player names to ranking
    pid_by_car: dict[int, str] = {}
    if "player_id" in df.columns and "car_no" in df.columns:
        pid_by_car = dict(zip(df["car_no"].tolist(), df["player_id"].tolist()))
    for r in ranking:
        pid = pid_by_car.get(r["car_no"], "")
        r["name"] = player_names.get(pid, "?")

    # --- Participants ---
    def _safe(row, col, default=None):
        v = row.get(col) if isinstance(row, dict) else getattr(row, col, None)
        return v if v is not None and (not isinstance(v, float) or not np.isnan(v)) else default

    participants = []
    for _, row in df.iterrows():
        row_d = row.to_dict()
        car_no = int(row_d.get("car_no", 0))
        pid = str(row_d.get("player_id", ""))
        participants.append({
            "car_no": car_no,
            "player_id": pid,
            "name": player_names.get(pid, "?"),
            "rank_class": _safe(row_d, "rank_class"),
            "rating": _safe(row_d, "rating"),
            "style": _safe(row_d, "style"),
            "line_id": _safe(row_d, "line_id"),
            "line_pos": _safe(row_d, "line_pos"),
            "is_leader": bool(int(_safe(row_d, "is_leader") or 0)),
            "is_follower": bool(int(_safe(row_d, "is_follower") or 0)),
            "is_solo": bool(int(_safe(row_d, "is_solo") or 0)),
            "gear_ratio": _safe(row_d, "gear_ratio"),
            "gear_z": _safe(row_d, "gear_z"),
            "top3_rate_5": _safe(row_d, "top3_rate_5"),
            "top3_rate_10": _safe(row_d, "top3_rate_10"),
            "avg_finish_5": _safe(row_d, "avg_finish_5"),
            "rest_days": _safe(row_d, "rest_days"),
            "venue_specialist_score": _safe(row_d, "venue_specialist_score"),
            "divergence_score": _safe(row_d, "divergence_score"),
            "km_nige": _safe(row_d, "km_nige"),
            "km_sashi": _safe(row_d, "km_sashi"),
            "km_maki": _safe(row_d, "km_maki"),
            "km_mark": _safe(row_d, "km_mark"),
        })

    # --- Line analysis ---
    lines_out: list[dict] = []
    if "line_id" in df.columns:
        ms_series = mismatch_score(df)
        groups: dict[int, list[tuple[int, int]]] = {}
        for _, row in df.iterrows():
            lid = row.get("line_id")
            lpos = row.get("line_pos")
            cn = int(row.get("car_no", 0))
            if lid is not None and not (isinstance(lid, float) and np.isnan(lid)):
                groups.setdefault(int(lid), []).append((int(lpos or 0), cn))

        for lid in sorted(groups):
            cars_sorted = [cn for _, cn in sorted(groups[lid])]
            line_df = df[df["line_id"] == lid] if "line_id" in df.columns else pd.DataFrame()
            ms_max = float(ms_series[line_df.index].max()) if not line_df.empty else 0.0
            leader_r = float(line_df[line_df["line_pos"] == 1]["rating"].iloc[0]) if not line_df.empty and (line_df["line_pos"] == 1).any() else 0.0
            max_follower_r = float(line_df[line_df["line_pos"] > 1]["rating"].max()) if not line_df.empty and (line_df["line_pos"] > 1).any() else 0.0
            note = "実力番手の可能性あり" if ms_max > 3.0 else ("番手有利ライン" if ms_max > 1.0 else "")
            lines_out.append({
                "line_id": lid,
                "cars": cars_sorted,
                "leader_rating": leader_r,
                "max_follower_rating": max_follower_r,
                "mismatch_score": round(ms_max, 2),
                "interpretation": note,
            })

    # --- Similar races ---
    similar: list[dict] = []
    try:
        fp = race_fingerprint(engine, race_id, df)
        if fp:
            similar = find_similar_races(
                engine, race_id, fp,
                n=similar_n,
                lookback_days=lookback_days,
            )
    except Exception:
        log.debug("similar_races lookup failed", exc_info=True)

    return {
        "race_id": race_id,
        "date": meta.get("date", date_iso),
        "venue_name": meta.get("venue_name", "?"),
        "race_no": meta.get("race_no"),
        "grade": meta.get("grade"),
        "bank_length": meta.get("bank_length"),
        "distance_m": meta.get("distance_m"),
        "model_version": model_version,
        "used_ranker": rank_scores is not None,
        "participants": participants,
        "lines": lines_out,
        "similar_races": similar,
        "ranking": ranking,
    }


def _build_line_display(df: pd.DataFrame) -> list[list[int]]:
    if "line_id" not in df.columns:
        return []
    groups: dict[int, list[tuple[int, int]]] = {}
    for _, row in df.iterrows():
        lid = row.get("line_id")
        lpos = row.get("line_pos")
        cn = row.get("car_no")
        if pd.notna(lid) and pd.notna(lpos):
            groups.setdefault(int(lid), []).append((int(lpos), int(cn)))
    result = []
    for lid in sorted(groups):
        cars = [cn for _, cn in sorted(groups[lid])]
        result.append(cars)
    return result


def _best_axis(df: pd.DataFrame) -> dict | None:
    if "rating" not in df.columns or df.empty:
        return None
    top = df.loc[df["rating"].idxmax()]
    name = top.get("name", "?") if "name" in df.columns else "?"
    venue_score = top.get("venue_specialist_score", 0) or 0
    note = f"得点1位 / バンク適性{venue_score:+.2f}" if abs(venue_score) > 0.05 else "得点1位"
    return {
        "car_no": int(top["car_no"]),
        "name": str(name)[:6],
        "note": note,
    }
