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
