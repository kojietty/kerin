"""LightGBM top-3 binary classifier training with time-series walk-forward validation.

Model target: label_top3 (1 = car finishes in top 3, 0 = car doesn't).
One row per (race × car). Each race contributes 7-9 rows; 3 rows are label=1.

Walk-forward scheme:
  train on N months → validate on 1 month → repeat sliding forward.
  Group key = race_id (no leakage between train/val for same race).
"""
from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Features used during training (divergence signals excluded — need live odds)
TRAIN_FEATURES = [
    # Player ability
    "rating",
    "rank_s1", "rank_s2", "rank_a1", "rank_a2", "rank_a3",
    # Recent form
    "avg_finish_5", "top3_rate_5", "top3_rate_10", "top3_rate_20",
    "win_rate_10", "form_trend", "rest_days",
    # Kimarite (many will be NaN if not scraped — LightGBM handles it)
    "km_nige", "km_maki", "km_sashi", "km_mark",
    # Venue
    "bank_length", "venue_top3_rate", "venue_specialist_score",
    "top3_333", "top3_400", "top3_500",
    # Line (mostly NaN in imported data, but useful once scraped)
    "line_pos", "line_len", "is_leader", "is_follower", "is_solo",
    "rating_vs_leader_gap",
    # Gear
    "gear_ratio", "gear_z", "gear_rank", "gear_above_field",
    # Race context
    "grade_gp", "grade_g1", "grade_g2", "grade_g3", "grade_f1", "grade_f2",
    "car_no",
]


def train_model(
    df: pd.DataFrame,
    *,
    val_df: pd.DataFrame | None = None,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    min_child_samples: int = 30,
    feature_fraction: float = 0.8,
    bagging_fraction: float = 0.8,
    bagging_freq: int = 5,
    class_weight: str = "balanced",
) -> Any:
    """Train LightGBM binary model. Returns fitted model."""
    try:
        import lightgbm as lgb
    except ImportError:
        raise RuntimeError("lightgbm not installed — run: pip install lightgbm")

    from keirin.models.train import TRAIN_FEATURES

    feature_cols = [c for c in TRAIN_FEATURES if c in df.columns]
    X_train = df[feature_cols].values.astype(np.float32)
    y_train = df["label_top3"].values.astype(np.int32)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "min_child_samples": min_child_samples,
        "feature_fraction": feature_fraction,
        "bagging_fraction": bagging_fraction,
        "bagging_freq": bagging_freq,
        "is_unbalance": True,
        "verbose": -1,
        "n_jobs": -1,
        "random_state": 42,
    }

    model = lgb.LGBMClassifier(**params)

    if val_df is not None and not val_df.empty:
        val_feature_cols = [c for c in TRAIN_FEATURES if c in val_df.columns]
        X_val = val_df[val_feature_cols].reindex(columns=feature_cols, fill_value=np.nan).values.astype(np.float32)
        y_val = val_df["label_top3"].values.astype(np.int32)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
    else:
        model.fit(X_train, y_train)

    return model, feature_cols


def walk_forward_eval(
    df: pd.DataFrame,
    *,
    train_months: int = 2,
    val_months: int = 1,
) -> list[dict[str, Any]]:
    """Walk-forward validation. Returns per-fold metrics."""
    from sklearn.metrics import log_loss, roc_auc_score
    from keirin.models.calibrate import calibrate_isotonic

    df = df.copy()
    df["_month"] = pd.to_datetime(df["race_id"].str[:6], format="%Y%m").dt.to_period("M")
    months = sorted(df["_month"].unique())

    if len(months) < train_months + val_months:
        log.warning("walk_forward_eval: not enough months (%d)", len(months))
        return []

    results = []
    for i in range(train_months, len(months) - val_months + 1):
        train_periods = months[i - train_months: i]
        val_periods = months[i: i + val_months]

        train = df[df["_month"].isin(train_periods)].dropna(subset=["label_top3"])
        val = df[df["_month"].isin(val_periods)].dropna(subset=["label_top3"])

        if len(train) < 200 or len(val) < 50:
            continue

        model, feature_cols = train_model(train, val_df=val)
        X_val = val[feature_cols].values.astype(float)
        y_val = val["label_top3"].values.astype(int)

        raw_probs = model.predict_proba(X_val)[:, 1]
        calib = calibrate_isotonic(raw_probs, y_val)
        cal_probs = calib.predict(raw_probs)

        try:
            ll = log_loss(y_val, cal_probs)
            auc = roc_auc_score(y_val, cal_probs)
        except Exception:
            ll = auc = float("nan")

        results.append({
            "train_period": [str(p) for p in train_periods],
            "val_period": [str(p) for p in val_periods],
            "n_train": len(train),
            "n_val": len(val),
            "log_loss": round(ll, 4),
            "auc": round(auc, 4),
        })
        log.info(
            "fold val=%s  n_train=%d  n_val=%d  log_loss=%.4f  auc=%.4f",
            val_periods, len(train), len(val), ll, auc,
        )

    return results


def retrain_and_save(
    engine,
    models_dir: Path,
    *,
    from_date: str | None = None,
    to_date: str | None = None,
) -> Path:
    """Build full training frame, train model, calibrate, save artifacts.

    Returns path to saved model file.
    """
    from keirin.features.builder import build_training_frame
    from keirin.models.calibrate import fit_and_save_calibrator

    today = date.today().isoformat()
    to_date = to_date or today
    if from_date is None:
        # default: 90 days back
        from_date = (date.today() - timedelta(days=90)).isoformat()

    log.info("Building training frame %s → %s", from_date, to_date)
    df = build_training_frame(engine, from_date=from_date, to_date=to_date)

    if df.empty:
        raise ValueError("No training data found in date range")

    df_clean = df.dropna(subset=["label_top3"])
    log.info("Training rows: %d  (positive rate %.2f%%)",
             len(df_clean), 100 * df_clean["label_top3"].mean())

    # Chronological split: last 15% for calibration
    split_idx = int(len(df_clean) * 0.85)
    df_train = df_clean.iloc[:split_idx]
    df_cal = df_clean.iloc[split_idx:]

    model, feature_cols = train_model(df_train, val_df=df_cal)

    X_cal = df_cal.reindex(columns=feature_cols, fill_value=float("nan")).values.astype(float)
    y_cal = df_cal["label_top3"].values.astype(int)
    raw_probs = model.predict_proba(X_cal)[:, 1]

    models_dir.mkdir(parents=True, exist_ok=True)
    stamp = today.replace("-", "")
    model_path = models_dir / f"lgbm_top3_v{stamp}.pkl"
    joblib.dump({"model": model, "feature_cols": feature_cols}, model_path)
    log.info("Model saved to %s", model_path)

    calibrator = fit_and_save_calibrator(
        raw_probs, y_cal,
        out_path=models_dir / f"calibrator_v{stamp}.pkl",
    )

    # Save walk-forward metrics for reference
    try:
        wf = walk_forward_eval(df_clean)
        if wf:
            metrics_path = models_dir / f"wf_metrics_v{stamp}.json"
            metrics_path.write_text(json.dumps(wf, ensure_ascii=False, indent=2), encoding="utf-8")
            log.info("Walk-forward metrics: %s", wf[-1])
    except Exception as e:
        log.warning("walk_forward_eval failed: %s", e)

    return model_path


def load_latest_model(models_dir: Path) -> tuple[Any, list[str], Any] | None:
    """Load the most recent model + calibrator pair. Returns (model, feature_cols, calibrator)."""
    model_files = sorted(models_dir.glob("lgbm_top3_v*.pkl"), reverse=True)
    if not model_files:
        return None
    data = joblib.load(model_files[0])
    model = data["model"]
    feature_cols = data["feature_cols"]

    cal_files = sorted(models_dir.glob("calibrator_v*.pkl"), reverse=True)
    calibrator = joblib.load(cal_files[0]) if cal_files else None
    log.info("Loaded model %s  calibrator=%s", model_files[0].name,
             cal_files[0].name if cal_files else "None")
    return model, feature_cols, calibrator
