"""Isotonic regression probability calibration.

LightGBM's raw probabilities are not well-calibrated (they tend to be
overconfident). Calibrated probabilities are essential for EV calculation:
  EV = calibrated_prob × odds
An uncalibrated model overestimates p → inflates EV → overfits to trash picks.

We use sklearn's IsotonicRegression fitted on a held-out chronological split.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression


def calibrate_isotonic(
    raw_probs: np.ndarray,
    y_true: np.ndarray,
) -> IsotonicRegression:
    """Fit an isotonic calibrator on (raw_probs, y_true). Returns fitted object."""
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(raw_probs, y_true)
    return cal


def fit_and_save_calibrator(
    raw_probs: np.ndarray,
    y_true: np.ndarray,
    out_path: Path,
) -> IsotonicRegression:
    cal = calibrate_isotonic(raw_probs, y_true)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(cal, out_path)
    return cal


def apply_calibration(raw_probs: np.ndarray, calibrator: IsotonicRegression | None) -> np.ndarray:
    if calibrator is None:
        return raw_probs
    return calibrator.predict(raw_probs)


def calibration_metrics(raw_probs: np.ndarray, cal_probs: np.ndarray, y_true: np.ndarray) -> dict:
    """Brier score and ECE (expected calibration error) before/after calibration."""
    from sklearn.metrics import brier_score_loss

    def ece(probs, y, n_bins=10):
        bins = np.linspace(0, 1, n_bins + 1)
        ece_val = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (probs >= lo) & (probs < hi)
            if mask.sum() == 0:
                continue
            acc = y[mask].mean()
            conf = probs[mask].mean()
            ece_val += mask.sum() * abs(acc - conf)
        return ece_val / len(y)

    return {
        "brier_raw": round(brier_score_loss(y_true, raw_probs), 4),
        "brier_cal": round(brier_score_loss(y_true, cal_probs), 4),
        "ece_raw": round(ece(raw_probs, y_true), 4),
        "ece_cal": round(ece(cal_probs, y_true), 4),
    }
