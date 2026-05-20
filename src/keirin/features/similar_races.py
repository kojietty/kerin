"""Similar race lookup: find historical races structurally similar to a target race.

Similarity is computed as cosine similarity over a z-score-normalised fingerprint
vector extracted from race-level aggregations (grade, bank length, field strength,
line structure). No need to run the full 48-feature pipeline on every historical race.
"""
from __future__ import annotations

import logging
import math

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

log = logging.getLogger(__name__)

_GRADE_NUMERIC = {
    "GP": 6, "GⅠ": 5, "GI": 5, "GⅡ": 4, "GII": 4,
    "GⅢ": 3, "GIII": 3, "F1": 2, "F2": 1,
}
_GRADE_MAP = {
    "GP": "gp", "GⅠ": "g1", "GI": "g1", "GⅡ": "g2", "GII": "g2",
    "GⅢ": "g3", "GIII": "g3", "F1": "f1", "F2": "f2",
}

# Features used for cosine similarity (excludes mismatch/divergence not in SQL agg)
_SIM_KEYS = [
    "bank_length", "grade_numeric", "avg_rating", "max_rating",
    "rating_spread", "n_lines", "max_line_len",
    "grade_gp", "grade_g1", "grade_g2", "grade_g3", "grade_f1", "grade_f2",
]


def race_fingerprint(engine: Engine, race_id: str, df: pd.DataFrame) -> dict[str, float]:
    """Extract a compact numeric fingerprint from the already-computed features DataFrame.

    Uses df (output of build_race_features) passed by the caller — no extra DB query.
    Returns a dict keyed by _SIM_KEYS.
    """
    if df.empty:
        return {}

    row0 = df.iloc[0]

    def _col(name: str, default: float = 0.0) -> float:
        if name in df.columns:
            v = row0[name]
            return float(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else default
        return default

    ratings = df["rating"].dropna() if "rating" in df.columns else pd.Series(dtype=float)
    avg_rating = float(ratings.mean()) if not ratings.empty else 75.0
    max_rating = float(ratings.max()) if not ratings.empty else 75.0
    rating_spread = float(ratings.max() - ratings.min()) if len(ratings) > 1 else 0.0

    n_lines = 1
    max_line_len = 1
    if "line_id" in df.columns:
        valid = df[df["line_id"].notna()]
        if not valid.empty:
            try:
                n_lines = int(valid["line_id"].nunique())
                max_line_len = int(valid.groupby("line_id")["car_no"].count().max())
            except Exception:
                pass

    grade_raw = str(row0["grade"]) if "grade" in df.columns and row0.get("grade") is not None else ""
    grade_key = _GRADE_MAP.get(grade_raw, "")

    return {
        "bank_length": _col("bank_length", 400.0),
        "grade_numeric": float(_GRADE_NUMERIC.get(grade_raw, 2)),
        "avg_rating": avg_rating,
        "max_rating": max_rating,
        "rating_spread": rating_spread,
        "n_lines": float(n_lines),
        "max_line_len": float(max_line_len),
        "grade_gp": 1.0 if grade_key == "gp" else 0.0,
        "grade_g1": 1.0 if grade_key == "g1" else 0.0,
        "grade_g2": 1.0 if grade_key == "g2" else 0.0,
        "grade_g3": 1.0 if grade_key == "g3" else 0.0,
        "grade_f1": 1.0 if grade_key == "f1" else 0.0,
        "grade_f2": 1.0 if grade_key == "f2" else 0.0,
    }


def find_similar_races(
    engine: Engine,
    target_race_id: str,
    target_fp: dict[str, float],
    *,
    n: int = 5,
    lookback_days: int = 365,
    min_similarity: float = 0.50,
) -> list[dict]:
    """Find N most historically similar races to target_race_id.

    Returns list of dicts sorted by similarity_score descending with result info.
    Returns empty list if insufficient history (< 20 races).
    """
    if not target_fp:
        return []

    ref_date = f"{target_race_id[0:4]}-{target_race_id[4:6]}-{target_race_id[6:8]}"
    from datetime import date, timedelta
    try:
        from_date = (date.fromisoformat(ref_date) - timedelta(days=lookback_days)).isoformat()
    except ValueError:
        return []

    with engine.begin() as conn:
        hist_df = pd.read_sql(
            text(
                """
                WITH line_sizes AS (
                    SELECT race_id, line_id, COUNT(*) AS llen
                    FROM entries
                    WHERE scratched = 0 AND line_id IS NOT NULL
                    GROUP BY race_id, line_id
                ),
                race_line_stats AS (
                    SELECT race_id,
                           COUNT(DISTINCT line_id) AS n_lines,
                           MAX(llen)               AS max_line_len
                    FROM line_sizes GROUP BY race_id
                )
                SELECT r.race_id, r.date, r.grade, r.race_no,
                       v.name       AS venue_name,
                       v.bank_length,
                       AVG(e.rating)                   AS avg_rating,
                       MAX(e.rating)                   AS max_rating,
                       (MAX(e.rating) - MIN(e.rating)) AS rating_spread,
                       COALESCE(rls.n_lines, 1)        AS n_lines,
                       COALESCE(rls.max_line_len, 1)   AS max_line_len
                FROM races r
                JOIN entries e   ON r.race_id  = e.race_id
                JOIN venues v    ON r.venue_id  = v.venue_id
                LEFT JOIN race_line_stats rls ON rls.race_id = r.race_id
                WHERE r.date >= :from_date
                  AND r.date <  :ref_date
                  AND r.race_id != :target_id
                  AND e.scratched = 0
                GROUP BY r.race_id, r.date, r.grade, r.race_no,
                         v.name, v.bank_length, rls.n_lines, rls.max_line_len
                ORDER BY r.date DESC
                LIMIT 2000
                """
            ),
            conn,
            params={"from_date": from_date, "ref_date": ref_date, "target_id": target_race_id},
        )

    if hist_df.empty or len(hist_df) < 20:
        log.info("find_similar_races: only %d historical races found — skipping", len(hist_df))
        return []

    # Build fingerprints for historical races
    hist_fps: list[dict[str, float]] = []
    for _, row in hist_df.iterrows():
        grade_raw = str(row.get("grade") or "")
        grade_key = _GRADE_MAP.get(grade_raw, "")
        hist_fps.append({
            "bank_length": float(row["bank_length"] or 400),
            "grade_numeric": float(_GRADE_NUMERIC.get(grade_raw, 2)),
            "avg_rating": float(row["avg_rating"] or 75),
            "max_rating": float(row["max_rating"] or 75),
            "rating_spread": float(row["rating_spread"] or 0),
            "n_lines": float(row["n_lines"] or 1),
            "max_line_len": float(row["max_line_len"] or 1),
            "grade_gp": 1.0 if grade_key == "gp" else 0.0,
            "grade_g1": 1.0 if grade_key == "g1" else 0.0,
            "grade_g2": 1.0 if grade_key == "g2" else 0.0,
            "grade_g3": 1.0 if grade_key == "g3" else 0.0,
            "grade_f1": 1.0 if grade_key == "f1" else 0.0,
            "grade_f2": 1.0 if grade_key == "f2" else 0.0,
        })

    # Z-score normalise over all historical races + target together
    all_fps = hist_fps + [target_fp]
    fp_df = pd.DataFrame(all_fps, columns=_SIM_KEYS)
    mean = fp_df.mean()
    std = fp_df.std().replace(0.0, 1.0)
    fp_norm = (fp_df - mean) / std

    target_norm = fp_norm.iloc[-1].to_dict()

    # Compute cosine similarity for each historical race
    race_ids_needed: list[str] = []
    scored: list[tuple[float, int]] = []  # (similarity, row_index)
    for idx in range(len(hist_fps)):
        hist_norm = fp_norm.iloc[idx].to_dict()
        sim = _cosine_similarity(target_norm, hist_norm)
        if sim >= min_similarity:
            scored.append((sim, idx))

    scored.sort(key=lambda x: -x[0])
    top_indices = [idx for _, idx in scored[:n]]

    if not top_indices:
        return []

    race_ids_needed = [hist_df.iloc[i]["race_id"] for i in top_indices]

    # Fetch results for those races
    in_clause = ",".join([f":rid{i}" for i in range(len(race_ids_needed))])
    with engine.begin() as conn:
        results = pd.read_sql(
            text(
                f"""
                SELECT race_id, car_no, finish, kimarite
                FROM results
                WHERE race_id IN ({in_clause}) AND finish IN (1, 2, 3)
                ORDER BY race_id, finish
                """
            ),
            conn,
            params={f"rid{i}": rid for i, rid in enumerate(race_ids_needed)},
        )

    results_map: dict[str, dict] = {}
    for rid in race_ids_needed:
        r = results[results["race_id"] == rid].sort_values("finish")
        results_map[rid] = {
            "result": r["car_no"].tolist(),
            "winner_kimarite": r.iloc[0]["kimarite"] if not r.empty else None,
        }

    output = []
    for sim, idx in scored[:n]:
        row = hist_df.iloc[idx]
        rid = row["race_id"]
        res = results_map.get(rid, {})
        output.append({
            "race_id": rid,
            "date": row["date"],
            "venue_name": row["venue_name"],
            "race_no": int(row["race_no"]),
            "grade": row["grade"],
            "similarity_score": round(sim, 3),
            "result": res.get("result", []),
            "winner_style": res.get("winner_kimarite"),
        })

    return output


def _cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    keys = [k for k in _SIM_KEYS if k in a and k in b]
    if not keys:
        return 0.0
    dot = sum(a[k] * b[k] for k in keys)
    norm_a = math.sqrt(sum(a[k] ** 2 for k in keys))
    norm_b = math.sqrt(sum(b[k] ** 2 for k in keys))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
