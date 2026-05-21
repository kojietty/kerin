"""Repository functions: upsert/insert helpers and read queries.

We use raw SQL via SQLAlchemy core for portability and to leverage SQLite's
ON CONFLICT clauses. ORM models are intentionally omitted at this stage to
keep the dependency surface small.
"""
from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine


# ---------------------------------------------------------------------------
# Venues
# ---------------------------------------------------------------------------

def upsert_venue(engine: Engine, *, venue_id: str, name: str, bank_length: int, bank_angle: float | None = None) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO venues (venue_id, name, bank_length, bank_angle)
                VALUES (:venue_id, :name, :bank_length, :bank_angle)
                ON CONFLICT(venue_id) DO UPDATE SET
                  name = excluded.name,
                  bank_length = excluded.bank_length,
                  bank_angle = excluded.bank_angle
                """
            ),
            {"venue_id": venue_id, "name": name, "bank_length": bank_length, "bank_angle": bank_angle},
        )


# ---------------------------------------------------------------------------
# Races
# ---------------------------------------------------------------------------

def upsert_race(engine: Engine, race: dict[str, Any]) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO races (race_id, date, venue_id, race_no, grade, race_class,
                                   distance_m, weather, track_cond, post_time)
                VALUES (:race_id, :date, :venue_id, :race_no, :grade, :race_class,
                        :distance_m, :weather, :track_cond, :post_time)
                ON CONFLICT(race_id) DO UPDATE SET
                  grade = COALESCE(excluded.grade, races.grade),
                  race_class = COALESCE(excluded.race_class, races.race_class),
                  distance_m = COALESCE(excluded.distance_m, races.distance_m),
                  weather = COALESCE(excluded.weather, races.weather),
                  track_cond = COALESCE(excluded.track_cond, races.track_cond),
                  post_time = COALESCE(excluded.post_time, races.post_time)
                """
            ),
            race,
        )


def upsert_entries(engine: Engine, race_id: str, entries: Iterable[dict[str, Any]]) -> None:
    rows = list(entries)
    if not rows:
        return
    for row in rows:
        row.setdefault("race_id", race_id)
        row.setdefault("scratched", 0)
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO entries (race_id, car_no, player_id, rank_class, rating,
                                     gear_ratio, line_id, line_pos, style, scratched)
                VALUES (:race_id, :car_no, :player_id, :rank_class, :rating,
                        :gear_ratio, :line_id, :line_pos, :style, :scratched)
                ON CONFLICT(race_id, car_no) DO UPDATE SET
                  player_id  = excluded.player_id,
                  rank_class = excluded.rank_class,
                  rating     = excluded.rating,
                  gear_ratio = excluded.gear_ratio,
                  line_id    = excluded.line_id,
                  line_pos   = excluded.line_pos,
                  style      = excluded.style,
                  scratched  = excluded.scratched
                """
            ),
            rows,
        )


# ---------------------------------------------------------------------------
# Odds
# ---------------------------------------------------------------------------

def insert_odds_snapshot(
    engine: Engine,
    race_id: str,
    snapshot_at: str,
    combos: Iterable[tuple[str, float]],
) -> int:
    rows = [
        {"race_id": race_id, "snapshot_at": snapshot_at, "combo": c, "odds": o}
        for c, o in combos
    ]
    if not rows:
        return 0
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT OR IGNORE INTO odds_trifecta (race_id, snapshot_at, combo, odds)
                VALUES (:race_id, :snapshot_at, :combo, :odds)
                """
            ),
            rows,
        )
    return len(rows)


# ---------------------------------------------------------------------------
# Results & payouts
# ---------------------------------------------------------------------------

def insert_results(engine: Engine, race_id: str, results: Iterable[dict[str, Any]]) -> None:
    rows = [{**r, "race_id": race_id} for r in results]
    if not rows:
        return
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT OR IGNORE INTO results (race_id, car_no, finish, kimarite, time_sec)
                VALUES (:race_id, :car_no, :finish, :kimarite, :time_sec)
                """
            ),
            rows,
        )


def insert_payouts(engine: Engine, race_id: str, payouts: Iterable[dict[str, Any]]) -> None:
    rows = [{**p, "race_id": race_id} for p in payouts]
    if not rows:
        return
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT OR IGNORE INTO payouts (race_id, bet_type, combo, payout_yen, popularity)
                VALUES (:race_id, :bet_type, :combo, :payout_yen, :popularity)
                """
            ),
            rows,
        )


# ---------------------------------------------------------------------------
# Bets
# ---------------------------------------------------------------------------

def record_bet(
    engine: Engine,
    *,
    race_id: str,
    combo: str,
    stake_yen: int,
    pred_prob: float,
    pred_ev: float,
    is_paper: bool,
    placed_at: str | None = None,
) -> int:
    placed_at = placed_at or datetime.now().isoformat(timespec="seconds")
    with engine.begin() as conn:
        result = conn.execute(
            text(
                """
                INSERT INTO bets (race_id, combo, stake_yen, pred_prob, pred_ev, placed_at, is_paper)
                VALUES (:race_id, :combo, :stake_yen, :pred_prob, :pred_ev, :placed_at, :is_paper)
                """
            ),
            {
                "race_id": race_id,
                "combo": combo,
                "stake_yen": stake_yen,
                "pred_prob": pred_prob,
                "pred_ev": pred_ev,
                "placed_at": placed_at,
                "is_paper": 1 if is_paper else 0,
            },
        )
        return int(result.lastrowid or 0)


def settle_bets_for_race(engine: Engine, race_id: str) -> int:
    """Look up the trifecta payout for race_id and stamp each bet with payout_yen and hit."""
    with engine.begin() as conn:
        winners = conn.execute(
            text(
                """
                SELECT combo, payout_yen FROM payouts
                WHERE race_id = :race_id AND bet_type = 'trifecta'
                """
            ),
            {"race_id": race_id},
        ).fetchall()
        winning = {row[0]: row[1] for row in winners}
        bets = conn.execute(
            text("SELECT bet_id, combo, stake_yen FROM bets WHERE race_id = :rid AND payout_yen IS NULL"),
            {"rid": race_id},
        ).fetchall()
        n = 0
        for bet_id, combo, stake_yen in bets:
            payout_per_100 = winning.get(combo, 0) or 0
            payout = int(payout_per_100 * (stake_yen / 100)) if combo in winning else 0
            hit = 1 if combo in winning else 0
            conn.execute(
                text(
                    "UPDATE bets SET payout_yen = :p, hit = :h WHERE bet_id = :id"
                ),
                {"p": payout, "h": hit, "id": bet_id},
            )
            n += 1
        return n


def month_pnl(engine: Engine, year_month: str) -> dict[str, Any]:
    """Aggregate bets for YYYY-MM. Returns dict with bet_count, hits, stake, payout, pnl."""
    with engine.begin() as conn:
        row = conn.execute(
            text(
                """
                SELECT
                  COUNT(*)                                    AS bet_count,
                  COALESCE(SUM(hit), 0)                       AS hits,
                  COALESCE(SUM(stake_yen), 0)                 AS stake,
                  COALESCE(SUM(COALESCE(payout_yen, 0)), 0)   AS payout
                FROM bets
                WHERE substr(placed_at, 1, 7) = :ym
                  AND is_paper = 0
                """
            ),
            {"ym": year_month},
        ).fetchone()
        if row is None:
            return {"bet_count": 0, "hits": 0, "stake": 0, "payout": 0, "pnl": 0, "roi": None}
        bet_count, hits, stake, payout = row
        pnl = int(payout) - int(stake)
        roi = (payout / stake) if stake > 0 else None
        hit_rate = (hits / bet_count) if bet_count else None
        return {
            "bet_count": int(bet_count),
            "hits": int(hits),
            "stake": int(stake),
            "payout": int(payout),
            "pnl": int(pnl),
            "roi": roi,
            "hit_rate": hit_rate,
        }


# ---------------------------------------------------------------------------
# Fetch log
# ---------------------------------------------------------------------------

def sync_player_race_log(engine: Engine) -> int:
    """Populate player_race_log by joining entries + results + races.

    This derives the per-player history table from data already in the DB.
    Run after every `record-result` cycle to keep player_race_log current.
    Returns the number of rows inserted.
    """
    with engine.begin() as conn:
        result = conn.execute(
            text(
                """
                INSERT OR IGNORE INTO player_race_log
                  (player_id, race_id, race_date, venue_id, grade, race_class,
                   bank_length, car_no, finish, kimarite, time_sec, line_pos, scratched)
                SELECT
                  e.player_id,
                  e.race_id,
                  r.date          AS race_date,
                  r.venue_id,
                  r.grade,
                  r.race_class,
                  v.bank_length,
                  e.car_no,
                  res.finish,
                  res.kimarite,
                  res.time_sec,
                  e.line_pos,
                  e.scratched
                FROM entries e
                JOIN races r    ON e.race_id  = r.race_id
                LEFT JOIN venues v   ON r.venue_id  = v.venue_id
                LEFT JOIN results res ON e.race_id = res.race_id AND e.car_no = res.car_no
                WHERE e.player_id IS NOT NULL
                  AND r.date IS NOT NULL
                """
            )
        )
        return result.rowcount


def log_fetch(engine: Engine, url: str, status: int, bytes_: int) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO fetch_log (url, status, fetched_at, bytes)
                VALUES (:url, :status, :fetched_at, :bytes)
                """
            ),
            {
                "url": url,
                "status": status,
                "fetched_at": datetime.now().isoformat(timespec="seconds"),
                "bytes": bytes_,
            },
        )


# ---------------------------------------------------------------------------
# Prediction log
# ---------------------------------------------------------------------------

def log_prediction(
    engine: Engine,
    *,
    race_id: str,
    pred_rank1_car: int,
    pred_rank2_car: int,
    pred_rank3_car: int,
    pred_rank1_prob: float,
    pred_rank2_prob: float,
    pred_rank3_prob: float,
    model_version: str,
    predicted_at: str | None = None,
) -> int:
    """Insert a prediction log row. Returns log_id."""
    predicted_at = predicted_at or datetime.now().isoformat(timespec="seconds")
    with engine.begin() as conn:
        result = conn.execute(
            text(
                """
                INSERT INTO prediction_log
                  (race_id, predicted_at, pred_rank1_car, pred_rank2_car, pred_rank3_car,
                   pred_rank1_prob, pred_rank2_prob, pred_rank3_prob, model_version)
                VALUES
                  (:race_id, :predicted_at, :p1c, :p2c, :p3c, :p1p, :p2p, :p3p, :mv)
                """
            ),
            {
                "race_id": race_id,
                "predicted_at": predicted_at,
                "p1c": pred_rank1_car,
                "p2c": pred_rank2_car,
                "p3c": pred_rank3_car,
                "p1p": float(pred_rank1_prob),
                "p2p": float(pred_rank2_prob),
                "p3p": float(pred_rank3_prob),
                "mv": model_version,
            },
        )
        return int(result.lastrowid or 0)


def settle_prediction_log(engine: Engine, race_id: str) -> int:
    """Fill actual finish positions and hit flags from the results table.

    Called after settle_bets_for_race() in the record-result flow.
    Returns the number of rows updated.
    """
    with engine.begin() as conn:
        result_rows = conn.execute(
            text(
                "SELECT car_no, finish FROM results WHERE race_id = :rid AND finish IN (1, 2, 3)"
            ),
            {"rid": race_id},
        ).fetchall()

        if not result_rows:
            return 0

        finish_to_car = {int(r[1]): int(r[0]) for r in result_rows}
        actual_1 = finish_to_car.get(1)
        actual_2 = finish_to_car.get(2)
        actual_3 = finish_to_car.get(3)
        if actual_1 is None:
            return 0

        actual_set = {c for c in (actual_1, actual_2, actual_3) if c is not None}

        preds = conn.execute(
            text(
                """
                SELECT log_id, pred_rank1_car, pred_rank2_car, pred_rank3_car
                FROM prediction_log
                WHERE race_id = :rid AND settled_at IS NULL
                """
            ),
            {"rid": race_id},
        ).fetchall()

        n = 0
        settled_at = datetime.now().isoformat(timespec="seconds")
        for log_id, p1, p2, p3 in preds:
            pred_set = {c for c in (p1, p2, p3) if c is not None}
            rank1_hit = 1 if p1 == actual_1 else 0
            top3_hit = 1 if pred_set == actual_set else 0
            exact_hit = 1 if (p1 == actual_1 and p2 == actual_2 and p3 == actual_3) else 0
            conn.execute(
                text(
                    """
                    UPDATE prediction_log SET
                      actual_rank1_car = :a1, actual_rank2_car = :a2, actual_rank3_car = :a3,
                      rank1_hit = :r1h, top3_hit = :t3h, trifecta_exact_hit = :exa,
                      settled_at = :sat
                    WHERE log_id = :lid
                    """
                ),
                {
                    "a1": actual_1, "a2": actual_2, "a3": actual_3,
                    "r1h": rank1_hit, "t3h": top3_hit, "exa": exact_hit,
                    "sat": settled_at, "lid": log_id,
                },
            )
            n += 1
        return n


def accuracy_metrics(
    engine: Engine,
    *,
    from_date: str | None = None,
    to_date: str | None = None,
    model_version: str | None = None,
) -> dict[str, Any]:
    """Aggregate prediction_log for accuracy statistics."""
    from_dt = (from_date or "2020-01-01") + "T00:00:00"
    to_dt = (to_date or datetime.now().isoformat()[:10]) + "T23:59:59"

    params: dict[str, Any] = {"from_dt": from_dt, "to_dt": to_dt}
    extra = ""
    if model_version:
        extra = " AND model_version = :mv"
        params["mv"] = model_version

    base_where = f"predicted_at >= :from_dt AND predicted_at <= :to_dt{extra}"

    with engine.begin() as conn:
        row = conn.execute(
            text(
                f"""
                SELECT COUNT(*),
                       SUM(CASE WHEN settled_at IS NOT NULL THEN 1 ELSE 0 END),
                       SUM(COALESCE(rank1_hit, 0)),
                       SUM(COALESCE(top3_hit, 0)),
                       SUM(COALESCE(trifecta_exact_hit, 0))
                FROM prediction_log WHERE {base_where}
                """
            ),
            params,
        ).fetchone()

        total, settled, r1_hits, t3_hits, ex_hits = (row or (0, 0, 0, 0, 0))
        total = int(total or 0)
        settled = int(settled or 0)

        monthly = conn.execute(
            text(
                f"""
                SELECT substr(predicted_at, 1, 7) as month,
                       COUNT(*) as cnt,
                       ROUND(1.0 * SUM(COALESCE(rank1_hit, 0)) / MAX(COUNT(*), 1), 3),
                       ROUND(1.0 * SUM(COALESCE(top3_hit, 0))  / MAX(COUNT(*), 1), 3)
                FROM prediction_log
                WHERE {base_where} AND settled_at IS NOT NULL
                GROUP BY month ORDER BY month
                """
            ),
            params,
        ).fetchall()

    return {
        "total_predictions": total,
        "settled": settled,
        "rank1_hit_rate": round(int(r1_hits) / settled, 3) if settled else None,
        "top3_hit_rate": round(int(t3_hits) / settled, 3) if settled else None,
        "trifecta_exact_hit_rate": round(int(ex_hits) / settled, 3) if settled else None,
        "monthly": [
            {"month": r[0], "count": int(r[1]), "rank1_rate": r[2], "top3_rate": r[3]}
            for r in monthly
        ],
    }


def recently_fetched(engine: Engine, url: str, within_hours: int = 24) -> bool:
    with engine.begin() as conn:
        row = conn.execute(
            text(
                """
                SELECT 1 FROM fetch_log
                WHERE url = :url AND status = 200
                  AND fetched_at >= datetime('now', :delta)
                LIMIT 1
                """
            ),
            {"url": url, "delta": f"-{within_hours} hours"},
        ).fetchone()
        return row is not None
