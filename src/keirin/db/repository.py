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
