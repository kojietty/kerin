"""D:\\keirin のキャッシュ JSON をローカル SQLite に一括インポートする。

使い方:
  python -m keirin import-cache --file D:\\keirin\\backtest\\real_race_cache.json

インポートされるもの:
  - venues (会場マスタ)
  - players (選手マスタ — 最新エントリの値で上書き)
  - races   (レース基本情報)
  - entries (出走表)
  - results (着順 — finish フィールドから)
  - payouts (三連単払戻 — trifecta_combo/trifecta_odds から)
  - player_race_log (sync_player_race_log() で自動生成)
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

log = logging.getLogger(__name__)

# 会場コード → (名前, バンク周長)
VENUE_TABLE: dict[str, tuple[str, int]] = {
    "11": ("函館", 400), "12": ("青森", 400), "13": ("いわき平", 400),
    "21": ("弥彦", 400), "22": ("前橋", 335), "23": ("取手", 400),
    "24": ("宇都宮", 500), "25": ("大宮", 500), "26": ("西武園", 400),
    "27": ("京王閣", 400), "28": ("立川", 400), "31": ("松戸", 333),
    "32": ("千葉", 500), "34": ("川崎", 400), "35": ("平塚", 400),
    "36": ("小田原", 333), "37": ("伊東温泉", 333), "38": ("静岡", 400),
    "42": ("名古屋", 400), "43": ("岐阜", 400), "44": ("大垣", 400),
    "45": ("豊橋", 400), "46": ("富山", 333), "47": ("松阪", 400),
    "48": ("四日市", 400), "51": ("福井", 400), "53": ("奈良", 333),
    "54": ("向日町", 400), "55": ("和歌山", 400), "56": ("岸和田", 400),
    "61": ("玉野", 400), "62": ("広島", 400), "63": ("防府", 333),
    "71": ("高松", 400), "73": ("小松島", 400), "74": ("高知", 500),
    "75": ("松山", 400), "81": ("小倉", 400), "83": ("久留米", 400),
    "84": ("武雄", 400), "85": ("佐世保", 400), "86": ("別府", 400),
    "87": ("熊本", 400),
}


def import_cache(
    engine: Engine,
    cache_path: Path,
    *,
    skip_existing: bool = True,
    verbose: bool = True,
) -> dict[str, int]:
    """Import races from a cache JSON file.

    Returns a dict of count statistics.
    """
    with open(cache_path, encoding="cp932", errors="replace") as f:
        data: dict[str, Any] = json.load(f)

    log.info("Loaded %d entries from %s", len(data), cache_path)

    stats = {"races": 0, "entries": 0, "results": 0, "payouts": 0, "skipped": 0}

    # Pre-insert all venues once
    _ensure_venues(engine)

    # Track already-imported races to allow --skip-existing
    existing: set[str] = set()
    if skip_existing:
        with engine.begin() as conn:
            rows = conn.execute(text("SELECT race_id FROM races")).fetchall()
            existing = {r[0] for r in rows}

    total = len(data)
    for idx, (race_id, race) in enumerate(data.items(), 1):
        if not race.get("valid"):
            stats["skipped"] += 1
            continue
        entries_raw = race.get("entries", [])
        if len(entries_raw) < 5:
            stats["skipped"] += 1
            continue
        if race_id in existing:
            stats["skipped"] += 1
            continue

        if verbose and idx % 500 == 0:
            log.info("  progress %d / %d  (races=%d)", idx, total, stats["races"])

        try:
            _import_race(engine, race_id, race)
            stats["races"] += 1
            stats["entries"] += len(entries_raw)
            if race.get("finish") and len(race["finish"]) >= 3:
                stats["results"] += len(race["finish"])
            if race.get("trifecta_combo") and race.get("trifecta_odds"):
                stats["payouts"] += 1
        except Exception as e:
            log.warning("  failed to import race_id=%s: %s", race_id, e)

    log.info(
        "Import done: %d races, %d entries, %d results, %d payouts, %d skipped",
        stats["races"], stats["entries"], stats["results"],
        stats["payouts"], stats["skipped"],
    )
    return stats


def _import_race(engine: Engine, race_id: str, race: dict) -> None:
    venue_id = race_id[8:10]
    race_no = int(race_id[10:12])
    date_str = f"{race_id[0:4]}-{race_id[4:6]}-{race_id[6:8]}"
    grade = race.get("grade") or _detect_grade(race.get("entries", []))

    with engine.begin() as conn:
        # --- races ---
        conn.execute(
            text(
                """
                INSERT OR IGNORE INTO races
                  (race_id, date, venue_id, race_no, grade, race_class, distance_m)
                VALUES (:race_id, :date, :venue_id, :race_no, :grade, NULL, NULL)
                """
            ),
            {
                "race_id": race_id,
                "date": date_str,
                "venue_id": venue_id,
                "race_no": race_no,
                "grade": grade,
            },
        )

        # --- players + entries ---
        for e in race.get("entries", []):
            player_id = str(e.get("player_id") or "").strip()
            if not player_id:
                continue

            car_no = _int(e.get("car_no"))
            name = (e.get("name") or "").strip() or "?"

            # Upsert player (latest race's stats)
            conn.execute(
                text(
                    """
                    INSERT INTO players (player_id, name, style, rank_class, rating, updated_at)
                    VALUES (:pid, :name, :style, :rank_class, :rating, :updated)
                    ON CONFLICT(player_id) DO UPDATE SET
                      name       = excluded.name,
                      style      = excluded.style,
                      rank_class = excluded.rank_class,
                      rating     = excluded.rating,
                      updated_at = excluded.updated_at
                    """
                ),
                {
                    "pid": player_id,
                    "name": name,
                    "style": _style(e.get("style_code")),
                    "rank_class": _grade(e.get("grade")),
                    "rating": _float(e.get("race_score")),
                    "updated": date_str,
                },
            )

            # Upsert entry
            conn.execute(
                text(
                    """
                    INSERT INTO entries
                      (race_id, car_no, player_id, rank_class, rating, gear_ratio,
                       line_id, line_pos, style, scratched)
                    VALUES
                      (:race_id, :car_no, :pid, :rank_class, :rating, :gear,
                       NULL, NULL, :style, 0)
                    ON CONFLICT(race_id, car_no) DO UPDATE SET
                      player_id  = excluded.player_id,
                      rank_class = excluded.rank_class,
                      rating     = excluded.rating,
                      gear_ratio = excluded.gear_ratio,
                      style      = excluded.style
                    """
                ),
                {
                    "race_id": race_id,
                    "car_no": car_no,
                    "pid": player_id,
                    "rank_class": _grade(e.get("grade")),
                    "rating": _float(e.get("race_score")),
                    "gear": _float(e.get("gear")),
                    "style": _style(e.get("style_code")),
                },
            )

            # Store per-entry top3/win rates in player_history as a snapshot
            # (This gives us pre-computed rates even before player_race_log is fully built)
            _store_entry_rates(conn, player_id, car_no, race_id, e)

        # --- results ---
        finish = race.get("finish", [])
        for pos, car_no in enumerate(finish[:3], start=1):
            conn.execute(
                text(
                    """
                    INSERT OR IGNORE INTO results (race_id, car_no, finish, kimarite, time_sec)
                    VALUES (:rid, :cn, :finish, NULL, NULL)
                    """
                ),
                {"rid": race_id, "cn": int(car_no), "finish": pos},
            )

        # --- payout (winning trifecta) ---
        combo = race.get("trifecta_combo")
        odds = race.get("trifecta_odds")
        if combo and odds:
            # keirin odds are the multiplier (倍率); payout_yen = odds × 100
            payout_yen = int(float(odds) * 100)
            conn.execute(
                text(
                    """
                    INSERT OR IGNORE INTO payouts (race_id, bet_type, combo, payout_yen, popularity)
                    VALUES (:rid, 'trifecta', :combo, :payout, NULL)
                    """
                ),
                {"rid": race_id, "combo": combo, "payout": payout_yen},
            )


def _store_entry_rates(conn, player_id: str, car_no: int, race_id: str, e: dict) -> None:
    """Store win/top3 rates as a snapshot so features can fall back to these
    when player_race_log doesn't have enough history.
    This goes into a dedicated snapshot column in players.
    Currently a no-op placeholder — real rolling stats come from player_race_log.
    """
    pass


def _ensure_venues(engine: Engine) -> None:
    with engine.begin() as conn:
        for vid, (name, bank) in VENUE_TABLE.items():
            conn.execute(
                text(
                    """
                    INSERT OR IGNORE INTO venues (venue_id, name, bank_length)
                    VALUES (:vid, :name, :bank)
                    """
                ),
                {"vid": vid, "name": name, "bank": bank},
            )


def _detect_grade(entries: list) -> str | None:
    if not entries:
        return None
    grades = [e.get("grade") for e in entries if e.get("grade")]
    if not grades:
        return None
    # Take the most common grade (or the best one)
    rank = {"SS": 0, "S1": 1, "S2": 2, "A1": 3, "A2": 4, "A3": 5}
    return min(set(grades), key=lambda g: rank.get(g, 9))


def _int(v) -> int:
    try:
        return int(str(v).split(".")[0])
    except (TypeError, ValueError):
        return 0


def _float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _style(code) -> str | None:
    if not code:
        return None
    mapping = {"逃": "逃", "捲": "捲", "差": "差", "マ": "マーク", "追": "マーク", "自": "逃"}
    s = str(code).strip()
    for k, v in mapping.items():
        if k in s:
            return v
    return s[:4] if s else None


def _grade(g) -> str | None:
    if not g:
        return None
    s = str(g).upper().replace("Ａ", "A").replace("Ｓ", "S")
    for key in ("SS", "S1", "S2", "A1", "A2", "A3"):
        if key in s:
            return key
    return g
