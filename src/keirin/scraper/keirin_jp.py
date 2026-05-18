"""Scraper façade — uses keirin.netkeiba.com (netkeirin) as primary source.

netkeirin provides rich structured data (comments, gear, all stats) and was
verified against the live site in D:\\keirin. See scraper/netkeirin.py for the
URL patterns and CSS selectors.

race_id format: YYYYMMDD{venue_code:02d}{race_no:02d}  (12 digits)
"""
from __future__ import annotations

import logging
from datetime import date as _date
from datetime import datetime
from pathlib import Path

from keirin.config import load_app_config
from keirin.db.engine import get_engine, init_db
from keirin.db.repository import (
    insert_odds_snapshot,
    insert_payouts,
    insert_results,
    upsert_entries,
    upsert_race,
    upsert_venue,
)
from keirin.scraper import netkeirin as nk
from keirin.scraper.rate_limiter import DailyRequestCap, RateLimiter, RobotsCache
from keirin.scraper.session import PoliteSession

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scraper facade
# ---------------------------------------------------------------------------

class KeirinJp:
    """Façade over netkeirin.com scraping + SQLite upsert."""

    def __init__(self) -> None:
        cfg = load_app_config()
        self.cfg = cfg
        self.engine = get_engine(cfg.paths.db)
        init_db(self.engine)
        self.session = PoliteSession(
            base_url=nk.BASE,
            user_agent=cfg.scraper.user_agent,
            rate_limiter=RateLimiter(
                interval_sec=cfg.scraper.request_interval_sec,
                jitter_sec=cfg.scraper.jitter_sec,
            ),
            robots=RobotsCache(nk.BASE, cfg.scraper.user_agent),
            daily_cap=DailyRequestCap(cfg.scraper.daily_request_cap),
            cache_dir=cfg.paths.raw_html if cfg.scraper.cache_raw_html else None,
            engine=self.engine,
            timeout_sec=cfg.scraper.timeout_sec,
            respect_robots=cfg.scraper.respect_robots,
        )
        self._ensure_venues()

    # --- schedule -------------------------------------------------------

    def fetch_schedule(self, day: _date) -> list[dict]:
        """Fetch today's race list from netkeirin.com."""
        url = nk.schedule_url(day)
        cache_sub = f"{day:%Y%m%d}/schedule"
        res = self.session.get(url, cache_subdir=cache_sub)
        if res.status != 200:
            log.warning("schedule fetch failed status=%s", res.status)
            return []
        from bs4 import BeautifulSoup
        import re
        soup = BeautifulSoup(res.text, "lxml")
        races: list[dict] = []
        seen: set[str] = set()
        for a in soup.select("a[href*='race_id=']"):
            m = re.search(r"race_id=(\d{12})", a.get("href", ""))
            if not m:
                continue
            race_id = m.group(1)
            if race_id in seen:
                continue
            seen.add(race_id)
            venue_id = race_id[8:10]
            race_no = int(race_id[10:12])
            date_iso = f"{race_id[0:4]}-{race_id[4:6]}-{race_id[6:8]}"
            race = {
                "race_id": race_id,
                "date": date_iso,
                "venue_id": venue_id,
                "race_no": race_no,
                "grade": None, "race_class": None,
                "distance_m": None, "weather": None,
                "track_cond": None, "post_time": None,
            }
            upsert_race(self.engine, race)
            races.append(race)
        log.info("schedule: %d races on %s", len(races), day)
        return races

    # --- cards ----------------------------------------------------------

    def fetch_card(self, race_id: str) -> dict:
        """Fetch race entry page and save to DB (entries + race meta)."""
        url = nk.entry_url(race_id)
        cache_sub = f"{race_id[:8]}/cards"
        res = self.session.get(url, cache_subdir=cache_sub)
        if res.status != 200:
            log.warning("card fetch failed status=%s race_id=%s", res.status, race_id)
            return {}

        raw_entries = nk.parse_entries(res.text)
        enriched = nk.parse_line_assignments(raw_entries)

        venue_id = race_id[8:10]
        date_iso = f"{race_id[0:4]}-{race_id[4:6]}-{race_id[6:8]}"

        # Grade detection from entries
        grades = [e.get("grade") for e in enriched if e.get("grade")]
        grade = _best_grade(grades)

        upsert_race(self.engine, {
            "race_id": race_id,
            "date": date_iso,
            "venue_id": venue_id,
            "race_no": int(race_id[10:12]),
            "grade": grade,
            "race_class": None, "distance_m": None,
            "weather": None, "track_cond": None, "post_time": None,
        })

        db_entries = [_to_db_entry(e, race_id) for e in enriched]
        upsert_entries(self.engine, race_id, db_entries)

        # Upsert players with current stats
        from sqlalchemy import text
        with self.engine.begin() as conn:
            for e in enriched:
                if not e.get("player_id"):
                    continue
                conn.execute(text(
                    """
                    INSERT INTO players (player_id, name, style, rank_class, rating, updated_at)
                    VALUES (:pid, :name, :style, :rank_class, :rating, :upd)
                    ON CONFLICT(player_id) DO UPDATE SET
                      name=excluded.name, style=excluded.style,
                      rank_class=excluded.rank_class, rating=excluded.rating,
                      updated_at=excluded.updated_at
                    """
                ), {
                    "pid": e["player_id"],
                    "name": e.get("name", "?"),
                    "style": _style_norm(e.get("style_code")),
                    "rank_class": e.get("grade"),
                    "rating": _flt(e.get("race_score")),
                    "upd": date_iso,
                })

        log.info("card: race_id=%s  %d entries", race_id, len(db_entries))
        return {"race_id": race_id, "entries": db_entries}

    # --- odds -----------------------------------------------------------

    def fetch_odds(self, race_id: str) -> int:
        """Fetch trifecta odds snapshot from netkeirin and save to DB."""
        url = nk.odds_url(race_id)
        cache_sub = f"{race_id[:8]}/odds"
        res = self.session.get(url, cache_subdir=cache_sub, skip_cache=True)
        if res.status != 200:
            log.warning("odds fetch failed status=%s race_id=%s", res.status, race_id)
            return 0
        snapshot_at = datetime.now().isoformat(timespec="seconds")
        combos = nk.parse_trifecta_odds(res.text)
        n = insert_odds_snapshot(self.engine, race_id, snapshot_at, combos)
        log.info("odds: race_id=%s  %d combos saved", race_id, n)
        return n

    # --- results --------------------------------------------------------

    def fetch_result(self, race_id: str) -> dict:
        """Fetch race result from netkeirin and save to DB."""
        url = nk.result_url(race_id)
        cache_sub = f"{race_id[:8]}/results"
        res = self.session.get(url, cache_subdir=cache_sub)
        if res.status != 200:
            log.warning("result fetch failed status=%s race_id=%s", res.status, race_id)
            return {}
        parsed = nk.parse_result(res.text)
        if parsed.get("results"):
            insert_results(self.engine, race_id, parsed["results"])
        if parsed.get("payouts"):
            insert_payouts(self.engine, race_id, parsed["payouts"])
        return parsed

    # --- helpers --------------------------------------------------------

    def _ensure_venues(self) -> None:
        from keirin.data.import_cache import VENUE_TABLE
        for vid, (name, bank) in VENUE_TABLE.items():
            upsert_venue(self.engine, venue_id=vid, name=name, bank_length=bank)

    def raw_dir_for(self, day: _date) -> Path:
        d = self.cfg.paths.raw_html / day.strftime("%Y%m%d")
        d.mkdir(parents=True, exist_ok=True)
        return d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_db_entry(e: dict, race_id: str) -> dict:
    return {
        "race_id": race_id,
        "car_no": int(e.get("car_no") or 0),
        "player_id": e.get("player_id"),
        "rank_class": e.get("grade"),
        "rating": _flt(e.get("race_score")),
        "gear_ratio": _flt(e.get("gear")),
        "line_id": e.get("line_id"),
        "line_pos": e.get("line_pos"),
        "style": _style_norm(e.get("style_code")),
        "scratched": 0,
    }


def _best_grade(grades: list[str | None]) -> str | None:
    rank = {"SS": 0, "S1": 1, "S2": 2, "A1": 3, "A2": 4, "A3": 5}
    gs = [g for g in grades if g]
    return min(set(gs), key=lambda g: rank.get(g, 9)) if gs else None


def _style_norm(code: str | None) -> str | None:
    if not code:
        return None
    s = str(code).strip()
    mapping = {"逃": "逃", "捲": "捲", "差": "差", "マ": "マーク", "追": "マーク", "自": "逃"}
    for k, v in mapping.items():
        if k in s:
            return v
    return s[:4]


def _flt(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
