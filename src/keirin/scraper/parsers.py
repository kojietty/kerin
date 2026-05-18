"""HTML parsers for KEIRIN.JP pages.

These parsers are intentionally defensive: KEIRIN.JP markup has changed
historically, so we use a mix of pandas.read_html and BeautifulSoup with broad
fallbacks. Each parser returns plain dicts/lists ready for DB upsert.

NOTE: The live HTML structure may need adjustment in the field. Each parser
returns an empty container rather than raising on unknown layout, so the
pipeline keeps running while the scraping rules are refined.
"""
from __future__ import annotations

import logging
import re
from datetime import date as _date
from io import StringIO
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Venue master (subset; expand as needed)
# ---------------------------------------------------------------------------

VENUES: dict[str, tuple[str, int]] = {
    # venue_id : (display name, default bank length m)
    "11": ("函館", 400), "12": ("青森", 400), "13": ("いわき平", 400),
    "21": ("弥彦", 400), "22": ("前橋", 335), "23": ("取手", 400),
    "24": ("宇都宮", 500), "25": ("大宮", 500), "26": ("西武園", 400),
    "27": ("京王閣", 400), "28": ("立川", 400),
    "31": ("松戸", 333), "32": ("千葉", 500), "34": ("川崎", 400),
    "35": ("平塚", 400), "36": ("小田原", 333), "37": ("伊東", 333),
    "38": ("静岡", 400),
    "41": ("名古屋", 400), "42": ("岐阜", 400), "43": ("大垣", 400),
    "44": ("豊橋", 400), "45": ("富山", 333), "46": ("松阪", 400),
    "47": ("四日市", 400),
    "51": ("福井", 400), "52": ("奈良", 333), "53": ("向日町", 400),
    "54": ("和歌山", 400), "55": ("岸和田", 400),
    "61": ("玉野", 400), "62": ("広島", 400), "63": ("防府", 333),
    "71": ("高松", 400), "72": ("小松島", 400), "73": ("高知", 500),
    "74": ("松山", 400),
    "81": ("小倉", 400), "83": ("久留米", 400), "84": ("武雄", 400),
    "85": ("佐世保", 400), "86": ("別府", 400), "87": ("熊本", 400),
}


def venue_name(venue_id: str) -> str:
    return VENUES.get(venue_id, ("不明", 400))[0]


def default_bank_length(venue_id: str) -> int:
    return VENUES.get(venue_id, ("不明", 400))[1]


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

_RACE_CODE_RE = re.compile(r"raceCode=([0-9A-Za-z]+)")


def parse_schedule(html: str, *, day: _date) -> list[dict[str, Any]]:
    """Extract race headers from the daily race table.

    Returns dicts with keys: race_id, date, venue_id, race_no, grade, post_time.
    """
    races: list[dict[str, Any]] = []
    soup = BeautifulSoup(html, "lxml")
    seen: set[str] = set()
    for a in soup.select("a[href*='raceCode=']"):
        m = _RACE_CODE_RE.search(a.get("href", ""))
        if not m:
            continue
        race_id = m.group(1)
        if race_id in seen:
            continue
        seen.add(race_id)
        # Try to infer venue_id and race_no from race_id (KEIRIN.JP convention:
        # YYYYMMDD + venue + race). This is heuristic; refine when needed.
        venue_id, race_no = _split_race_id(race_id)
        races.append(
            {
                "race_id": race_id,
                "date": day.isoformat(),
                "venue_id": venue_id,
                "race_no": race_no,
                "grade": None,
                "race_class": None,
                "distance_m": None,
                "weather": None,
                "track_cond": None,
                "post_time": None,
            }
        )
    return races


def _split_race_id(race_id: str) -> tuple[str, int]:
    """Best-effort venue/race extraction from a race_id token."""
    digits = re.sub(r"\D", "", race_id)
    if len(digits) >= 12:
        # YYYYMMDD + venue(2) + race(2)
        return digits[8:10], int(digits[10:12] or "0")
    if len(digits) >= 10:
        return digits[8:10], int(digits[10:] or "0")
    return "00", 0


# ---------------------------------------------------------------------------
# Race card
# ---------------------------------------------------------------------------

def parse_card(html: str, *, race_id: str) -> dict[str, Any]:
    """Extract race meta + entries.

    Returns: {"race": {...}, "entries": [...]}
    """
    soup = BeautifulSoup(html, "lxml")

    race_meta = _extract_card_meta(soup, race_id)
    entries = _extract_card_entries(html, soup, race_id)

    # Add line_id/line_pos heuristic from line-list element if present.
    _populate_line_info(soup, entries)

    return {"race": race_meta, "entries": entries}


def _extract_card_meta(soup: BeautifulSoup, race_id: str) -> dict[str, Any]:
    txt = soup.get_text(" ", strip=True)
    venue_id, race_no = _split_race_id(race_id)
    date_iso = _date_from_race_id(race_id)
    grade = None
    for tag in ("GP", "GⅠ", "GI", "GⅡ", "GII", "GⅢ", "GIII", "F1", "F2"):
        if tag in txt:
            grade = tag
            break
    distance = None
    m = re.search(r"(\d{3,4})\s*m", txt)
    if m:
        distance = int(m.group(1))
    return {
        "race_id": race_id,
        "date": date_iso,
        "venue_id": venue_id,
        "race_no": race_no,
        "grade": grade,
        "race_class": None,
        "distance_m": distance,
        "weather": None,
        "track_cond": None,
        "post_time": None,
    }


def _date_from_race_id(race_id: str) -> str:
    digits = re.sub(r"\D", "", race_id)
    if len(digits) >= 8:
        return f"{digits[0:4]}-{digits[4:6]}-{digits[6:8]}"
    return ""


def _extract_card_entries(html: str, soup: BeautifulSoup, race_id: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    # First try pandas.read_html on the largest plausible table.
    try:
        dfs = pd.read_html(StringIO(html), flavor="lxml")
    except ValueError:
        dfs = []
    target_df: pd.DataFrame | None = None
    for df in dfs:
        cols = [str(c) for c in df.columns]
        joined = " ".join(cols)
        if any(k in joined for k in ("車番", "選手", "級班", "得点")):
            target_df = df
            break
        if df.shape[0] in (7, 8, 9) and df.shape[1] >= 4:
            target_df = df
            break
    if target_df is not None:
        for _, row in target_df.iterrows():
            entry = _row_to_entry(row, race_id)
            if entry is not None:
                entries.append(entry)
    if not entries:
        log.debug("card entries fallback: no tables matched for race_id=%s", race_id)
    # Player IDs from anchor links
    pid_by_row = []
    for a in soup.select("a[href*='playerCode=']"):
        m = re.search(r"playerCode=([0-9A-Za-z]+)", a.get("href", ""))
        if m:
            pid_by_row.append(m.group(1))
    for i, e in enumerate(entries):
        if i < len(pid_by_row) and not e.get("player_id"):
            e["player_id"] = pid_by_row[i]
    return entries


def _row_to_entry(row: pd.Series, race_id: str) -> dict[str, Any] | None:
    car_no = _first_int(row, ["車番", "Unnamed: 0"])
    if car_no is None:
        return None
    return {
        "race_id": race_id,
        "car_no": car_no,
        "player_id": None,
        "rank_class": _first_str(row, ["級班"]),
        "rating": _first_float(row, ["競走得点", "得点"]),
        "gear_ratio": _first_float(row, ["ギア比", "ギア"]),
        "line_id": None,
        "line_pos": None,
        "style": _first_str(row, ["脚質"]),
        "scratched": 0,
    }


def _populate_line_info(soup: BeautifulSoup, entries: list[dict[str, Any]]) -> None:
    """Best-effort: look for a 'ライン' or '並び' indicator like '3-1-5 / 2-7 / 4-6'."""
    text = soup.get_text(" ", strip=True)
    m = re.search(r"並び[^0-9]*([0-9\-\s/／]+)", text)
    if not m:
        return
    raw = m.group(1).replace("／", "/")
    by_car = {e["car_no"]: e for e in entries}
    for line_id, group in enumerate(raw.split("/"), start=1):
        cars = [int(x) for x in re.findall(r"\d", group)]
        for pos, car_no in enumerate(cars, start=1):
            if car_no in by_car:
                by_car[car_no]["line_id"] = line_id
                by_car[car_no]["line_pos"] = pos


# ---------------------------------------------------------------------------
# Trifecta odds
# ---------------------------------------------------------------------------

_COMBO_RE = re.compile(r"^(\d)[-→](\d)[-→](\d)$")


def parse_trifecta_odds(html: str) -> list[tuple[str, float]]:
    """Return list of (combo_str, odds) pairs for trifecta combos."""
    combos: dict[str, float] = {}
    try:
        dfs = pd.read_html(StringIO(html), flavor="lxml")
    except ValueError:
        dfs = []
    for df in dfs:
        # Skip too-small frames
        if df.size < 4:
            continue
        df_str = df.astype(str)
        for _, row in df_str.iterrows():
            for cell in row.values:
                _maybe_extract_combo_with_odds(cell, row.values, combos)
    return list(combos.items())


def _maybe_extract_combo_with_odds(cell: str, row_values, combos: dict[str, float]) -> None:
    """Look for a 'A-B-C' combo in the row and a numeric odds in another cell."""
    m = _COMBO_RE.match(cell.strip())
    if not m:
        return
    combo = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    for v in row_values:
        try:
            f = float(str(v).replace(",", ""))
        except (TypeError, ValueError):
            continue
        if 1.0 < f < 100000.0:
            combos.setdefault(combo, f)
            return


# ---------------------------------------------------------------------------
# Results & payouts
# ---------------------------------------------------------------------------

def parse_result(html: str, *, race_id: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")
    results = _extract_results(soup)
    payouts = _extract_payouts(html)
    return {"results": results, "payouts": payouts}


def _extract_results(soup: BeautifulSoup) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    text = soup.get_text(" ", strip=True)
    # Look for "着順" table block; otherwise fall back to small finish blocks
    m = re.search(r"着順[^0-9]*((?:\d\s*\(\d\)\s*)+)", text)
    if m:
        for finish, car_no in re.findall(r"(\d)\s*\((\d)\)", m.group(1)):
            results.append(
                {
                    "car_no": int(car_no),
                    "finish": int(finish),
                    "kimarite": None,
                    "time_sec": None,
                }
            )
    return results


def _extract_payouts(html: str) -> list[dict[str, Any]]:
    payouts: list[dict[str, Any]] = []
    try:
        dfs = pd.read_html(StringIO(html), flavor="lxml")
    except ValueError:
        dfs = []
    for df in dfs:
        df_str = df.astype(str)
        for _, row in df_str.iterrows():
            text = " ".join(str(v) for v in row.values)
            if "三連単" in text:
                combo, yen = _parse_payout_row(row.values)
                if combo and yen:
                    payouts.append(
                        {
                            "bet_type": "trifecta",
                            "combo": combo,
                            "payout_yen": yen,
                            "popularity": None,
                        }
                    )
    return payouts


def _parse_payout_row(values) -> tuple[str | None, int | None]:
    combo = None
    yen = None
    for v in values:
        s = str(v)
        if combo is None:
            m = _COMBO_RE.match(s.strip())
            if m:
                combo = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        if yen is None:
            m2 = re.search(r"([\d,]+)\s*円", s)
            if m2:
                yen = int(m2.group(1).replace(",", ""))
    return combo, yen


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _first_int(row: pd.Series, keys: list[str]) -> int | None:
    for k in keys:
        if k in row.index:
            try:
                return int(row[k])
            except (TypeError, ValueError):
                continue
    return None


def _first_float(row: pd.Series, keys: list[str]) -> float | None:
    for k in keys:
        if k in row.index:
            try:
                return float(row[k])
            except (TypeError, ValueError):
                continue
    return None


def _first_str(row: pd.Series, keys: list[str]) -> str | None:
    for k in keys:
        if k in row.index:
            v = row[k]
            if pd.notna(v):
                return str(v).strip()
    return None
