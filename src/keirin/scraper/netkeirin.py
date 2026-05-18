"""Scraper for keirin.netkeiba.com (netkeirin).

URL patterns and CSS selectors ported from D:\\keirin\\scraper\\netkeirin_scraper.py
which was verified against the live site. This is our primary data source:
it provides rich data including player comments, gear ratios, all stats, and
structured HTML that's more stable than keirin.go.jp.

race_id format: YYYYMMDD{venue_code:02d}{race_no:02d}  (12 chars, all digits)
  Example: 202605178102  →  2026-05-17, 小倉 (81), race 2

Venue codes are two-digit zero-padded integers matching VENUE_CODES dict.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date as _date
from typing import Any

from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

BASE = "https://keirin.netkeiba.com"

# Venue code ↔ name (same as D:\keirin)
VENUE_CODES: dict[str, int] = {
    "函館": 11, "青森": 12, "いわき平": 13, "弥彦": 21, "前橋": 22,
    "取手": 23, "宇都宮": 24, "大宮": 25, "西武園": 26, "京王閣": 27,
    "立川": 28, "松戸": 31, "千葉": 32, "川崎": 34, "平塚": 35,
    "小田原": 36, "伊東温泉": 37, "静岡": 38, "名古屋": 42, "岐阜": 43,
    "大垣": 44, "豊橋": 45, "富山": 46, "松阪": 47, "四日市": 48,
    "福井": 51, "奈良": 53, "向日町": 54, "和歌山": 55, "岸和田": 56,
    "玉野": 61, "広島": 62, "防府": 63, "高松": 71, "小松島": 73,
    "高知": 74, "松山": 75, "小倉": 81, "久留米": 83, "武雄": 84,
    "佐世保": 85, "別府": 86, "熊本": 87,
}
CODE_VENUE: dict[int, str] = {v: k for k, v in VENUE_CODES.items()}


@dataclass
class RiderEntry:
    car_no: int
    player_id: str
    name: str
    grade: str
    race_score: float
    style_code: str
    win_rate: float
    top2_rate: float
    top3_rate: float
    gear: float
    comment: str
    nige: int = 0
    makuri: int = 0
    sashi: int = 0
    mark: int = 0
    sprints: int = 0
    recent_results: list[dict] = field(default_factory=list)


def race_id_for(day: _date, venue_code: int, race_no: int) -> str:
    return f"{day.strftime('%Y%m%d')}{venue_code:02d}{race_no:02d}"


def schedule_url(day: _date) -> str:
    return f"{BASE}/race/"


def entry_url(race_id: str) -> str:
    return f"{BASE}/race/entry/?race_id={race_id}"


def odds_url(race_id: str) -> str:
    return f"{BASE}/race/odds/?race_id={race_id}&rf=top"


def result_url(race_id: str) -> str:
    return f"{BASE}/race/result/?race_id={race_id}"


def player_result_url(player_id: str) -> str:
    return f"{BASE}/db/player_result/result_summary.html?id={player_id}"


# ---------------------------------------------------------------------------
# Parsers (adapted from D:\keirin\scraper\netkeirin_scraper.py)
# Column index map for table[1] (RaceCardCell01):
#   0=枠, 1=車番, 2=本紙, 3=ライン, 4=選手(名+詳細), 5=競走得点,
#   6=脚質, 7=S, 8=B, 9=逃, 10=捲, 11=差, 12=マ,
#   13=1着, 14=2着, 15=3着, 16=着外, 17=勝率, 18=2連対率, 19=3連対率,
#   20=ギヤ, 21=コメント
# ---------------------------------------------------------------------------

def parse_entries(html: str) -> list[dict[str, Any]]:
    """Parse race entry page HTML. Returns list of entry dicts for DB upsert."""
    soup = BeautifulSoup(html, "lxml")
    entries: list[dict[str, Any]] = []

    # player_id from <a id="names_{pid}_{race_id}">
    pid_list: list[str] = []
    for tag in soup.select("a[id^='names_']"):
        m = re.match(r"names_(\d+)_", tag.get("id", ""))
        if m:
            pid_list.append(m.group(1))

    tables = soup.select("table")
    if len(tables) < 2:
        # Try with just one table
        if tables:
            detail_rows = tables[0].select("tbody tr")
        else:
            log.debug("parse_entries: no tables found")
            return []
    else:
        detail_rows = tables[1].select("tbody tr")

    for i, row in enumerate(detail_rows):
        cells = row.select("td")
        if len(cells) < 20:
            continue
        try:
            style_raw = _txt(cells, 6)
            style_m = re.match(r"(\d*)(.*)", style_raw)
            style_code = style_m.group(2).strip() if style_m else style_raw

            entries.append({
                "car_no": _int(cells, 1),
                "player_id": pid_list[i] if i < len(pid_list) else None,
                "name": _name(cells, 4),
                "grade": _grade_cell(cells, 4),
                "race_score": _float(cells, 5),
                "style_code": style_code,
                "sprints": _int(cells, 7),
                "nige": _int(cells, 9),
                "makuri": _int(cells, 10),
                "sashi": _int(cells, 11),
                "mark": _int(cells, 12),
                "win_rate": _pct(cells, 17),
                "top2_rate": _pct(cells, 18),
                "top3_rate": _pct(cells, 19),
                "gear": _float(cells, 20),
                "comment": _txt(cells, 21) if len(cells) > 21 else "",
                "line_raw": _txt(cells, 3),  # raw ライン cell text e.g. "3-1-5"
            })
        except Exception as e:
            log.debug("parse_entries row %d error: %s", i, e)
            continue

    return entries


def parse_line_assignments(entries: list[dict]) -> list[dict]:
    """Assign line_id and line_pos from line_raw fields.

    ライン cell format: "3-1-5 / 2-7 / 4-6" (line groups separated by /)
    Each group lists car numbers from front to back.
    """
    # Collect all line strings; find the most common non-trivial one.
    # In practice, all riders' line_raw cells should contain the same string.
    line_raw = ""
    for e in entries:
        if e.get("line_raw") and len(e.get("line_raw", "")) > 1:
            line_raw = e["line_raw"]
            break

    if not line_raw:
        return entries

    result = {e["car_no"]: e.copy() for e in entries}
    groups = re.split(r"[/／]", line_raw)
    for line_id, group in enumerate(groups, start=1):
        cars = [int(x) for x in re.findall(r"\d", group)]
        for pos, cn in enumerate(cars, start=1):
            if cn in result:
                result[cn]["line_id"] = line_id
                result[cn]["line_pos"] = pos

    return list(result.values())


def parse_trifecta_odds(html: str) -> list[tuple[str, float]]:
    """Parse odds page. Returns list of (combo, odds) tuples."""
    soup = BeautifulSoup(html, "lxml")
    combos: dict[str, float] = {}

    # Method 1: rows with data-bet attribute
    for row in soup.select("table tr[data-bet], tr.odds-row"):
        combo_cell = row.select_one(".combo, td:first-child")
        odds_cell = row.select_one(".odds-val, td:last-child")
        if combo_cell and odds_cell:
            combo = combo_cell.get_text(strip=True)
            combo = re.sub(r"[－−ー→]", "-", combo)
            val_str = re.sub(r"[^\d.]", "", odds_cell.get_text())
            if re.match(r"\d-\d-\d", combo) and val_str:
                try:
                    combos[combo] = float(val_str)
                except ValueError:
                    pass

    # Method 2: scan all table cells for A-B-C patterns
    if len(combos) < 30:
        import pandas as pd
        from io import StringIO
        try:
            dfs = pd.read_html(StringIO(html), flavor="lxml")
            for df in dfs:
                for _, row in df.astype(str).iterrows():
                    vals = list(row.values)
                    for j, cell in enumerate(vals):
                        if re.match(r"^\d-\d-\d$", str(cell).strip()):
                            combo = str(cell).strip()
                            for v in vals[j + 1:]:
                                try:
                                    f = float(re.sub(r"[^\d.]", "", str(v)))
                                    if 1.0 < f < 100000:
                                        combos.setdefault(combo, f)
                                        break
                                except ValueError:
                                    pass
        except Exception:
            pass

    return list(combos.items())


def parse_result(html: str) -> dict[str, Any]:
    """Parse race result page. Returns {results: [...], payouts: [...]}."""
    soup = BeautifulSoup(html, "lxml")
    results: list[dict] = []
    payouts: list[dict] = []

    # Results table: typically first table with 着順 in header
    for table in soup.select("table"):
        header = table.get_text()
        if "着順" in header or "1着" in header:
            for row in table.select("tr"):
                cells = row.select("td")
                if len(cells) < 2:
                    continue
                finish = _to_int(_txt(cells, 0))
                car = _to_int(_txt(cells, 1))
                if finish and car:
                    kimarite_txt = _txt(cells, -1) if len(cells) > 5 else None
                    results.append({
                        "car_no": car, "finish": finish,
                        "kimarite": kimarite_txt, "time_sec": None,
                    })

    # Payouts
    for row in soup.select("table tr"):
        text = " ".join(c.get_text(strip=True) for c in row.select("td"))
        if "三連単" in text:
            m_combo = re.search(r"(\d[-－]?\d[-－]?\d)", text)
            m_yen = re.search(r"([\d,]+)\s*円", text)
            if m_combo and m_yen:
                combo = re.sub(r"[－−]", "-", m_combo.group(1))
                payouts.append({
                    "bet_type": "trifecta",
                    "combo": combo,
                    "payout_yen": int(m_yen.group(1).replace(",", "")),
                    "popularity": None,
                })

    return {"results": results, "payouts": payouts}


def parse_player_recent(html: str) -> list[dict]:
    """Parse player recent results page. Returns list of {race_date, venue, grade, finish}."""
    soup = BeautifulSoup(html, "lxml")
    items: list[dict] = []
    for row in soup.select("table tr"):
        cells = row.select("td")
        if len(cells) < 2:
            continue
        event_text = cells[0].get_text(strip=True)
        if not event_text or "レース名" in event_text:
            continue
        race_results: list[dict] = []
        for cell in cells[1:]:
            txt = cell.get_text(strip=True)
            m = re.match(r"([^\d]+)(\d+)", txt)
            if m:
                race_results.append({"type": m.group(1), "finish": int(m.group(2))})
        if race_results:
            items.append({"event": event_text[:40], "races": race_results})
    return items[:10]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _txt(cells, idx: int) -> str:
    try:
        return cells[idx].get_text(strip=True)
    except (IndexError, AttributeError):
        return ""


def _int(cells, idx: int) -> int:
    try:
        return int(re.sub(r"[^\d]", "", cells[idx].get_text()) or "0")
    except (ValueError, IndexError):
        return 0


def _to_int(s: str) -> int | None:
    try:
        return int(re.sub(r"[^\d]", "", s))
    except ValueError:
        return None


def _float(cells, idx: int) -> float:
    try:
        return float(re.sub(r"[^\d.]", "", cells[idx].get_text()) or "0")
    except (ValueError, IndexError):
        return 0.0


def _pct(cells, idx: int) -> float:
    try:
        txt = cells[idx].get_text(strip=True)
        f = float(re.sub(r"[^\d.]", "", txt))
        return f / 100 if f > 1 else f
    except (ValueError, IndexError):
        return 0.0


def _name(cells, idx: int) -> str:
    txt = _txt(cells, idx)
    m = re.search(r"[゠-ヿ]+\s+([一-龯々〆ヶ]{2,6})", txt)
    if m:
        return m.group(1)
    m2 = re.search(r"[一-龯々]{2,6}", txt)
    return m2.group() if m2 else txt[:8]


def _grade_cell(cells, idx: int) -> str:
    txt = _txt(cells, idx)
    normalized = txt.translate(str.maketrans("ＳＡ１２３", "SA123"))
    for g in ("SS", "S1", "S2", "A1", "A2", "A3"):
        if g in normalized:
            return g
    return "A1"
