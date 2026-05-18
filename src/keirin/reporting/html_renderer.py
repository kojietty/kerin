"""Render the daily HTML dashboard from a structured payload."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from keirin.config import HARD_DAILY_BUDGET_YEN, HARD_EMERGENCY_STOP_YEN

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_ASSETS_DIR = Path(__file__).parent / "assets"


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "j2"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _ev_class(ev: float) -> int:
    if ev >= 1.60:
        return 3
    if ev >= 1.30:
        return 2
    return 1


def render_dashboard(payload: dict[str, Any]) -> str:
    """Render the dashboard HTML to a string. Payload is shaped like:

    {
      "date_label": "2026-05-18 (Mon)",
      "generated_at": "2026-05-18 18:00 JST",
      "mtd_pnl": int,
      "mtd_roi": float | None,
      "mtd_hit_rate": float | None,
      "daily_invest": int,
      "daily_remaining": int,
      "daily_cap": int,
      "total_races": int,
      "recommended": [
        {
          "venue_name": str, "race_no": int,
          "grade": str|None, "race_class": str|None,
          "post_time": str|None, "distance_m": int|None, "weather": str|None,
          "lines": [[1,3,5], [2,7], [4,6]],
          "axis": {"car_no": int, "name": str, "note": str|None} | None,
          "picks": [
            {"cars": [3,5,1], "odds": 38.4, "prob": 0.041, "ev": 1.57, "stake_yen": 100, "hit": None}
          ],
          "max_ev": float, "total_stake": int
        }, ...
      ],
      "skipped": [{"venue_name": str, "race_no": int, "max_ev": float | None}, ...],
      "ev_threshold": float,
      "emergency_stop": bool
    }
    """
    enriched = []
    for r in payload.get("recommended", []):
        new_picks = []
        for p in r["picks"]:
            new_picks.append({**p, "ev_class": _ev_class(p["ev"])})
        enriched.append({**r, "picks": new_picks})

    ctx = {
        **payload,
        "recommended": enriched,
        "recommended_count": len(enriched),
        "total_picks": sum(len(r["picks"]) for r in enriched),
        "inline_css": _ASSETS_DIR.joinpath("style.css").read_text(encoding="utf-8"),
        "daily_cap": payload.get("daily_cap", HARD_DAILY_BUDGET_YEN),
        "emergency_threshold": HARD_EMERGENCY_STOP_YEN,
        "ev_threshold": payload.get("ev_threshold", 1.10),
    }
    template = _env().get_template("dashboard.html.j2")
    return template.render(**ctx)


def write_dashboard(out_path: Path, payload: dict[str, Any]) -> Path:
    html = render_dashboard(payload)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def example_payload(date_label: str | None = None) -> dict[str, Any]:
    """A small sample payload to verify the renderer end-to-end without real data."""
    return {
        "date_label": date_label or datetime.now().strftime("%Y-%m-%d"),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M JST"),
        "mtd_pnl": 1820,
        "mtd_roi": 1.12,
        "mtd_hit_rate": 0.18,
        "daily_invest": 400,
        "daily_remaining": 600,
        "daily_cap": HARD_DAILY_BUDGET_YEN,
        "total_races": 84,
        "ev_threshold": 1.10,
        "emergency_stop": False,
        "recommended": [
            {
                "venue_name": "小倉",
                "race_no": 11,
                "grade": "S級",
                "race_class": "決勝",
                "post_time": "18:45",
                "distance_m": 400,
                "weather": "晴",
                "lines": [[3, 1, 5], [2, 7], [4, 6]],
                "axis": {"car_no": 3, "name": "◯◯◯◯", "note": "地元3着内率 68%"},
                "picks": [
                    {"cars": [3, 5, 1], "odds": 38.4, "prob": 0.041, "ev": 1.57, "stake_yen": 100, "hit": None},
                    {"cars": [3, 1, 5], "odds": 42.7, "prob": 0.035, "ev": 1.49, "stake_yen": 100, "hit": None},
                    {"cars": [3, 1, 2], "odds": 55.0, "prob": 0.025, "ev": 1.38, "stake_yen": 100, "hit": None},
                    {"cars": [3, 2, 1], "odds": 78.2, "prob": 0.016, "ev": 1.22, "stake_yen": 100, "hit": None},
                ],
                "max_ev": 1.57,
                "total_stake": 400,
            }
        ],
        "skipped": [
            {"venue_name": "函館", "race_no": 9, "max_ev": 0.82},
            {"venue_name": "川崎", "race_no": 5, "max_ev": 0.94},
            {"venue_name": "前橋", "race_no": 7, "max_ev": 1.03},
            {"venue_name": "松山", "race_no": 11, "max_ev": 0.71},
        ],
    }
