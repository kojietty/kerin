"""Config loader. Loads YAML from configs/ and exposes typed access via pydantic."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


class Paths(BaseModel):
    db: Path = Path("data/keirin.db")
    raw_html: Path = Path("data/raw")
    reports: Path = Path("reports")
    models: Path = Path("models")
    logs: Path = Path("logs")


class ScraperCfg(BaseModel):
    base_url: str = "https://keirin.jp"
    user_agent: str
    request_interval_sec: float = 2.0
    jitter_sec: float = 0.5
    daily_request_cap: int = 5000
    retry_max_attempts: int = 3
    retry_backoff_base_sec: float = 2.0
    timeout_sec: int = 20
    respect_robots: bool = True
    cache_raw_html: bool = True


class LoggingCfg(BaseModel):
    level: str = "INFO"
    rotation_mb: int = 10
    retention_days: int = 30


class ReportingCfg(BaseModel):
    output_html: bool = True
    output_markdown: bool = True
    timezone: str = "Asia/Tokyo"


class AppConfig(BaseModel):
    paths: Paths = Field(default_factory=Paths)
    scraper: ScraperCfg
    logging: LoggingCfg = Field(default_factory=LoggingCfg)
    reporting: ReportingCfg = Field(default_factory=ReportingCfg)


# -- Betting config ---------------------------------------------------------

# These caps are intentionally hardcoded constants. The values are duplicated
# in betting.yaml for visibility, but the CLI must never override these.
HARD_DAILY_BUDGET_YEN = 1000
HARD_EMERGENCY_STOP_YEN = -3000


class BettingCfg(BaseModel):
    ev_threshold: float = 1.10
    odds_min: float = 10.0
    odds_max: float = 100.0
    prob_min: float = 0.005
    max_picks_per_race: int = 10
    min_picks_per_race: int = 0
    daily_budget_yen: int = HARD_DAILY_BUDGET_YEN
    per_bet_min_yen: int = 100
    kelly_fraction: float = 0.25
    skip_if_no_pick: bool = True
    emergency_stop_drawdown_yen: int = HARD_EMERGENCY_STOP_YEN
    combo_method: str = "plackett_luce"

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        # Enforce hardcoded ceilings regardless of YAML/CLI input.
        if self.daily_budget_yen > HARD_DAILY_BUDGET_YEN:
            self.daily_budget_yen = HARD_DAILY_BUDGET_YEN
        if self.emergency_stop_drawdown_yen < HARD_EMERGENCY_STOP_YEN:
            self.emergency_stop_drawdown_yen = HARD_EMERGENCY_STOP_YEN


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


@lru_cache(maxsize=1)
def load_app_config() -> AppConfig:
    raw = _read_yaml(CONFIG_DIR / "config.yaml")
    raw.setdefault("scraper", {})
    ua_env = os.environ.get("KEIRIN_USER_AGENT")
    if ua_env:
        raw["scraper"]["user_agent"] = ua_env
    raw["scraper"].setdefault(
        "user_agent",
        "KeirinResearchBot/0.1 (personal research; contact: pagudaruma@gmail.com)",
    )
    db_env = os.environ.get("KEIRIN_DB_PATH")
    if db_env:
        raw.setdefault("paths", {})["db"] = db_env
    cfg = AppConfig.model_validate(raw)
    # Resolve relative paths against project root for stable behavior.
    for field in ("db", "raw_html", "reports", "models", "logs"):
        p = getattr(cfg.paths, field)
        if not p.is_absolute():
            setattr(cfg.paths, field, (PROJECT_ROOT / p).resolve())
    return cfg


@lru_cache(maxsize=1)
def load_betting_config() -> BettingCfg:
    raw = _read_yaml(CONFIG_DIR / "betting.yaml")
    return BettingCfg.model_validate(raw)
