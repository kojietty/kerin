"""HTTP session with retries, rate limiting, robots, audit logging, and raw-HTML cache."""
from __future__ import annotations

import gzip
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from keirin.db.repository import log_fetch, recently_fetched
from keirin.scraper.rate_limiter import DailyRequestCap, RateLimiter, RobotsCache

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lightweight fetch helpers (shared by ci_morning / ci_evening / backfill_*)
#
# 旧スクリプトの _get() は apparent_encoding が windows-1252 を返すと cp932 を
# 強制し、UTF-8 ページを誤デコードして style 等を文字化けさせていた
# (騾�/霑ｽ/荳｡)。ここでは「UTF-8 を厳密に試す → 失敗時のみ cp932 → euc-jp」の
# 決定的な判定にする。UTF-8 は厳密デコードなので cp932 ページを誤って通すことは
# 実質ない。
# ---------------------------------------------------------------------------

_DECODE_CANDIDATES = ("utf-8", "cp932", "euc-jp")


def decode_bytes(content: bytes) -> str:
    for enc in _DECODE_CANDIDATES:
        try:
            return content.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return content.decode("utf-8", errors="replace")


def fetch_text(url: str, *, headers: dict | None = None, timeout: int = 20) -> str:
    """GET url and return decoded text ("" on failure)."""
    try:
        r = requests.get(url, headers=headers or {}, timeout=timeout)
        return decode_bytes(r.content)
    except Exception as e:  # network errors, timeouts
        log.warning("fetch_text failed %s: %s", url[:60], e)
        return ""


class FetchBlocked(RuntimeError):
    """Raised when robots.txt or our own policy disallows a URL."""


@dataclass
class FetchResult:
    url: str
    status: int
    text: str
    from_cache: bool


class PoliteSession:
    def __init__(
        self,
        *,
        base_url: str,
        user_agent: str,
        rate_limiter: RateLimiter,
        robots: RobotsCache,
        daily_cap: DailyRequestCap,
        cache_dir: Path | None,
        engine,  # SQLAlchemy engine, kept untyped to avoid circular import
        timeout_sec: int = 20,
        respect_robots: bool = True,
    ) -> None:
        self.base_url = base_url
        self._rate = rate_limiter
        self._robots = robots
        self._cap = daily_cap
        self._cache_dir = cache_dir
        self._engine = engine
        self._timeout = timeout_sec
        self._respect_robots = respect_robots

        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "ja,en;q=0.8",
            }
        )

    def get(self, url: str, *, skip_cache: bool = False, cache_subdir: str | None = None) -> FetchResult:
        if self._respect_robots and not self._robots.allowed(url):
            raise FetchBlocked(f"Blocked by robots.txt: {url}")

        if not skip_cache and recently_fetched(self._engine, url, within_hours=24):
            cached = self._read_cache(url, cache_subdir)
            if cached is not None:
                log.debug("cache hit: %s", url)
                return FetchResult(url=url, status=200, text=cached, from_cache=True)

        self._cap.check_and_increment()
        self._rate.wait()
        return self._do_get(url, cache_subdir=cache_subdir)

    @retry(
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2.0, min=2.0, max=20.0),
        reraise=True,
    )
    def _do_get(self, url: str, *, cache_subdir: str | None) -> FetchResult:
        log.info("GET %s", url)
        resp = self._session.get(url, timeout=self._timeout)
        try:
            log_fetch(self._engine, url, resp.status_code, len(resp.content))
        except Exception as e:
            log.warning("fetch_log insert failed: %s", e)

        if resp.status_code >= 500:
            resp.raise_for_status()  # triggers retry
        if resp.status_code >= 400:
            return FetchResult(url=url, status=resp.status_code, text="", from_cache=False)

        resp.encoding = resp.apparent_encoding or "utf-8"
        text = resp.text
        self._write_cache(url, text, cache_subdir)
        return FetchResult(url=url, status=resp.status_code, text=text, from_cache=False)

    # -- raw HTML cache ----------------------------------------------------

    def _cache_path(self, url: str, cache_subdir: str | None) -> Path | None:
        if self._cache_dir is None:
            return None
        sub = self._cache_dir / (cache_subdir or "misc")
        sub.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
        return sub / f"{h}.html.gz"

    def _write_cache(self, url: str, text: str, cache_subdir: str | None) -> None:
        path = self._cache_path(url, cache_subdir)
        if path is None:
            return
        try:
            with gzip.open(path, "wb") as f:
                f.write(text.encode("utf-8"))
        except Exception as e:
            log.warning("cache write failed for %s: %s", url, e)

    def _read_cache(self, url: str, cache_subdir: str | None) -> str | None:
        path = self._cache_path(url, cache_subdir)
        if path is None or not path.exists():
            return None
        try:
            with gzip.open(path, "rb") as f:
                return f.read().decode("utf-8")
        except Exception as e:
            log.warning("cache read failed for %s: %s", url, e)
            return None
