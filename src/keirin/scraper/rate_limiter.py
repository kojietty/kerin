"""Polite rate limiting + robots.txt enforcement.

Token-bucket style: sleeps until the configured minimum interval has elapsed
since the last call. Adds jitter to avoid synchronized request patterns.
"""
from __future__ import annotations

import logging
import random
import threading
import time
import urllib.robotparser
from dataclasses import dataclass
from urllib.parse import urlparse

log = logging.getLogger(__name__)


@dataclass
class RateLimiter:
    interval_sec: float
    jitter_sec: float = 0.5
    _last_call: float = 0.0
    _lock: threading.Lock = threading.Lock()  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            target = self.interval_sec + random.uniform(0, self.jitter_sec)
            if elapsed < target:
                time.sleep(target - elapsed)
            self._last_call = time.monotonic()


class RobotsCache:
    """Caches a single domain's robots.txt for the process lifetime."""

    def __init__(self, base_url: str, user_agent: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.user_agent = user_agent
        self._rp: urllib.robotparser.RobotFileParser | None = None

    def _load(self) -> urllib.robotparser.RobotFileParser:
        if self._rp is not None:
            return self._rp
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(self.base_url + "/robots.txt")
        try:
            rp.read()
        except Exception as e:
            log.warning("robots.txt fetch failed for %s: %s — defaulting to allow", self.base_url, e)
        self._rp = rp
        return rp

    def allowed(self, url: str) -> bool:
        rp = self._load()
        try:
            return rp.can_fetch(self.user_agent, url)
        except Exception:
            return True


class DailyRequestCap:
    def __init__(self, cap: int) -> None:
        self.cap = cap
        self._count = 0
        self._date = ""
        self._lock = threading.Lock()

    def check_and_increment(self) -> None:
        import datetime as _dt
        today = _dt.date.today().isoformat()
        with self._lock:
            if self._date != today:
                self._date = today
                self._count = 0
            self._count += 1
            if self._count > self.cap:
                raise RuntimeError(
                    f"Daily request cap exceeded: {self._count} > {self.cap}. "
                    "Aborting to remain a polite citizen."
                )


def host_of(url: str) -> str:
    return urlparse(url).netloc
