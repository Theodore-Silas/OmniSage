"""
Self-healing locator cache (v4.0).

Remembers the last *stable* CSS selector that worked for a
(url, element-description) pair, so repeated tasks on the same site avoid
re-doing fragile text/role matching. Entries carry a timestamp and are
evicted on TTL or capacity overflow.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

from src.browser.driver.base import ElementTarget


class LocatorCache:
    """Tiny TTL-bounded store mapping (url, target) → stable css selector."""

    def __init__(self, max_entries: int = 300, ttl: int = 3600):
        self._store: Dict[str, dict] = {}
        self._max = max_entries
        self._ttl = ttl

    @staticmethod
    def _key(url: str, target: ElementTarget) -> str:
        desc = target.name or target.text or target.css or target.role or target.xpath or ""
        return f"{url}::{desc}"

    def get_css(self, url: str, target: ElementTarget) -> Optional[str]:
        entry = self._store.get(self._key(url, target))
        if not entry:
            return None
        if time.time() - entry["ts"] > self._ttl:
            self._store.pop(self._key(url, target), None)
            return None
        return entry["css"]

    def put_css(self, url: str, target: ElementTarget, css: Optional[str]) -> None:
        if not css:
            return
        self._store[self._key(url, target)] = {"css": css, "ts": time.time()}
        if len(self._store) > self._max:
            oldest = min(self._store, key=lambda k: self._store[k]["ts"])
            self._store.pop(oldest, None)

    def __len__(self) -> int:
        return len(self._store)
