"""ReEntryEngine to throttle quick re-entries after closing trades.

Example
-------
>>> from types import SimpleNamespace
>>> engine = ReEntryEngine(SimpleNamespace(reentry_cooldown_sec=1))
>>> engine.record_close("BTC/USDT", "buy")
>>> engine.allow("BTC/USDT", "buy")
False
"""
from __future__ import annotations

import time
from typing import Any, Dict, Tuple


class ReEntryEngine:
    """Track last close times and enforce a cooldown."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.enabled = getattr(cfg, "reentry_enabled", True)
        self.cooldown = getattr(cfg, "reentry_cooldown_sec", 10)
        self._last_close: Dict[Tuple[str, str], float] = {}

    def record_close(self, symbol: str, side: str) -> None:
        if not self.enabled:
            return
        self._last_close[(symbol, side)] = time.time()

    def allow(self, symbol: str, side: str) -> bool:
        if not self.enabled:
            return True
        ts = self._last_close.get((symbol, side))
        if ts is None:
            return True
        return (time.time() - ts) >= self.cooldown
