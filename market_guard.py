"""MarketGuard emergency exit module.

Provides a lightweight gate that can be called for each open trade to decide
whether it should be force-closed due to sudden volatility spikes, regime
changes or hot news events.

Example
-------
>>> import pandas as pd
>>> from types import SimpleNamespace
>>> row = pd.Series({"atr": 5, "atr_sma_50": 2})
>>> trade = SimpleNamespace(reason="MR bounce", meta={})
>>> guard = MarketGuard(SimpleNamespace())
>>> guard.check("BTC/USDT", row, trade)
(True, 'ATR spike')
"""
from __future__ import annotations

from typing import Tuple, Any
import pandas as pd


class MarketGuard:
    """Emergency exit checks for open trades."""

    def __init__(self, cfg: Any, news: Any | None = None) -> None:
        self.cfg = cfg
        self.news = news
        self.enabled = getattr(cfg, "market_guard_enabled", True)
        self.atr_spike_mult = getattr(cfg, "atr_spike_mult", 1.8)
        self.atr_spike_len = getattr(cfg, "atr_spike_len", 50)
        self.regime_force_exit = getattr(cfg, "regime_force_exit", True)

    def check(self, symbol: str, last_row: pd.Series, trade: Any) -> Tuple[bool, str]:
        """Return (should_exit, reason)."""
        if not self.enabled:
            return False, ""

        # 1) ATR spike detection
        try:
            atr_val = float(last_row.get("atr", 0))
            mean_col = f"atr_sma_{self.atr_spike_len}"
            mean_val = last_row.get(mean_col)
            if pd.isna(mean_val):
                mean_col = f"atr_mean_{self.atr_spike_len}"
                mean_val = last_row.get(mean_col)
            if atr_val > 0 and mean_val and not pd.isna(mean_val):
                if atr_val > self.atr_spike_mult * float(mean_val):
                    return True, "ATR spike"
        except Exception:
            pass

        # 2) Regime switch for mean-reversion trades
        if self.regime_force_exit:
            reason = str(getattr(trade, "reason", "")).lower()
            if reason.startswith("mr") or "mean" in reason:
                try:
                    regime = classify_regime(last_row, self.cfg)  # type: ignore
                except Exception:
                    regime = None
                label = None
                if isinstance(regime, str):
                    label = regime
                elif regime is not None:
                    label = str(getattr(regime, "vol_bucket", regime))
                if label and "high" in label:
                    return True, "Regime switch"

        # 3) News filter
        if self.news is not None:
            base = symbol.split(":")[0].split("/")[0]
            try:
                if self.news.too_hot(base):
                    return True, "News hot"
            except Exception:
                pass

        return False, ""
