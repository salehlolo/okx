"""BracketManager for managing protective stop-loss and take-profit orders.

Example
-------
>>> from types import SimpleNamespace
>>> ex = SimpleNamespace(create_order=lambda *a, **k: {}, close_position=lambda *a, **k: {})
>>> cfg = SimpleNamespace(bracket_enabled=True, trail_enabled=True,
...                       trail_atr_mult=1.0, partial_tp_R_levels=(1.0,),
...                       partial_tp_sizes=(0.5,))
>>> bm = BracketManager(cfg, ex)
>>> bm.place("BTC/USDT", "buy", 1.0, 100.0, 95.0, 105.0)
True
"""
from __future__ import annotations

from typing import Any


class BracketManager:
    """Handles placement and maintenance of bracket orders on OKX."""

    def __init__(self, cfg: Any, exchange: Any) -> None:
        self.cfg = cfg
        self.ex = exchange
        self.enabled = getattr(cfg, "bracket_enabled", True)
        self.trail_enabled = getattr(cfg, "trail_enabled", True)
        self.trail_atr_mult = getattr(cfg, "trail_atr_mult", 1.0)
        self.partial_tp_R_levels = getattr(cfg, "partial_tp_R_levels", ())
        self.partial_tp_sizes = getattr(cfg, "partial_tp_sizes", ())

    def place(self, symbol: str, side: str, contract_qty: float,
              entry: float, sl: float, tp: float) -> bool:
        """Place both SL and TP orders. Returns True on success."""
        if not self.enabled:
            return True
        opp = "sell" if side == "buy" else "buy"
        params = {"reduceOnly": True}
        try:
            self.ex.create_order(symbol, opp, contract_qty, type="stop", price=sl, params=params)
            self.ex.create_order(symbol, opp, contract_qty, type="take_profit", price=tp, params=params)
            return True
        except Exception:
            try:
                self.ex.close_position(symbol, opp, contract_qty)
            except Exception:
                pass
            return False

    def trail(self, trade: Any, last_row: Any) -> None:
        """Adjust stop-loss using ATR trailing logic."""
        if not (self.enabled and self.trail_enabled):
            return
        atr = getattr(last_row, "get", lambda k, d=None: last_row[k])("atr", None)
        if atr is None or atr <= 0:
            return
        opp = "sell" if trade.side == "buy" else "buy"
        params = {"reduceOnly": True}
        if trade.side == "buy":
            ref = getattr(trade, "highest_since_entry", None)
            if ref is None:
                return
            candidate = ref - atr * self.trail_atr_mult
            if candidate > trade.sl:
                trade.sl = candidate
                self._amend_stop(trade.symbol, opp, trade.contract_qty, candidate, params)
        else:
            ref = getattr(trade, "lowest_since_entry", None)
            if ref is None:
                return
            candidate = ref + atr * self.trail_atr_mult
            if candidate < trade.sl:
                trade.sl = candidate
                self._amend_stop(trade.symbol, opp, trade.contract_qty, candidate, params)

    def _amend_stop(self, symbol: str, side: str, qty: float, price: float, params: dict) -> None:
        try:
            self.ex.create_order(symbol, side, qty, type="stop", price=price, params=params)
        except Exception:
            pass

    def maybe_partial_take_profit(self, trade: Any, last_price: float) -> None:
        """Close portions of the position when R levels are reached."""
        if not self.enabled:
            return
        risk = abs(trade.entry - trade.sl)
        if not risk:
            return
        opp = "sell" if trade.side == "buy" else "buy"
        R = (last_price - trade.entry) / risk if trade.side == "buy" else (trade.entry - last_price) / risk
        for lvl, size in zip(self.partial_tp_R_levels, self.partial_tp_sizes):
            key = f"ptp@{lvl}"
            if R >= lvl and not trade.meta.get(key):
                qty = trade.contract_qty * size
                try:
                    self.ex.close_position(trade.symbol, opp, qty)
                    trade.contract_qty -= qty
                    trade.meta[key] = True
                except Exception:
                    pass
