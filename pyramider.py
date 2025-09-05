"""Pyramider module to scale into winning trades.

Example
-------
>>> from types import SimpleNamespace
>>> ex = SimpleNamespace(create_order=lambda *a, **k: {})
>>> cfg = SimpleNamespace(pyramid_enabled=True, pyramid_R_levels=(0.5,),
...                       pyramid_sizes=(0.5,))
>>> trade = SimpleNamespace(symbol="BTC/USDT", side="buy", entry=100,
...                          sl=90, contract_qty=1, initial_contract_qty=1,
...                          meta={})
>>> py = Pyramider(cfg, ex)
>>> py.maybe_add(trade, 110)
>>> trade.contract_qty
1.5
"""
from __future__ import annotations

from typing import Any


class Pyramider:
    """Scale into profitable positions based on R multiples."""

    def __init__(self, cfg: Any, exchange: Any) -> None:
        self.cfg = cfg
        self.ex = exchange
        self.enabled = getattr(cfg, "pyramid_enabled", True)
        self.levels = getattr(cfg, "pyramid_R_levels", ())
        self.sizes = getattr(cfg, "pyramid_sizes", ())

    def maybe_add(self, trade: Any, last_price: float) -> None:
        if not self.enabled:
            return
        risk = abs(trade.entry - trade.sl)
        if risk <= 0:
            return
        R = (last_price - trade.entry) / risk if trade.side == "buy" else (trade.entry - last_price) / risk
        for idx, (lvl, size) in enumerate(zip(self.levels, self.sizes)):
            key = f"py@{lvl}"
            if R >= lvl and not trade.meta.get(key):
                qty = trade.initial_contract_qty * size
                try:
                    self.ex.create_order(trade.symbol, trade.side, qty, type="market")
                    trade.contract_qty += qty
                    trade.meta[key] = True
                    if idx == 0:
                        trade.sl = trade.entry
                except Exception:
                    pass
