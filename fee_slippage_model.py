"""FeeSlippageModel simulates trading costs in the paper engine.

Example
-------
>>> from types import SimpleNamespace
>>> model = FeeSlippageModel(SimpleNamespace())
>>> model.fill_market("buy", 100)
100.007
>>> round(model.pnl_after_costs(100, 110, 1, "buy"), 4)
9.9895
"""
from __future__ import annotations

from typing import Any


class FeeSlippageModel:
    """Compute basic fees and slippage adjustments."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.fees_taker_bps = getattr(cfg, "fees_taker_bps", 5)
        self.fees_maker_bps = getattr(cfg, "fees_maker_bps", 2)
        self.slippage_bps_market = getattr(cfg, "slippage_bps_market", 7)

    def taker_fee(self) -> float:
        return self.fees_taker_bps / 10_000

    def maker_fee(self) -> float:
        return self.fees_maker_bps / 10_000

    def slip(self) -> float:
        return self.slippage_bps_market / 10_000

    def fill_market(self, side: str, px: float) -> float:
        s = self.slip()
        return px * (1 + s) if side == "buy" else px * (1 - s)

    def pnl_after_costs(self, entry_px: float, exit_px: float, qty: float, side: str) -> float:
        if side == "buy":
            gross = (exit_px - entry_px) * qty
        else:
            gross = (entry_px - exit_px) * qty
        notional = abs(entry_px * qty) + abs(exit_px * qty)
        costs = notional * self.taker_fee()
        return gross - costs
