#!/usr/bin/env python3
"""Bollinger Bands + RSI scalping strategy."""
from typing import Tuple, Optional

import numpy as np
import pandas as pd


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average True Range."""
    high, low, close = df["high"], df["low"], df["close"]
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def bbands(close: pd.Series, n: int = 20, k: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Lower, middle and upper Bollinger Bands."""
    ma = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0)
    upper = ma + k * sd
    lower = ma - k * sd
    return lower, ma, upper


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = close.diff()
    up = delta.clip(lower=0.0)
    dn = -delta.clip(upper=0.0)
    ma_up = up.ewm(alpha=1 / n, adjust=False).mean()
    ma_dn = dn.ewm(alpha=1 / n, adjust=False).mean()
    rs = ma_up / ma_dn.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def strat(df: pd.DataFrame) -> Optional[Tuple[str, float, float, str]]:
    """Return trading signal based on Bollinger Bands and RSI."""
    if len(df) < 40:
        return None
    a = atr(df, 14).iloc[-1]
    if np.isnan(a) or a <= 0:
        return None
    lo = df["close"].rolling(20).mean() - 2.0 * df["close"].rolling(20).std(ddof=0)
    hi = df["close"].rolling(20).mean() + 2.0 * df["close"].rolling(20).std(ddof=0)
    bb_lo = float(lo.iloc[-1])
    bb_hi = float(hi.iloc[-1])
    delta = df["close"].diff()
    up = delta.clip(lower=0.0)
    dn = -delta.clip(upper=0.0)
    r = float(
        100
        - (
            100
            / (
                1
                + (
                    up.ewm(alpha=1 / 14, adjust=False).mean()
                    / (dn.replace(0, np.nan).ewm(alpha=1 / 14, adjust=False).mean())
                )
            )
        )
    ).iloc[-1]
    px = float(df["close"].iloc[-1])
    tp, sl = 1.2 * a, 0.8 * a
    if px <= bb_lo and r <= 40:
        return ("buy", px - sl, px + tp, f"SCALP: px<=BBlo & RSI={r:.1f}")
    if px >= bb_hi and r >= 60:
        return ("sell", px + sl, px - tp, f"SCALP: px>=BBhi & RSI={r:.1f}")
    return None


if __name__ == "__main__":
    # Example usage with dummy data
    data = {
        "open": [1, 1, 1],
        "high": [1, 1, 1],
        "low": [1, 1, 1],
        "close": [1, 1, 1],
        "volume": [1, 1, 1],
    }
    df = pd.DataFrame(data)
    print(strat(df))
