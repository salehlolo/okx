import numpy as np
import pandas as pd
from typing import Tuple, Optional


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def bbands(close: pd.Series, n: int = 20, k: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ma = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0)
    upper = ma + k * sd
    lower = ma - k * sd
    return lower, ma, upper


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    dn = -delta.clip(upper=0.0)
    ma_up = up.ewm(alpha=1 / n, adjust=False).mean()
    ma_dn = dn.ewm(alpha=1 / n, adjust=False).mean()
    rs = ma_up / ma_dn.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def strat(df: pd.DataFrame) -> Optional[Tuple[str, float, float, str]]:
    """Return (side, stop, target, reason) or None."""
    if len(df) < 40:
        return None
    a = atr(df, 14).iloc[-1]
    if np.isnan(a) or a <= 0:
        return None
    lo, _, hi = bbands(df["close"], 20, 2.0)
    bb_lo = float(lo.iloc[-1])
    bb_hi = float(hi.iloc[-1])
    r = float(rsi(df["close"], 14).iloc[-1])
    px = float(df["close"].iloc[-1])
    tp, sl = 1.2 * a, 0.8 * a
    if px <= bb_lo and r <= 40:
        return ("buy", px - sl, px + tp, f"SCALP: px<=BBlo & RSI={r:.1f}")
    if px >= bb_hi and r >= 60:
        return ("sell", px + sl, px - tp, f"SCALP: px>=BBhi & RSI={r:.1f}")
    return None
