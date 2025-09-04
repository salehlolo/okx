#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
bot.py — Triple+3 Strategies (Self-Evolving) Scalper — OKX USDT Swap
(نسخة بدون أي تكامل مع OpenAI — تداول/إشعارات فقط)

تشغيل تجريبي على بيئة الديمو الخاصة بـOKX.
Env:
  OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSWORD
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
  (اختياري) CRYPTOPANIC_TOKEN, NEWSAPI_KEY  ← تقدر تسيبهم فاضيين
"""

import os, time, json, argparse, datetime as dt, math
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import ccxt
import ta
import requests

# =========================
# Constants
# =========================

# Each trade uses 50 USDT of margin at 10× leverage (~500 USDT notional)
TRADE_MARGIN_USD = 50.0
LEVERAGE = 10

# =========================
# Helpers
# =========================

def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)

def fmt_ts(ts: Optional[dt.datetime] = None) -> str:
    t = (ts or now_utc())
    s = t.isoformat()
    return s.replace("+00:00", "Z")

def ensure_dir(path: str):
    d = path if os.path.splitext(path)[1] == "" else os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def clamp(v, lo, hi): return max(lo, min(hi, v))
def safe_float(x, default=np.nan):
    try: return float(x)
    except Exception: return default
def pct(n): return f"{n*100:.2f}%"

# =========================
# Config
# =========================

@dataclass
class Config:
    timeframe: str = "15m"
    lookback: int = 800

    # Indicators / windows
    ema_fast: int = 9
    ema_slow: int = 21
    atr_window: int = 14
    rsi_len: int = 14
    bb_len: int = 20
    bb_std: float = 2.0
    vol_ma_len: int = 30
    box_len: int = 20
    regime_lookback: int = 120
    low_vol_pct: float = 0.35
    high_vol_pct: float = 0.70

    # Fixed TP/SL (يستخدم كخيار احتياطي)
    fixed_tp_pct: float = 0.01
    fixed_sl_pct: float = 0.005

    # ==== تعديل #2: إضافة إعدادات TP/SL الديناميكية ====
    use_atr_tp_sl: bool = True    # لتفعيل/تعطيل الميزة بسهولة
    atr_tp_mult: float = 2.0      # TP = السعر + (2.0 * ATR)
    atr_sl_mult: float = 1.2      # SL = السعر - (1.2 * ATR)
    
    # TREND
    trend_min_slope: float = 0.0003
    trend_vol_mult: float = 1.3

    # BO
    bo_vol_mult: float = 1.2
    bo_range_share: float = 0.5

    # MR
    mr_rsi_buy: float = 25.0
    mr_rsi_sell: float = 75.0
    sr_lookback: int = 50

    # PB (Pullback)
    pb_pullback_pct: float = 0.0035
    pb_wick_ratio: float = 0.35

    # VWAP-R
    vwap_dev_mult: float = 1.5

    # KSQ (Keltner Squeeze)
    keltner_len: int = 20
    keltner_mult: float = 1.5
    squeeze_bb_mult: float = 1.6

    # Sizing
    max_open_trades: int = 2

    # Filters
    funding_filter: bool = True
    max_abs_funding: float = 0.003

    # Account modes (override if auto-detect fails)
    okx_pos_mode: Optional[str] = None   # "net_mode" or "long_short_mode"
    okx_margin_mode: Optional[str] = None  # "cross" or "isolated"


    # Quiet windows (UTC HH:MM)
    event_quiet_minutes: int = 10
    quiet_windows_utc: Tuple[str, ...] = ()

    # News Guard — افتراضيًا مقفول
    news_enabled: bool = False
    news_lookback_minutes: int = 60
    news_keywords: Tuple[str, ...] = ("ETF","hack","exploit","ban","SEC","lawsuit","fork","upgrade","halving")

    # Universe
    # ==== تعديل #1: تغيير العدد إلى 10 ====
    top_n_symbols: int = 10
    refresh_universe_minutes: int = 360
    health_refresh_minutes: int = 90
    health_test_limit: int = 50

    # Telegram
    telegram_enabled: bool = True
    telegram_token: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat_id: Optional[str] = os.getenv("TELEGRAM_CHAT_ID")

    # Throttles
    min_minutes_between_same_signal: int = 3
    min_seconds_between_alerts_global: int = 50

    # Committee & Bandit
    exploration_eps: float = 0.08
    dyn_quorum_base: float = 0.55
    quorum_boost_high_vol: float = -0.07
    quorum_boost_good_hit: float = -0.05
    quorum_penalty_bad_hit: float = 0.05

    # Self-Evolving (محلي — مش بيعتمد على OpenAI)
    evolve_enabled: bool = True
    evolve_mutations_per_round: int = 2
    evolve_trial_weight: float = 0.25
    evolve_decay: float = 0.98

    # ==== الإضافات الجديدة (إدارة المخاطر المتقدمة) ====
    # Confidence Filter
    min_confidence_accept: float = 0.75

    # Committee Override: لازم أقل حاجة X نماذج تتفق على نفس الاتجاه
    committee_min_agree: int = 2

    # Files
    logs_dir: str = "./logs"
    signals_csv: str = "./logs/signals_log.csv"
    trades_csv: str  = "./logs/trades_log.csv"
    models_csv: str  = "./logs/models_log.csv"
    ml_csv: str      = "./logs/ml_dataset.csv"
    state_json: str  = "./logs/state.json"

# =========================
# Telegram
# =========================

class Notifier:
    def __init__(self, cfg: Config):
        self.enabled = bool(cfg.telegram_enabled and cfg.telegram_token and cfg.telegram_chat_id)
        self.base = f"https://api.telegram.org/bot{cfg.telegram_token}" if self.enabled else None
        self.chat_id = cfg.telegram_chat_id
    def send(self, text: str):
        if not self.enabled:
            print(text)
            return
        try:
            r = requests.post(f"{self.base}/sendMessage",
                              json={"chat_id": self.chat_id, "text": text, "disable_web_page_preview": True},
                              timeout=10)
            if r.status_code != 200:
                print("[WARN] Telegram send failed:", r.text)
        except Exception as e:
            print("[WARN] Telegram exception:", e)

# =========================
# Exchange
# =========================

class FuturesExchange:
    def __init__(self, cfg: Config):
        key = os.getenv("OKX_API_KEY")
        secret = os.getenv("OKX_API_SECRET")
        password = os.getenv("OKX_API_PASSWORD") or os.getenv("OKX_API_PASSPHRASE")
        self.x = ccxt.okx({
            "apiKey": key,
            "secret": secret,
            "password": password,
            # Use swap markets in OKX demo environment
            "options": {
                "defaultType": "swap",
                "demo": True,
            },
            "headers": {"x-simulated-trading": "1"},
            "enableRateLimit": True,
            "timeout": 15000,
        })
        # Force all requests to hit the demo environment
        try:
            self.x.set_sandbox_mode(True)
        except Exception:
            pass
        # Demo accounts cannot access the private currencies endpoint; disable it (نغلق من الجهتين)
        try:
            self.x.options["fetchCurrencies"] = False
        except Exception:
            pass
        try:
            self.x.has["fetchCurrencies"] = False
        except Exception:
            pass
        self.x.load_markets()

        # Determine account modes for proper order parameters
        self.pos_mode = "net"        # "net" أو "long_short"
        self.margin_mode = "cross"   # "cross" أو "isolated"
        self.leverage = LEVERAGE
        try:
            info = self.x.privateGetAccountConfig()
            data = info.get("data", [])
            if data:
                cfg0 = data[0]
                pm = cfg0.get("posMode") or self.pos_mode
                mm = cfg0.get("marginMode") or cfg0.get("mgnMode") or self.margin_mode
                self.pos_mode = str(pm).replace("-", "_").lower()
                self.margin_mode = str(mm).lower()
        except Exception as e:
            print("[WARN] fetch account config failed:", e)
        # Manual overrides from config
        if cfg.okx_pos_mode:
            self.pos_mode = str(cfg.okx_pos_mode).replace("-", "_").lower()
        if cfg.okx_margin_mode:
            self.margin_mode = str(cfg.okx_margin_mode).lower()

        # Ensure (or set) account modes to avoid 51010
        try:
            setup_symbol = next((s for s in self.x.symbols if s.endswith(":USDT")), "BTC/USDT:USDT")
            hedged = not self.pos_mode.startswith("net")
            try:
                self.x.set_position_mode(hedged)
            except Exception:
                self.x.privatePostAccountSetPositionMode({
                    "posMode": "long_short_mode" if hedged else "net_mode"
                })
            try:
                self.x.set_leverage(self.leverage, setup_symbol, {"mgnMode": self.margin_mode})
            except Exception:
                m = self.x.market(setup_symbol)
                self.x.privatePostAccountSetLeverage({
                    "instId": m["id"],
                    "lever": str(self.leverage),
                    "mgnMode": self.margin_mode,
                })
        except Exception as e:
            msg = str(getattr(e, 'args', [''])[0])
            if "59000" in msg:
                print("[WARN] ensure account modes failed: open orders or positions exist; attempting to flatten")
                self.flatten_all()
                try:
                    self.x.set_position_mode(hedged)
                    self.x.set_leverage(self.leverage, setup_symbol, {"mgnMode": self.margin_mode})
                except Exception as e2:
                    msg2 = str(getattr(e2, 'args', [''])[0])
                    if "59000" in msg2:
                        print("[WARN] ensure account modes skipped: close orders/positions before changing modes (code 59000)")
                    else:
                        print("[WARN] ensure account modes failed:", e2)
            else:
                print("[WARN] ensure account modes failed:", e)

        self.cfg = cfg
        self._universe_cache: Dict[str, any] = {"ts": 0.0, "symbols": []}
        self._health_cache: Dict[str, float] = {}
        self._bad_cache: Dict[str, float] = {}

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        ohlcv = self.x.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("datetime").drop(columns=["timestamp"])
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.dropna()

    def fetch_funding_rate(self, symbol: str) -> Optional[float]:
        try:
            fr = self.x.fetch_funding_rate(symbol)
            return safe_float(fr.get("fundingRate"))
        except Exception:
            return None

    def get_balance_usdt(self) -> float:
        try:
            bal = self.x.fetch_balance(params={"type": "swap"})
            return float(bal["free"].get("USDT", 0.0))
        except Exception:
            return 0.0

    def create_demo_order(self, symbol: str, side: str, contract_amt: float, reduce_only: bool = False):
        """Execute a market order in contract units on OKX."""
        params = {"tdMode": self.margin_mode, "reduceOnly": reduce_only}
        if not self.pos_mode.startswith("net"):
            params["posSide"] = "long" if side.lower() == "buy" else "short"
        if self.leverage:
            params["lever"] = str(self.leverage)
        try:
            o = self.x.create_order(symbol, "market", side, contract_amt, params=params)
            oid = o.get("id") or o.get("orderId") or o.get("info", {}).get("ordId")
            if not oid:
                print("[WARN] create_order returned no id:", o)
                return None
            return o
        except Exception as e:
            print("[WARN] create_order failed:", e)
            return None

    def close_demo_position(self, symbol: str, orig_side: str, contract_amt: float):
        """Close an open position by sending the opposite order in contract units."""
        opp = "sell" if orig_side.lower() == "buy" else "buy"
        return self.create_demo_order(symbol, opp, contract_amt, reduce_only=True)

    def flatten_all(self):
        try:
            for s in self.x.symbols:
                try:
                    self.x.cancel_all_orders(s)
                except Exception:
                    continue
            try:
                positions = self.x.fetch_positions(params={"type": "swap"})
                for p in positions:
                    amt = abs(float(p.get("contracts") or p.get("positionAmt") or 0))
                    if amt <= 0:
                        continue
                    sym = p.get("symbol") or p.get("info", {}).get("instId")
                    side = p.get("side") or ("long" if float(p.get("contracts", 0)) > 0 else "short")
                    orig = "buy" if side == "long" else "sell"
                    self.close_demo_position(sym, orig, amt)
            except Exception:
                pass
        except Exception as e:
            print("[WARN] flatten failed:", e)

    def get_top_symbols(self, n: int = 50) -> List[str]:
        nowt = time.time()
        if (nowt - self._universe_cache["ts"]) < self.cfg.refresh_universe_minutes*60 and self._universe_cache["symbols"]:
            return self._universe_cache["symbols"]
        top: List[Tuple[str,float]] = []
        try:
            tickers = self.x.fetch_tickers()
            for m in self.x.markets.values():
                if not m.get("swap") or not m.get("contract"): continue
                if m.get("quote") != "USDT": continue
                if not m.get("active", True): continue
                sym = m["symbol"]
                t = tickers.get(sym, {})
                qv = t.get("quoteVolume")
                if qv is None:
                    qv = float(t.get("info", {}).get("quoteVolume", 0) or 0)
                top.append((sym, float(qv)))
            top.sort(key=lambda x: x[1], reverse=True)
            syms = [s for s,_ in top[:n]] or ["BTC/USDT:USDT","ETH/USDT:USDT"]
        except Exception:
            syms = ["BTC/USDT:USDT","ETH/USDT:USDT"]
        self._universe_cache = {"ts": nowt, "symbols": syms}
        return syms

    def filter_healthy(self, symbols: List[str]) -> List[str]:
        ok = []
        nowt = time.time()
        for s in symbols:
            last_bad = self._bad_cache.get(s, 0)
            if nowt - last_bad < self.cfg.health_refresh_minutes*60:
                continue
            last_ok = self._health_cache.get(s, 0)
            if nowt - last_ok < self.cfg.health_refresh_minutes*60:
                ok.append(s); continue
            try:
                _ = self.fetch_ohlcv(s, self.cfg.timeframe, limit=self.cfg.health_test_limit)
                if _.shape[0] > 10:
                    ok.append(s); self._health_cache[s] = nowt
                else:
                    self._bad_cache[s] = nowt
            except Exception:
                self._bad_cache[s] = nowt
            time.sleep(0.05)
        if not ok:
            ok = ["BTC/USDT:USDT","ETH/USDT:USDT"]
        return ok

# =========================
# Indicators
# =========================

def compute_indicators(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    d = df.copy()

    d["ema9"]  = ta.trend.EMAIndicator(d["close"], window=cfg.ema_fast).ema_indicator()
    d["ema21"] = ta.trend.EMAIndicator(d["close"], window=cfg.ema_slow).ema_indicator()

    d["rsi"] = ta.momentum.RSIIndicator(d["close"], window=cfg.rsi_len).rsi()

    bb = ta.volatility.BollingerBands(d["close"], window=cfg.bb_len, window_dev=cfg.bb_std)
    d["bb_mid"], d["bb_up"], d["bb_dn"] = bb.bollinger_mavg(), bb.bollinger_hband(), bb.bollinger_lband()

    atr = ta.volatility.AverageTrueRange(d["high"], d["low"], d["close"], window=cfg.atr_window)
    d["atr"] = atr.average_true_range()
    d["atr_pct"] = d["atr"] / d["close"]

    session = d.index.tz_convert("UTC").normalize()
    d["vwap_num"] = (d["close"] * d["volume"]).groupby(session).cumsum()
    d["vwap_den"] = d["volume"].groupby(session).cumsum().replace(0, np.nan)
    d["vwap"] = d["vwap_num"] / d["vwap_den"]

    d["vol_ma"] = d["volume"].rolling(cfg.vol_ma_len).mean()
    d["vol_spike"] = d["volume"] > (d["vol_ma"] * cfg.trend_vol_mult)
    d["bo_vol_spike"] = d["volume"] > (d["vol_ma"] * cfg.bo_vol_mult)

    d["recent_high"] = d["high"].rolling(cfg.box_len).max()
    d["recent_low"]  = d["low"].rolling(cfg.box_len).min()

    d["sr_high"] = d["high"].rolling(cfg.sr_lookback).max()
    d["sr_low"]  = d["low"].rolling(cfg.sr_lookback).min()

    d["ema9_slope"] = (d["ema9"] - d["ema9"].shift(3)) / d["close"]
    d["ema21_slope"] = (d["ema21"] - d["ema21"].shift(3)) / d["close"]
    d["bb_width"] = (d["bb_up"] - d["bb_dn"]) / d["bb_mid"]

    adx = ta.trend.ADXIndicator(d["high"], d["low"], d["close"], window=cfg.atr_window)
    d["adx"] = adx.adx()
    d["di_pos"] = adx.adx_pos()
    d["di_neg"] = adx.adx_neg()

    ema = ta.trend.EMAIndicator(d["close"], window=cfg.keltner_len).ema_indicator()
    rng = ta.volatility.AverageTrueRange(d["high"], d["low"], d["close"], window=cfg.keltner_len).average_true_range()
    d["kel_mid"] = ema
    d["kel_up"] = ema + cfg.keltner_mult * rng
    d["kel_dn"] = ema - cfg.keltner_mult * rng

    def rolling_pctl(x: pd.Series):
        last = x.iloc[-1]
        return float((x <= last).mean())
    d["atr_pct_pctl"] = d["atr_pct"].rolling(cfg.regime_lookback).apply(rolling_pctl, raw=False)

    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    return d

# =========================
# Regime
# =========================

@dataclass
class Regime:
    trend: str
    vol_bucket: str

def classify_regime(row: pd.Series, cfg: Config) -> Regime:
    if row["ema9"] > row["ema21"] and row["close"] > row["vwap"]:
        t = "up"
    elif row["ema9"] < row["ema21"] and row["close"] < row["vwap"]:
        t = "down"
    else:
        t = "neutral"
    p = row.get("atr_pct_pctl", np.nan)
    if np.isnan(p): v = "medium"
    elif p < cfg.low_vol_pct: v = "low"
    elif p > cfg.high_vol_pct: v = "high"
    else: v = "medium"
    return Regime(t, v)

# =========================
# Signal object
# =========================

@dataclass
class Signal:
    side: Optional[str]
    sl: float
    tp: float
    model: str
    reason: str
    confidence: float = 0.5

# ==== تعديل #2: إضافة دالة لحساب TP/SL بناءً على ATR ====
def make_tp_sl_atr(entry: float, side: str, atr: float, cfg: Config) -> Tuple[float, float]:
    """يحسب TP/SL بناءً على مضاعفات الـ ATR."""
    if side == "buy":
        tp = entry + (cfg.atr_tp_mult * atr)
        sl = entry - (cfg.atr_sl_mult * atr)
    else: # sell
        tp = entry - (cfg.atr_tp_mult * atr)
        sl = entry + (cfg.atr_sl_mult * atr)
    return tp, sl

def make_tp_sl(entry: float, side: str, cfg: Config) -> Tuple[float, float]:
    if side == "buy":
        tp = entry * (1 + cfg.fixed_tp_pct)
        sl = entry * (1 - cfg.fixed_sl_pct)
    else:
        tp = entry * (1 - cfg.fixed_tp_pct)
        sl = entry * (1 + cfg.fixed_sl_pct)
    return tp, sl

def get_tp_sl(entry: float, side: str, row: pd.Series, cfg: Config) -> Tuple[float, float]:
    """دالة موحدة لاختيار طريقة حساب TP/SL."""
    atr_val = safe_float(row.get("atr"))
    if cfg.use_atr_tp_sl and not np.isnan(atr_val) and atr_val > 0:
        return make_tp_sl_atr(entry, side, atr_val, cfg)
    else:
        return make_tp_sl(entry, side, cfg)

# SCALP strategy (BB + RSI)
# =========================

def sig_scalp(row: pd.Series, cfg: Config) -> Optional[Signal]:
    a = safe_float(row.get("atr", np.nan))
    if np.isnan(a) or a <= 0:
        return None
    price = float(row["close"])
    bb_lo = float(row["bb_dn"])
    bb_hi = float(row["bb_up"])
    r = float(row["rsi"])
    if price <= bb_lo and r <= 40:
        tp = price + 1.2 * a
        sl = price - 0.8 * a
        return Signal("buy", sl, tp, "SCALP", f"px<=BBlo & RSI={r:.1f}")
    if price >= bb_hi and r >= 60:
        tp = price - 1.2 * a
        sl = price + 0.8 * a
        return Signal("sell", sl, tp, "SCALP", f"px>=BBhi & RSI={r:.1f}")
    return None

def ctx_key(regime: Regime) -> str:
    return f"{regime.trend}|{regime.vol_bucket}"
# =========================
# Paper Engine
# =========================


@dataclass
class PaperTrade:
    id: str
    timestamp: str
    symbol: str
    timeframe: str
    side: str
    entry: float
    sl: float
    tp: float
    model: str
    qty: float
    status: str = "open"
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    result: Optional[str] = None
    pnl_usd: Optional[float] = None

class Paper:
    def __init__(self, cfg: Config, ref_equity: float):
        self.cfg = cfg
        self.ref_equity = ref_equity
        self.open: Dict[str, PaperTrade] = {}
        ensure_dir(cfg.signals_csv); ensure_dir(cfg.trades_csv); ensure_dir(cfg.ml_csv); ensure_dir(cfg.models_csv); ensure_dir(cfg.state_json)
        if not os.path.exists(cfg.signals_csv):
            pd.DataFrame(columns=[
                "time","symbol","tf","price","side","model","tp","sl",
                "ref_qty","ref_notional","rr","reason","conf",
                "trend","vol_bucket","ctx_key"
            ]).to_csv(cfg.signals_csv, index=False)
        if not os.path.exists(cfg.trades_csv):
            pd.DataFrame(columns=[
                "id","open_time","close_time","symbol","tf","side","entry","exit","result","model","pnl_usd","hold_sec","ctx_key"
            ]).to_csv(cfg.trades_csv, index=False)
        if not os.path.exists(cfg.ml_csv):
            pd.DataFrame(columns=[
                "trade_id","symbol","tf","side","model","open_time","close_time","result","pnl_usd",
                "price","ema9","ema21","ema9_slope","ema21_slope","rsi","bb_mid","bb_up","bb_dn","bb_width",
                "atr","atr_pct","atr_pct_pctl","vwap","vol","vol_ma","vol_spike","bo_vol_spike","recent_high",
                "recent_low","sr_high","sr_low","regime_trend","regime_vol","ctx_key","adx","di_pos","di_neg",
                "kel_mid","kel_up","kel_dn"
            ]).to_csv(cfg.ml_csv, index=False)
        if not os.path.exists(cfg.models_csv):
            pd.DataFrame(columns=["time","symbol","tf","model","ctx_key","decision_score","accepted","weight","conf","notes"]).to_csv(cfg.models_csv, index=False)

    def _gen_id(self) -> str: return f"T{int(time.time()*1000)}"

    def open_virtual(self, symbol: str, price: float, sig: Signal, qty: float, cfg: Config) -> PaperTrade:
        t = PaperTrade(
            id=self._gen_id(), timestamp=fmt_ts(), symbol=symbol, timeframe=cfg.timeframe,
            side=sig.side, entry=price, sl=float(sig.sl), tp=float(sig.tp), model=sig.model, qty=qty
        )
        self.open[t.id] = t
        return t

    def _hit(self, side: str, high: float, low: float, level: float, is_tp: bool) -> bool:
        if side == "buy":
            return high >= level if is_tp else low <= level
        else:
            return low <= level if is_tp else high >= level

    def update_with_candle(self, symbol: str, high: float, low: float):
        to_close = []
        for tid, t in list(self.open.items()):
            if t.status != "open" or t.symbol != symbol: continue
            hit_tp = self._hit(t.side, high, low, t.tp, True)
            hit_sl = self._hit(t.side, high, low, t.sl, False)
            if hit_tp and hit_sl:
                res, px = "sl", t.sl
            elif hit_tp:
                res, px = "tp", t.tp
            elif hit_sl:
                res, px = "sl", t.sl
            else:
                continue
            t.status = "closed"; t.result = res; t.exit_price = float(px); t.exit_time = fmt_ts()
            qty = t.qty
            pnl = (t.exit_price - t.entry) * qty * (1 if t.side=="buy" else -1)
            t.pnl_usd = round(pnl,4)
            to_close.append(tid)
        closed = [self.open[k] for k in to_close]
        for k in to_close: self.open.pop(k, None)
        return closed

    def persist_closed(self, closed: List[PaperTrade], cfg: Config, ctx: str):
        if not closed: return
        rows=[]
        for t in closed:
            hold = int((pd.to_datetime(t.exit_time) - pd.to_datetime(t.timestamp)).total_seconds())
            rows.append({
                "id": t.id, "open_time": t.timestamp, "close_time": t.exit_time, "symbol": t.symbol, "tf": t.timeframe,
                "side": t.side, "entry": round(t.entry,6), "exit": round(t.exit_price,6),
                "result": t.result, "model": t.model, "pnl_usd": t.pnl_usd, "hold_sec": hold, "ctx_key": ctx
            })
        pd.DataFrame(rows).to_csv(cfg.trades_csv, mode="a", header=False, index=False)

    def ml_snapshot(self, trade_id:str, symbol:str, row:pd.Series, regime:Regime):
        feat = {
            "trade_id": trade_id, "symbol": symbol, "tf": self.cfg.timeframe, "side": "", "model": "",
            "open_time": fmt_ts(), "close_time":"", "result":"", "pnl_usd":"",
            "price": float(row["close"]),
            "ema9": float(row.get("ema9", np.nan) or np.nan),
            "ema21": float(row.get("ema21", np.nan) or np.nan),
            "ema9_slope": float(row.get("ema9_slope", np.nan) or np.nan),
            "ema21_slope": float(row.get("ema21_slope", np.nan) or np.nan),
            "rsi": float(row.get("rsi", np.nan) or np.nan),
            "bb_mid": float(row.get("bb_mid", np.nan) or np.nan),
            "bb_up": float(row.get("bb_up", np.nan) or np.nan),
            "bb_dn": float(row.get("bb_dn", np.nan) or np.nan),
            "bb_width": float(row.get("bb_width", np.nan) or np.nan),
            "atr": float(row.get("atr", np.nan) or np.nan),
            "atr_pct": float(row.get("atr_pct", np.nan) or np.nan),
            "atr_pct_pctl": float(row.get("atr_pct_pctl", np.nan) or np.nan),
            "vwap": float(row.get("vwap", np.nan) or np.nan),
            "vol": float(row.get("volume", np.nan) or np.nan),
            "vol_ma": float(row.get("vol_ma", np.nan) or np.nan),
            "vol_spike": bool(row.get("vol_spike", False)),
            "bo_vol_spike": bool(row.get("bo_vol_spike", False)),
            "recent_high": float(row.get("recent_high", np.nan) or np.nan),
            "recent_low": float(row.get("recent_low", np.nan) or np.nan),
            "sr_high": float(row.get("sr_high", np.nan) or np.nan),
            "sr_low": float(row.get("sr_low", np.nan) or np.nan),
            "regime_trend": regime.trend, "regime_vol": regime.vol_bucket, "ctx_key": ctx_key(regime),
            "adx": float(row.get("adx", np.nan) or np.nan),
            "di_pos": float(row.get("di_pos", np.nan) or np.nan),
            "di_neg": float(row.get("di_neg", np.nan) or np.nan),
            "kel_mid": float(row.get("kel_mid", np.nan) or np.nan),
            "kel_up": float(row.get("kel_up", np.nan) or np.nan),
            "kel_dn": float(row.get("kel_dn", np.nan) or np.nan),
        }
        pd.DataFrame([feat]).to_csv(self.cfg.ml_csv, mode="a", header=False, index=False)

    def log_signal(self, symbol: str, row: pd.Series, sig: Signal, qty_ref: float,
                   notional_ref: float, rr: Optional[float], cfg: Config, regime: Regime):
        pd.DataFrame([{
            "time": fmt_ts(), "symbol": symbol, "tf": cfg.timeframe, "price": float(row["close"]),
            "side": sig.side, "model": sig.model, "tp": round(sig.tp,6), "sl": round(sig.sl,6),
            "ref_qty": round(qty_ref,8), "ref_notional": round(notional_ref,2),
            "rr": rr if rr is not None else "", "reason": sig.reason, "conf": round(sig.confidence,2),
            "trend": regime.trend, "vol_bucket": regime.vol_bucket, "ctx_key": ctx_key(regime)
        }]).to_csv(cfg.signals_csv, mode="a", header=False, index=False)

    def log_model_vote(self, cfg: Config, symbol: str, regime: Regime, model: str, score: float, accepted: bool, weight: float, conf: float, notes: str = ""):
        pd.DataFrame([{
            "time": fmt_ts(), "symbol": symbol, "tf": cfg.timeframe, "model": model, "ctx_key": ctx_key(regime),
            "decision_score": round(score,4), "accepted": int(accepted), "weight": round(weight,4), "conf": round(conf,4),
            "notes": notes[:120]
        }]).to_csv(cfg.models_csv, mode="a", header=False, index=False)

# =========================
# News (اختياري)
# =========================

class NewsGuard:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cp_token = os.getenv("CRYPTOPANIC_TOKEN")
        self.newsapi_key = os.getenv("NEWSAPI_KEY")
        self.cache: Dict[str, float] = {}
    def too_hot(self, asset: str) -> bool:
        if not self.cfg.news_enabled: return False
        hot = False
        try:
            if self._cryptopanic(asset): hot = True
        except Exception: pass
        try:
            if self._newsapi(asset): hot = True
        except Exception: pass
        if hot: self.cache[asset] = time.time()
        else:
            if asset in self.cache and (time.time() - self.cache[asset]) < 600:
                hot = True
        return hot
    def _cryptopanic(self, asset: str) -> bool:
        if not self.cp_token: return False
        url = "https://cryptopanic.com/api/v1/posts/"
        params = {"auth_token": self.cp_token, "currencies": asset.lower(), "kind": "news", "public": "true"}
        r = requests.get(url, params=params, timeout=8)
        if r.status_code != 200: return False
        data = r.json().get("results", [])
        since = int((now_utc() - dt.timedelta(minutes=self.cfg.news_lookback_minutes)).timestamp())
        for it in data:
            pub = it.get("published_at") or it.get("created_at") or ""
            try: ts = dt.datetime.fromisoformat(pub.replace("Z","+00:00")).timestamp()
            except Exception: ts = 0
            if ts >= since:
                title = (it.get("title") or "").lower()
                important = it.get("importance") in ("high","very_high")
                kw = any(k.lower() in title for k in self.cfg.news_keywords)
                if important or kw: return True
        return False
    def _newsapi(self, asset: str) -> bool:
        if not self.newsapi_key: return False
        q = "bitcoin" if asset.upper()=="BTC" else ("ethereum" if asset.upper()=="ETH" else asset)
        url = "https://newsapi.org/v2/everything"
        since = (now_utc() - dt.timedelta(minutes=self.cfg.news_lookback_minutes)).isoformat()
        params = {"q": q, "from": since, "language": "en", "sortBy": "publishedAt",
                  "apiKey": self.newsapi_key, "pageSize": 20}
        r = requests.get(url, params=params, timeout=8)
        if r.status_code != 200: return False
        arts = r.json().get("articles", [])
        for a in arts:
            title = (a.get("title") or "").lower()
            desc  = (a.get("description") or "").lower()
            if any(k.lower() in title or k.lower() in desc for k in self.cfg.news_keywords):
                return True
        return False

# =========================
# Quiet & sizing
# =========================

def in_quiet_window(cfg: Config) -> bool:
    if not cfg.quiet_windows_utc: return False
    nowt = now_utc().time().replace(second=0, microsecond=0)
    for hhmm in cfg.quiet_windows_utc:
        try:
            t = dt.datetime.strptime(hhmm, "%H:%M").time()
        except Exception:
            continue
        start = (dt.datetime.combine(dt.date.today(), t, tzinfo=dt.timezone.utc)
                 - dt.timedelta(minutes=cfg.event_quiet_minutes)).time()
        end   = (dt.datetime.combine(dt.date.today(), t, tzinfo=dt.timezone.utc)
                 + dt.timedelta(minutes=cfg.event_quiet_minutes)).time()
        if start <= nowt <= end:
            return True
    return False

# =========================
# Bot
# =========================

class Bot:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.ex = FuturesExchange(cfg)
        self.notifier = Notifier(cfg)
        self.ref_equity = self.ex.get_balance_usdt() or 10000.0
        self.paper = Paper(cfg, self.ref_equity)
        base_universe = self.ex.get_top_symbols(cfg.top_n_symbols)
        self.symbols: List[str] = self.ex.filter_healthy(base_universe)
        self.news = NewsGuard(cfg)
        self.last_key: Dict[str, Optional[str]] = {}
        self.last_time: Dict[str, Optional[dt.datetime]] = {}
        self.last_alert_ts: float = 0.0
        self.closed_trades: List[PaperTrade] = []
        self.last_hourly_report = now_utc()
        self.last_daily_report_date = now_utc().date()

    def _maybe_hourly_report(self):
        now = now_utc()
        if now - self.last_hourly_report >= dt.timedelta(hours=1):
            since = self.last_hourly_report
            trades = [t for t in self.closed_trades if pd.to_datetime(t.exit_time) >= since]
            profit = sum((t.pnl_usd or 0) for t in trades if (t.pnl_usd or 0) > 0)
            loss = sum((t.pnl_usd or 0) for t in trades if (t.pnl_usd or 0) < 0)
            net = profit + loss
            msg = (f"⏱ Hourly Report\n"
                   f"Trades: {len(trades)}\n"
                   f"Profit: {profit:.2f} USDT\n"
                   f"Loss: {loss:.2f} USDT\n"
                   f"Net: {net:.2f} USDT")
            self.notifier.send(msg)
            self.last_hourly_report = now
            cutoff = now - dt.timedelta(days=1)
            self.closed_trades = [t for t in self.closed_trades if pd.to_datetime(t.exit_time) >= cutoff]

    def _send_daily_report(self, date: dt.date):
        if not os.path.exists(self.cfg.trades_csv):
            self.notifier.send(f"📅 Daily Report {date}: No trades")
            return
        df = pd.read_csv(self.cfg.trades_csv)
        if df.empty:
            msg = f"📅 Daily Report {date}\nNo trades"
        else:
            df['close_time'] = pd.to_datetime(df['close_time'])
            day_df = df[df['close_time'].dt.date == date]
            if day_df.empty:
                msg = f"📅 Daily Report {date}\nNo trades"
            else:
                total = day_df['pnl_usd'].sum()
                lines = [f"📅 Daily Report {date}", f"Total PnL: {total:.2f} USDT"]
                for sym, val in day_df.groupby('symbol')['pnl_usd'].sum().items():
                    lines.append(f"{sym}: {val:+.2f} USDT")
                msg = "\n".join(lines)
        self.notifier.send(msg)

    def _maybe_daily_report(self):
        today = now_utc().date()
        if today != self.last_daily_report_date:
            self._send_daily_report(self.last_daily_report_date)
            self.last_daily_report_date = today

    # ==========================================

    def can_alert_now(self) -> bool:
        return (time.time() - self.last_alert_ts) >= self.cfg.min_seconds_between_alerts_global

    def _committee(self, symbol: str, row: pd.Series, regime: Regime) -> Optional[Signal]:
        return sig_scalp(row, self.cfg)

    def run(self):
        self.notifier.send(f"[START] Evolving Scalper | TOP {self.cfg.top_n_symbols} | TF {self.cfg.timeframe} | RefEq={self.ref_equity:.2f} USDT")
        while True:
            try:
                self.loop_once()
                time.sleep(2)
            except KeyboardInterrupt:
                self.notifier.send("[EXIT] Stopping…"); break
            except Exception as e:
                self.notifier.send(f"[ERROR main] {e}"); time.sleep(3)

    def loop_once(self):
        # رولات اليوم
        self._maybe_daily_report()

        base_universe = self.ex.get_top_symbols(self.cfg.top_n_symbols)
        self.symbols = self.ex.filter_healthy(base_universe)

        # تحديث إغلاقات الصفقات (حتى أثناء الوقف/التهدئة)
        for symbol in self.symbols:
            try:
                df = self.ex.fetch_ohlcv(symbol, self.cfg.timeframe, limit=self.cfg.lookback)
                d  = compute_indicators(df, self.cfg)
                if len(d) < 2: continue

                last = d.iloc[-1]
                closed = self.paper.update_with_candle(symbol, float(last["high"]), float(last["low"]))
                if closed:
                    for t in closed:
                        mkt = self.ex.x.market(t.symbol)
                        contract_size = float(mkt.get("contractSize") or 1)
                        contract_qty = t.qty / contract_size
                        contract_qty = float(self.ex.x.amount_to_precision(t.symbol, contract_qty))
                        self.ex.close_demo_position(t.symbol, t.side, contract_qty)
                    reg_row = d.iloc[-2] if len(d)>1 else last
                    regime = classify_regime(reg_row, self.cfg)
                    ctx = ctx_key(regime)
                    self.paper.persist_closed(closed, self.cfg, ctx)

                    # تحديث المخاطر (streak + daily pnl)
                    pnl_sum = 0.0
                    sl_count = 0
                    for t in closed:
                        pnl_sum += float(t.pnl_usd or 0.0)
                        if t.result == "sl": sl_count += 1
                        emoji = "✅" if t.result=="tp" else "❌"
                        hold_s = int((pd.to_datetime(t.exit_time)-pd.to_datetime(t.timestamp)).total_seconds())
                        self.notifier.send(
                            f"📤 Trade Closed {emoji}\n"
                            f"• Pair: {t.symbol} | TF: {t.timeframe}\n"
                            f"• Side: {t.side.upper()} | Model: {t.model}\n"
                            f"• Entry: {t.entry:.4f} → Exit: {t.exit_price:.4f}\n"
                            f"• PnL: {t.pnl_usd:+.2f} USDT | Hold: {hold_s}s",
                        )
                        self.closed_trades.append(t)
            except Exception:
                continue

        self._maybe_hourly_report()

        # لو في صفقات مفتوحة — نكتفي بتتبع الإغلاق فقط
        if len(self.paper.open) >= self.cfg.max_open_trades:
            return

        # هدوء أحداث أو ثروتل
        if not self.can_alert_now(): return

        # البحث عن إشارة جديدة
        for symbol in self.symbols:
            try:

                df = self.ex.fetch_ohlcv(symbol, self.cfg.timeframe, limit=self.cfg.lookback)
                d  = compute_indicators(df, self.cfg)
                if len(d) < 3: continue

                row = d.iloc[-2]
                price = float(row["close"])
                regime = classify_regime(row, self.cfg)

                sig = self._committee(symbol, row, regime)
                if not sig: continue

                # مانع تكرار نفس الإشارة
                key = f"{symbol}:{self.cfg.timeframe}:{sig.model}:{sig.side}"
                if self.last_key.get(symbol) == key and self.last_time.get(symbol):
                    if (now_utc() - self.last_time[symbol]).total_seconds()/60.0 < self.cfg.min_minutes_between_same_signal:
                        continue

                base_qty = (TRADE_MARGIN_USD * LEVERAGE) / price
                mkt = self.ex.x.market(symbol)
                contract_size = float(mkt.get("contractSize") or 1)
                contract_qty = base_qty / contract_size
                contract_qty = float(self.ex.x.amount_to_precision(symbol, contract_qty))
                base_qty = contract_qty * contract_size
                notional_ref = base_qty * price
                bal = self.ex.get_balance_usdt()
                req_margin = notional_ref / LEVERAGE
                if bal < req_margin:
                    print(f"[WARN] skipping {symbol}: need {req_margin:.2f} USDT but only {bal:.2f} available")
                    continue
                risk = abs(price - sig.sl); reward = abs(sig.tp - price)
                rr = round(reward / risk, 2) if risk > 0 else None

                order = self.ex.create_demo_order(symbol, sig.side, contract_qty)
                status_line = "🚀 Executed on OKX Demo" if order else "⚠️ Execution failed on OKX Demo"
                msg = (
                    f"📢 [SCALP] New Signal\n\n"
                    f"📍 Pair: {symbol}\n"
                    f"🕒 TF: {self.cfg.timeframe} | Ctx: trend={regime.trend}, vol={regime.vol_bucket}\n"
                    f"📈 Side: {sig.side.upper()} | Conf: {sig.confidence:.2f}\n\n"
                    f"💰 Entry: {price:.4f}\n"
                    f"🎯 TP: {sig.tp:.4f} ({'+' if sig.tp > price else ''}{pct((sig.tp-price)/price)})\n"
                    f"🛡 SL: {sig.sl:.4f} ({'-' if sig.sl < price else '+'}{pct(abs(sig.sl-price)/price)})\n"
                    f"📏 R:R = {rr if rr is not None else 'n/a'}\n\n"
                    f"🧠 Why: {sig.reason}\n"
                    f"📦 SizeRef: ~{base_qty:.6f} ({notional_ref:.2f} USDT)\n"
                    f"{status_line}"
                )
                self.notifier.send(msg)
                self.last_alert_ts = time.time()

                self.paper.log_signal(symbol, row, sig, base_qty, notional_ref, rr, self.cfg, regime)
                t = self.paper.open_virtual(symbol, price, sig, base_qty, self.cfg)
                self.paper.ml_snapshot(t.id, symbol, row, regime)

                self.last_key[symbol] = key
                self.last_time[symbol] = now_utc()
                break

            except Exception:
                continue

# =========================
# CLI
# =========================

def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Evolving Committee Scalper (Alerts Only) — No OpenAI")
    p.add_argument("--timeframe", default="15m")
    p.add_argument("--quiet", nargs="*", default=None, help="UTC HH:MM times to avoid (e.g., 12:30 18:00)")
    p.add_argument("--top", type=int, default=None, help="Top N USDT perpetuals to scan (override config)")
    args = p.parse_args()
    cfg = Config()
    cfg.timeframe = args.timeframe
    # اسمح بتجاوز الإعدادات من سطر الأوامر
    if args.top is not None: cfg.top_n_symbols = int(args.top)
    if args.quiet is not None: cfg.quiet_windows_utc = tuple(args.quiet)
    ensure_dir(cfg.logs_dir)
    ensure_dir(cfg.signals_csv); ensure_dir(cfg.trades_csv); ensure_dir(cfg.ml_csv); ensure_dir(cfg.models_csv)
    return cfg

def main():
    cfg = parse_args()
    # إخفاء توكن التيليجرام في الطباعة
    _cfg = asdict(cfg)
    if _cfg.get("telegram_token"):
        _tok = _cfg["telegram_token"]
        _cfg["telegram_token"] = (_tok[:4] + "…" + _tok[-4:]) if len(_tok) > 8 else "****"
    print("Config:\n", json.dumps(_cfg, indent=2, default=str))
    bot = Bot(cfg)
    bot.run()

if __name__ == "__main__":
    main()
