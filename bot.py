#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal OKX demo scalper using a single Bollinger+RSI strategy."""
import os, time, json, argparse, datetime as dt
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import ccxt
import requests

from scalp_strategy import strat

TRADE_MARGIN_USD = 50.0
LEVERAGE = 10
MAX_OPEN_TRADES = 2


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def fmt_ts(ts: Optional[dt.datetime] = None) -> str:
    return (ts or now_utc()).isoformat().replace("+00:00", "Z")


def ensure_dir(path: str):
    d = path if os.path.splitext(path)[1] == "" else os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


@dataclass
class Config:
    timeframe: str = "30m"
    lookback: int = 200
    top_n_symbols: int = 10
    telegram_enabled: bool = True
    telegram_token: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat_id: Optional[str] = os.getenv("TELEGRAM_CHAT_ID")
    logs_dir: str = "./logs"
    signals_csv: str = "./logs/signals_log.csv"
    trades_csv: str = "./logs/trades_log.csv"


class Notifier:
    def __init__(self, cfg: Config):
        self.enabled = bool(cfg.telegram_enabled and cfg.telegram_token and cfg.telegram_chat_id)
        self.base = f"https://api.telegram.org/bot{cfg.telegram_token}" if self.enabled else None
        self.chat_id = cfg.telegram_chat_id

    def send(self, text: str):
        print(text)
        if not self.enabled:
            return
        try:
            requests.post(f"{self.base}/sendMessage", json={"chat_id": self.chat_id, "text": text}, timeout=10)
        except Exception as e:
            print("[WARN] Telegram exception:", e)


class FuturesExchange:
    def __init__(self):
        key = os.getenv("OKX_API_KEY")
        secret = os.getenv("OKX_API_SECRET")
        password = os.getenv("OKX_API_PASSWORD") or os.getenv("OKX_API_PASSPHRASE")
        self.x = ccxt.okx({
            "apiKey": key,
            "secret": secret,
            "password": password,
            "options": {"defaultType": "swap", "demo": True},
            "headers": {"x-simulated-trading": "1"},
        })
        try:
            self.x.set_sandbox_mode(True)
        except Exception:
            pass
        self.x.load_markets()

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        ohlcv = self.x.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df.set_index("datetime").drop(columns=["timestamp"])

    def fetch_ticker_price(self, symbol: str) -> Optional[float]:
        try:
            t = self.x.fetch_ticker(symbol)
            return float(t.get("last") or t.get("close"))
        except Exception:
            return None

    def get_balance_usdt(self) -> float:
        try:
            bal = self.x.fetch_balance(params={"type": "swap"})
            free = bal.get("free", {}).get("USDT")
            if free is None:
                details = bal.get("info", {}).get("data", [{}])[0].get("details", [])
                for d in details:
                    if d.get("ccy") == "USDT":
                        free = d.get("availBal")
                        break
            return float(free or 0.0)
        except Exception:
            return 0.0

    def create_demo_order(self, symbol: str, side: str, amount: float):
        params = {"tdMode": "cross", "lever": str(LEVERAGE)}
        try:
            return self.x.create_order(symbol, "market", side, amount, params=params)
        except Exception as e:
            print("[WARN] create_order failed:", e)
            return None

    def close_demo_position(self, symbol: str, side: str, amount: float):
        opp = "sell" if side.lower() == "buy" else "buy"
        return self.create_demo_order(symbol, opp, amount)

    def get_top_symbols(self, n: int) -> List[str]:
        try:
            tickers = self.x.fetch_tickers()
            pairs = [(s, t.get("quoteVolume", 0)) for s, t in tickers.items() if s.endswith(":USDT")]
            pairs.sort(key=lambda x: x[1], reverse=True)
            return [s for s,_ in pairs[:n]]
        except Exception:
            return ["BTC/USDT:USDT", "ETH/USDT:USDT"]


@dataclass
class PaperTrade:
    id: str
    symbol: str
    side: str
    entry: float
    sl: float
    tp: float
    qty: float
    open_time: str
    status: str = "open"


class Paper:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.open: Dict[str, PaperTrade] = {}
        ensure_dir(cfg.signals_csv)
        ensure_dir(cfg.trades_csv)
        if not os.path.exists(cfg.signals_csv):
            pd.DataFrame(columns=["time","symbol","tf","price","side","tp","sl","qty","reason"]).to_csv(cfg.signals_csv, index=False)
        if not os.path.exists(cfg.trades_csv):
            pd.DataFrame(columns=["id","open_time","close_time","symbol","side","entry","exit","result","pnl_usd"]).to_csv(cfg.trades_csv, index=False)

    def open_trade(self, symbol: str, price: float, side: str, sl: float, tp: float, qty: float, reason: str) -> PaperTrade:
        t = PaperTrade(id=f"T{int(time.time()*1000)}", symbol=symbol, side=side, entry=price, sl=sl, tp=tp, qty=qty, open_time=fmt_ts())
        self.open[t.id] = t
        pd.DataFrame([{ "time": t.open_time, "symbol": symbol, "tf": cfg.timeframe, "price": price, "side": side, "tp": tp, "sl": sl, "qty": qty, "reason": reason }]).to_csv(self.cfg.signals_csv, mode="a", header=False, index=False)
        return t

    def update(self, symbol: str, high: float, low: float) -> List[PaperTrade]:
        closed = []
        for tid, t in list(self.open.items()):
            if t.symbol != symbol or t.status != "open":
                continue
            hit_tp = high >= t.tp if t.side == "buy" else low <= t.tp
            hit_sl = low <= t.sl if t.side == "buy" else high >= t.sl
            if hit_tp or hit_sl:
                t.status = "closed"
                px = t.tp if hit_tp else t.sl
                pnl = (px - t.entry) * t.qty * (1 if t.side == "buy" else -1)
                pd.DataFrame([{ "id": t.id, "open_time": t.open_time, "close_time": fmt_ts(), "symbol": symbol, "side": t.side, "entry": t.entry, "exit": px, "result": "tp" if hit_tp else "sl", "pnl_usd": pnl }]).to_csv(self.cfg.trades_csv, mode="a", header=False, index=False)
                closed.append(t)
                del self.open[tid]
        return closed


class Bot:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.notifier = Notifier(cfg)
        self.ex = FuturesExchange()
        self.paper = Paper(cfg)

    def run(self):
        self.notifier.send(f"[START] Scalper | TF {self.cfg.timeframe}")
        while True:
            try:
                self.loop_once()
                time.sleep(2)
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.notifier.send(f"[ERROR] {e}")
                time.sleep(3)

    def loop_once(self):
        symbols = self.ex.get_top_symbols(self.cfg.top_n_symbols)
        for symbol in symbols:
            df = self.ex.fetch_ohlcv(symbol, self.cfg.timeframe, self.cfg.lookback)
            if df.empty:
                continue
            closed = self.paper.update(symbol, float(df.high.iloc[-1]), float(df.low.iloc[-1]))
            for t in closed:
                self.ex.close_demo_position(symbol, t.side, t.qty)
                self.notifier.send(f"📤 Trade Closed {t.symbol} {t.side} pnl={t.tp - t.entry:.4f}")
            if len(self.paper.open) >= MAX_OPEN_TRADES:
                continue
            sig = strat(df)
            if not sig:
                continue
            side, sl, tp, reason = sig
            price = self.ex.fetch_ticker_price(symbol)
            if price is None:
                continue
            notional = TRADE_MARGIN_USD * LEVERAGE
            qty = notional / price
            bal = self.ex.get_balance_usdt()
            if bal < TRADE_MARGIN_USD:
                print(f"[WARN] skipping {symbol}: need {TRADE_MARGIN_USD} USDT but only {bal:.2f}")
                continue
            order = self.ex.create_demo_order(symbol, side, qty)
            status = "🚀 Executed" if order else "⚠️ Order Failed"
            self.notifier.send(
                f"📢 Signal {symbol}\nSide: {side}\nEntry: {price:.4f}\nTP: {tp:.4f}\nSL: {sl:.4f}\n{status}\n{reason}"
            )
            if order:
                self.paper.open_trade(symbol, price, side, sl, tp, qty, reason)
                break


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--timeframe", default="30m")
    args = p.parse_args()
    cfg = Config()
    cfg.timeframe = args.timeframe
    ensure_dir(cfg.logs_dir)
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    bot = Bot(cfg)
    bot.run()
