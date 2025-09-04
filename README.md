# OKX Demo Scalper

This repository contains a minimal bot that trades OKX USDT perpetual swaps on the demo environment using a single Bollinger Bands + RSI strategy.

## Key Features

- Fixed position size: every trade uses **50 USDT** margin with **10× leverage** (≈500 USDT notional).
- Maximum of **2** simultaneous open trades.
- Strategy runs on **30‑minute** candles.
- Targets and stops are derived from ATR and calculated at run‑time.

## Files

- `bot.py` – main executable script.
- `scalp_strategy.py` – strategy implementation.

## Running

```bash
python bot.py
```

Set `OKX_API_KEY`, `OKX_API_SECRET` and `OKX_API_PASSWORD` along with optional `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` for Telegram alerts.
