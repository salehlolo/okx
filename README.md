# SCALP Strategy

This repository provides a minimal implementation of a scalping strategy that combines Bollinger Bands with RSI. The strategy logic is implemented in `scalp_strategy.py`.

The `strat` function accepts a pandas DataFrame containing `open`, `high`, `low`, `close`, and `volume` columns and returns trade signals in the form `(side, stop_loss, take_profit, reason)`.
