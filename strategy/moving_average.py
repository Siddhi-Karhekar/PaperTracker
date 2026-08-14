"""
strategy/moving_average.py

The project's first strategy: a simple moving average crossover.

Signal (target position) at each bar:
    +1  fast SMA is above slow SMA  (uptrend regime -> long)
    -1  fast SMA is below slow SMA  (downtrend regime -> short)
     0  either the warm-up window hasn't filled yet, or fast == slow exactly

This is a *level-based* signal (the strategy's stance at every bar), not an
edge-triggered "buy only on the crossing bar" signal -- the position is
meant to be held continuously through a regime, which is what the
execution simulator (Phase 3) expects to diff into entry/exit trades.

Long-only variant: pass long_only=True to floor negative signals at 0
instead of going short, which is the more realistic default for a retail
NSE equity strategy (shorting cash equities has its own borrow/margin
mechanics this project isn't modeling).
"""

from __future__ import annotations

import pandas as pd

from strategy.base import Strategy


class MovingAverageCrossover(Strategy):
    def __init__(self, fast_window: int = 50, slow_window: int = 200, long_only: bool = True):
        if fast_window <= 0 or slow_window <= 0:
            raise ValueError("fast_window and slow_window must be positive")
        if fast_window >= slow_window:
            raise ValueError(f"fast_window ({fast_window}) must be < slow_window ({slow_window})")

        self.fast_window = fast_window
        self.slow_window = slow_window
        self.long_only = long_only
        self.name = f"ma_crossover_{fast_window}_{slow_window}"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]

        fast_sma = close.rolling(window=self.fast_window, min_periods=self.fast_window).mean()
        slow_sma = close.rolling(window=self.slow_window, min_periods=self.slow_window).mean()

        signal = pd.Series(0, index=df.index, dtype=int)
        has_history = slow_sma.notna()  # implies fast_sma is also non-NaN, since fast_window < slow_window

        signal.loc[has_history & (fast_sma > slow_sma)] = 1
        signal.loc[has_history & (fast_sma < slow_sma)] = -1
        # fast_sma == slow_sma (rare with real prices) stays 0, and rows
        # before slow_window bars have accumulated stay 0 -- both are
        # "no confirmed regime yet" and read the same as flat.

        if self.long_only:
            signal = signal.clip(lower=0)

        return signal
