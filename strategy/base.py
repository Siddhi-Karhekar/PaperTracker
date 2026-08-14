"""
strategy/base.py

Base interface for all trading strategies.

A Strategy takes a clean OHLCV DataFrame (the shape produced by
data.loader.fetch_ohlcv -- columns: date, open, high, low, close, volume)
and returns a *target position* signal series aligned to the same rows:

    +1  -> be long at this bar
     0  -> be flat (no position) at this bar
    -1  -> be short at this bar (long-only strategies simply never emit this)

This is a target-position signal, not a one-shot trade instruction --  it
says "where the strategy wants to be," not "buy now." The execution
simulator (engine/execution.py, Phase 3) is what diffs consecutive signals
into actual buy/sell orders and applies costs/slippage. Keeping that
translation in one place, downstream of every strategy, is what lets any
new Strategy subclass plug into the rest of the system unchanged.

Signal timing: generate_signals may only use information available up to
and including each row's own close -- never a later row. Whether a signal
computed at bar t's close gets executed at bar t's close or bar t+1's open
is an execution-simulator concern (Phase 3 fixes this to next-open to avoid
look-ahead bias), not a strategy concern. A strategy must not shift data
backward in time to "look ahead."
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

REQUIRED_COLUMNS = {"date", "open", "high", "low", "close", "volume"}
VALID_SIGNAL_VALUES = {-1, 0, 1}


class Strategy(ABC):
    """
    Subclass this and implement generate_signals(). Give every subclass a
    descriptive `name` so strategies are identifiable in logs, reports, and
    the dashboard (Phase 8) without extra bookkeeping.
    """

    name: str = "unnamed_strategy"

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        df: OHLCV DataFrame with columns in REQUIRED_COLUMNS, sorted
            ascending by date, one row per trading day.

        Returns a pd.Series of ints in {-1, 0, 1}, same length and index as
        df, representing the target position at each bar. Early bars where
        the strategy doesn't have enough history yet (e.g. before a moving
        average window fills) should be 0 (flat), not NaN -- downstream
        code assumes a fully-populated int signal.
        """
        raise NotImplementedError

    def run(self, df: pd.DataFrame) -> pd.Series:
        """
        Public entry point: validates the input, calls generate_signals,
        validates the output, and returns the signal series. Strategy
        subclasses should generally not need to override this.
        """
        _validate_input(df)
        signals = self.generate_signals(df)
        return _validate_signals(signals, expected_len=len(df), expected_index=df.index)


def _validate_input(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Strategy input is missing required columns: {missing}")
    if not df["date"].is_monotonic_increasing:
        raise ValueError("Strategy input must be sorted ascending by date")


def _validate_signals(signals: pd.Series, expected_len: int, expected_index: pd.Index) -> pd.Series:
    if not isinstance(signals, pd.Series):
        raise TypeError(f"generate_signals must return a pd.Series, got {type(signals)}")
    if len(signals) != expected_len:
        raise ValueError(f"Signal length {len(signals)} does not match input length {expected_len}")
    if not signals.index.equals(expected_index):
        raise ValueError("Signal index must match the input DataFrame's index")
    if signals.isna().any():
        raise ValueError("Signal series must not contain NaN -- use 0 (flat) during warm-up periods")
    bad_values = set(signals.unique()) - VALID_SIGNAL_VALUES
    if bad_values:
        raise ValueError(f"Signal series contains values outside {VALID_SIGNAL_VALUES}: {bad_values}")
    return signals.astype(int)
