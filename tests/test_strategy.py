"""
Unit tests for strategy/base.py and strategy/moving_average.py.

The crossover test uses a tiny hand-computed dataset (fast_window=2,
slow_window=3) so every expected signal value below was verified by hand,
not just by trusting pandas.rolling -- see the comment block above
_make_known_df for the arithmetic.
"""

import pandas as pd
import pytest

from strategy.base import Strategy
from strategy.moving_average import MovingAverageCrossover


def _make_known_df() -> pd.DataFrame:
    """
    closes:            [10, 20, 10,  5,  5,  5, 30, 30, 30]
    index:                0   1   2   3   4   5   6   7   8

    rolling mean w=2 (fast):   NaN, 15, 15, 7.5, 5, 5, 17.5, 30, 30
    rolling mean w=3 (slow):   NaN, NaN, 13.333, 11.667, 6.667, 5, 13.333, 21.667, 30

    fast vs slow (only defined once slow fills, idx >= 2):
      idx2: 15    > 13.333  -> +1
      idx3: 7.5   < 11.667  -> -1
      idx4: 5     < 6.667   -> -1
      idx5: 5     == 5      ->  0  (tie)
      idx6: 17.5  > 13.333  -> +1
      idx7: 30    > 21.667  -> +1
      idx8: 30    == 30     ->  0  (tie)

    idx0, idx1: warm-up, no slow SMA yet -> 0
    """
    closes = [10, 20, 10, 5, 5, 5, 30, 30, 30]
    dates = pd.bdate_range("2024-01-01", periods=len(closes))
    df = pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": [1000] * len(closes),
        }
    )
    return df


EXPECTED_RAW = [0, 0, 1, -1, -1, 0, 1, 1, 0]
EXPECTED_LONG_ONLY = [0, 0, 1, 0, 0, 0, 1, 1, 0]


def test_crossover_signal_matches_hand_computed_values_raw():
    df = _make_known_df()
    strat = MovingAverageCrossover(fast_window=2, slow_window=3, long_only=False)
    signal = strat.run(df)
    assert signal.tolist() == EXPECTED_RAW


def test_crossover_signal_matches_hand_computed_values_long_only():
    df = _make_known_df()
    strat = MovingAverageCrossover(fast_window=2, slow_window=3, long_only=True)
    signal = strat.run(df)
    assert signal.tolist() == EXPECTED_LONG_ONLY


def test_crossover_default_windows():
    strat = MovingAverageCrossover()
    assert strat.fast_window == 50
    assert strat.slow_window == 200
    assert strat.name == "ma_crossover_50_200"


def test_crossover_rejects_fast_not_less_than_slow():
    with pytest.raises(ValueError):
        MovingAverageCrossover(fast_window=200, slow_window=50)
    with pytest.raises(ValueError):
        MovingAverageCrossover(fast_window=50, slow_window=50)


def test_crossover_rejects_non_positive_windows():
    with pytest.raises(ValueError):
        MovingAverageCrossover(fast_window=0, slow_window=10)
    with pytest.raises(ValueError):
        MovingAverageCrossover(fast_window=-5, slow_window=10)


def test_signal_has_no_nan_and_no_bad_values():
    df = _make_known_df()
    strat = MovingAverageCrossover(fast_window=2, slow_window=3)
    signal = strat.run(df)
    assert not signal.isna().any()
    assert set(signal.unique()).issubset({-1, 0, 1})


def test_run_rejects_missing_columns():
    df = _make_known_df().drop(columns=["volume"])
    strat = MovingAverageCrossover(fast_window=2, slow_window=3)
    with pytest.raises(ValueError, match="missing required columns"):
        strat.run(df)


def test_run_rejects_unsorted_dates():
    df = _make_known_df().iloc[::-1].reset_index(drop=True)
    strat = MovingAverageCrossover(fast_window=2, slow_window=3)
    with pytest.raises(ValueError, match="sorted ascending"):
        strat.run(df)


class _BadLengthStrategy(Strategy):
    name = "bad_length"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series([0] * (len(df) - 1))


class _BadValueStrategy(Strategy):
    name = "bad_value"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series([2] * len(df), index=df.index)


class _NaNStrategy(Strategy):
    name = "nan_strategy"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        s = pd.Series([0] * len(df), index=df.index, dtype=float)
        s.iloc[0] = float("nan")
        return s


def test_run_rejects_wrong_length_signal():
    df = _make_known_df()
    with pytest.raises(ValueError, match="length"):
        _BadLengthStrategy().run(df)


def test_run_rejects_out_of_range_signal_values():
    df = _make_known_df()
    with pytest.raises(ValueError, match=r"outside \{"):
        _BadValueStrategy().run(df)


def test_run_rejects_nan_signal():
    df = _make_known_df()
    with pytest.raises(ValueError, match="NaN"):
        _NaNStrategy().run(df)
