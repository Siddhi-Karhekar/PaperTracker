"""
Unit tests for ml/features.py.

RSI is verified fully by hand (a plain rolling mean of gains/losses, no
recursive smoothing, so exact arithmetic is tractable -- see the comment
above test_rsi_matches_hand_computed_values). MACD's exponential moving
averages recurse from the start of the series in a way that's easy to get
subtly wrong by hand, so instead it's cross-checked against an
independently written pandas computation of the same standard formula --
this still catches span/formula/off-by-one bugs without risking a wrong
hand-derived "expected" value.
"""

import numpy as np
import pandas as pd
import pytest

from ml.features import FEATURE_COLUMNS, _macd, _rsi, compute_features, compute_forward_labels


def test_rsi_matches_hand_computed_values():
    # closes: [10, 12, 11, 14, 13, 16], window=3
    # delta:  [nan, 2, -1, 3, -1, 3]
    # gain:   [nan, 2,  0, 3,  0, 3]
    # loss:   [nan, 0,  1, 0,  1, 0]
    #
    # idx3: avg_gain = mean(gain[1:4]) = mean(2,0,3) = 1.6667
    #       avg_loss = mean(loss[1:4]) = mean(0,1,0) = 0.3333
    #       RS = 5.0 -> RSI = 100 - 100/6 = 83.3333
    # idx4: avg_gain = mean(0,3,0) = 1.0, avg_loss = mean(1,0,1) = 0.6667
    #       RS = 1.5 -> RSI = 100 - 100/2.5 = 60.0
    # idx5: avg_gain = mean(3,0,3) = 2.0, avg_loss = mean(0,1,0) = 0.3333
    #       RS = 6.0 -> RSI = 100 - 100/7 = 85.7143
    closes = pd.Series([10, 12, 11, 14, 13, 16], dtype=float)
    rsi = _rsi(closes, window=3)

    assert rsi.iloc[:3].isna().all()
    assert rsi.iloc[3] == pytest.approx(83.3333, abs=1e-3)
    assert rsi.iloc[4] == pytest.approx(60.0, abs=1e-3)
    assert rsi.iloc[5] == pytest.approx(85.7143, abs=1e-3)


def test_rsi_all_gains_saturates_at_100():
    closes = pd.Series([10, 11, 12, 13, 14], dtype=float)  # strictly rising -- no losses at all
    rsi = _rsi(closes, window=3)
    assert rsi.iloc[3] == pytest.approx(100.0)
    assert rsi.iloc[4] == pytest.approx(100.0)


def test_rsi_flat_price_is_neutral_50():
    closes = pd.Series([10.0] * 6)
    rsi = _rsi(closes, window=3)
    assert rsi.iloc[3] == pytest.approx(50.0)


def test_macd_matches_independent_ewm_computation():
    closes = pd.Series([10, 11, 13, 12, 15, 14, 16, 18, 17, 20], dtype=float)
    macd_line, signal_line, hist = _macd(closes, fast=3, slow=5, signal=2)

    expected_fast = closes.ewm(span=3, adjust=False, min_periods=3).mean()
    expected_slow = closes.ewm(span=5, adjust=False, min_periods=5).mean()
    expected_macd = expected_fast - expected_slow
    expected_signal = expected_macd.ewm(span=2, adjust=False, min_periods=2).mean()
    expected_hist = expected_macd - expected_signal

    pd.testing.assert_series_equal(macd_line, expected_macd, check_names=False)
    pd.testing.assert_series_equal(signal_line, expected_signal, check_names=False)
    pd.testing.assert_series_equal(hist, expected_hist, check_names=False)


def _make_df(n=210):
    dates = pd.bdate_range("2024-01-01", periods=n)
    rng = np.random.RandomState(42)
    closes = 100 + np.cumsum(rng.normal(0, 1, size=n))
    closes = np.abs(closes) + 50  # keep strictly positive
    return pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "high": closes + 1,
            "low": closes - 1,
            "close": closes,
            "volume": rng.randint(1000, 5000, size=n).astype(float),
        }
    )


def test_compute_features_has_expected_columns_and_index():
    df = _make_df(210)
    features = compute_features(df)
    assert list(features.columns) == FEATURE_COLUMNS
    assert list(features.index) == list(df.index)


def test_compute_features_warmup_rows_are_nan_then_resolve():
    df = _make_df(210)
    features = compute_features(df)
    # sma_ratio_10_50 needs 50 bars; close_to_sma200 needs 200
    assert features["sma_ratio_10_50"].iloc[:49].isna().all()
    assert features["sma_ratio_10_50"].iloc[49:].notna().all()
    assert features["close_to_sma200"].iloc[:199].isna().all()
    assert features["close_to_sma200"].iloc[199:].notna().all()


def test_compute_features_simple_columns_match_direct_formulas():
    df = _make_df(60)
    features = compute_features(df)
    close = df["close"]

    pd.testing.assert_series_equal(features["return_1d"], close.pct_change(1), check_names=False)
    pd.testing.assert_series_equal(features["momentum_10d"], close - close.shift(10), check_names=False)

    expected_volume_ratio = df["volume"] / df["volume"].rolling(20, min_periods=20).mean()
    pd.testing.assert_series_equal(features["volume_ratio_20d"], expected_volume_ratio, check_names=False)


def test_compute_forward_labels_hand_computed():
    # closes: [10, 11, 9, 12, 8], horizon=2, min_move=0.0
    # forward_return[t] = close[t+2]/close[t] - 1
    #   idx0: 9/10 - 1  = -0.10  -> label 0
    #   idx1: 12/11 - 1 = +0.0909 -> label 1
    #   idx2: 8/9 - 1   = -0.111 -> label 0
    #   idx3, idx4: no close 2 bars ahead -> NaN
    df = pd.DataFrame({"close": [10.0, 11.0, 9.0, 12.0, 8.0]})
    labels = compute_forward_labels(df, horizon=2, min_move=0.0)

    assert labels.iloc[0] == 0.0
    assert labels.iloc[1] == 1.0
    assert labels.iloc[2] == 0.0
    assert labels.iloc[3:].isna().all()


def test_compute_forward_labels_respects_min_move_threshold():
    # A tiny positive move shouldn't count as "up" if min_move filters it out.
    df = pd.DataFrame({"close": [100.0, 100.5, 100.0]})  # +0.5% at idx0 over horizon=1
    labels = compute_forward_labels(df, horizon=1, min_move=0.01)  # require >1% to count as "up"
    assert labels.iloc[0] == 0.0
