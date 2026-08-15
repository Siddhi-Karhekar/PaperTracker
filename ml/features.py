"""
ml/features.py

Technical feature engineering from OHLCV data, for the ML strategy
(strategy/ml_signal.py) and the ML position sizer (ml/position_sizing.py).

Every feature here is backward-looking only -- computed from a bar's own
close/high/low/volume and the bars before it, never anything after it.
That's what makes it safe to compute features for the *entire* history in
one pass (as `compute_features` does) without introducing look-ahead
bias: a feature value at row t never depends on row t+1 onward. The
leakage risk in an ML strategy isn't in feature computation, it's in
which rows get used to *fit* the model -- that's handled in
strategy/ml_signal.py, not here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Every column this module can produce. strategy/ml_signal.py trains and
# predicts on exactly this list, so adding a feature here automatically
# flows through to the model without touching that file.
FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_10d",
    "volatility_10d",
    "volatility_20d",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "sma_ratio_10_50",
    "close_to_sma50",
    "close_to_sma200",
    "momentum_10d",
    "volume_ratio_20d",
    "bollinger_pct_b",
]


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index (Wilder's original, via a simple rolling mean
    rather than Wilder's smoothing -- close enough for a feature input and
    much easier to hand-verify).

    RSI = 100 - 100 / (1 + RS), RS = avg gain / avg loss over `window` bars.
    A pure uptrend (no losses) saturates RSI at 100; a pure downtrend
    saturates it at 0.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # avg_loss == 0 with avg_gain > 0 means every move was a gain -> RSI = 100
    rsi = rsi.where(avg_loss != 0, 100.0)
    # both zero (flat price) -> RSI is conventionally treated as neutral (50)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50.0)
    return rsi


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """
    MACD = EMA(fast) - EMA(slow); signal line = EMA(MACD, signal window);
    histogram = MACD - signal. Standard definition, using pandas' `ewm`
    (exponential weighted mean) for the EMAs.
    """
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: OHLCV DataFrame (date, open, high, low, close, volume), sorted
        ascending. Returns a DataFrame with the same index as df and
        exactly FEATURE_COLUMNS as columns. Rows before enough history has
        accumulated (e.g. before the 200-day SMA has 200 bars) are NaN --
        callers are responsible for dropping/handling those, the same way
        Strategy subclasses handle their own warm-up period.
    """
    close = df["close"]
    volume = df["volume"]

    features = pd.DataFrame(index=df.index)

    features["return_1d"] = close.pct_change(1)
    features["return_5d"] = close.pct_change(5)
    features["return_10d"] = close.pct_change(10)

    daily_return = close.pct_change(1)
    features["volatility_10d"] = daily_return.rolling(10, min_periods=10).std()
    features["volatility_20d"] = daily_return.rolling(20, min_periods=20).std()

    features["rsi_14"] = _rsi(close, window=14)

    macd_line, signal_line, hist = _macd(close)
    features["macd"] = macd_line
    features["macd_signal"] = signal_line
    features["macd_hist"] = hist

    sma_10 = close.rolling(10, min_periods=10).mean()
    sma_50 = close.rolling(50, min_periods=50).mean()
    sma_200 = close.rolling(200, min_periods=200).mean()
    features["sma_ratio_10_50"] = sma_10 / sma_50
    features["close_to_sma50"] = close / sma_50
    features["close_to_sma200"] = close / sma_200

    features["momentum_10d"] = close - close.shift(10)

    avg_volume_20 = volume.rolling(20, min_periods=20).mean()
    features["volume_ratio_20d"] = volume / avg_volume_20.replace(0, np.nan)

    rolling_mean_20 = close.rolling(20, min_periods=20).mean()
    rolling_std_20 = close.rolling(20, min_periods=20).std()
    lower_band = rolling_mean_20 - 2 * rolling_std_20
    upper_band = rolling_mean_20 + 2 * rolling_std_20
    band_width = (upper_band - lower_band).replace(0, np.nan)
    features["bollinger_pct_b"] = (close - lower_band) / band_width

    return features[FEATURE_COLUMNS]


def compute_forward_labels(df: pd.DataFrame, horizon: int = 5, min_move: float = 0.0) -> pd.Series:
    """
    Binary label for training: 1 if close `horizon` bars ahead is higher
    than today's close by more than `min_move` (as a fraction, e.g. 0.0
    means "any positive move counts"), else 0. The last `horizon` rows
    have no future close to compare against and are NaN -- these must
    never be used as training examples (see strategy/ml_signal.py, which
    drops them explicitly rather than accidentally training on a NaN
    label pandas would otherwise silently coerce).
    """
    close = df["close"]
    forward_return = close.shift(-horizon) / close - 1.0
    label = (forward_return > min_move).astype(float)
    label[forward_return.isna()] = np.nan
    return label


def compute_forward_volatility(df: pd.DataFrame, horizon: int = 10) -> pd.Series:
    """
    Regression target for ml/position_sizing.py's volatility forecaster:
    the standard deviation of daily returns over the *next* `horizon`
    bars (t+1 .. t+horizon), not including today's own return. Like
    compute_forward_labels, this is NaN for the last `horizon` rows,
    which have no full forward window yet.

    Implementation note: `returns.rolling(horizon).std()` at row j gives
    the std of returns[j-horizon+1 .. j]. Shifting that whole series
    backward by `horizon` rows realigns it so the value at row t is the
    std of returns[t+1 .. t+horizon] -- exactly the forward window this
    function is meant to return.
    """
    daily_return = df["close"].pct_change(1)
    return daily_return.rolling(window=horizon, min_periods=horizon).std().shift(-horizon)
