"""
Unit tests for ml/features.compute_forward_volatility and
ml/position_sizing.py.

compute_forward_volatility is checked against an independently written
(but equivalent) formula rather than hand-substituted decimals -- the
values involve irrational sqrt(2) terms from a 2-point sample std that
are easy to transcribe wrong by hand; an independent numpy computation of
the same "std of the next `horizon` returns" definition is a more
reliable check. VolatilityForecaster's train/predict split reuses the
same leakage pattern already verified in test_ml_signal.py, so here it's
checked once, directly, rather than repeating every angle of that test.
"""

import numpy as np
import pandas as pd
import pytest

from ml.features import FEATURE_COLUMNS, compute_features, compute_forward_volatility
from ml.position_sizing import VolatilityForecaster, inverse_vol_position_size


def test_forward_volatility_matches_independent_computation():
    closes = pd.Series([100, 102, 101, 105, 103, 108, 106], dtype=float)
    horizon = 2
    forward_vol = compute_forward_volatility(pd.DataFrame({"close": closes}), horizon=horizon)

    returns = closes.pct_change(1)
    expected = pd.Series(index=closes.index, dtype=float)
    for t in range(len(closes)):
        window = returns.iloc[t + 1 : t + 1 + horizon]
        if len(window) == horizon:
            expected.iloc[t] = window.std()  # pandas default ddof=1, matches rolling().std()
        else:
            expected.iloc[t] = float("nan")

    pd.testing.assert_series_equal(forward_vol, expected, check_names=False)


def test_forward_volatility_last_horizon_rows_are_nan():
    closes = pd.Series(np.linspace(100, 120, 20))
    forward_vol = compute_forward_volatility(pd.DataFrame({"close": closes}), horizon=5)
    assert forward_vol.iloc[-5:].isna().all()
    assert forward_vol.iloc[:-5].notna().all()


class _StubRegressor:
    def __init__(self, predictions):
        self.predictions = list(predictions)
        self.fit_X = None
        self.fit_y = None
        self.predict_X = None

    def fit(self, X, y):
        self.fit_X = np.array(X)
        self.fit_y = np.array(y)
        return self

    def predict(self, X):
        self.predict_X = np.array(X)
        return np.array(self.predictions[: len(X)])


def _make_df(n=280, seed=11):
    dates = pd.bdate_range("2023-01-01", periods=n)
    rng = np.random.RandomState(seed)
    closes = 100 + np.cumsum(rng.normal(0, 1, size=n))
    closes = np.abs(closes) + 50
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


def test_volatility_forecaster_trains_only_on_rows_before_min_train_rows():
    df = _make_df(280)
    min_train_rows = 230

    features = compute_features(df)
    from ml.features import compute_forward_volatility as fwd_vol

    target = fwd_vol(df, horizon=10)
    train_mask = features.index < min_train_rows
    valid_train = features.loc[train_mask].notna().all(axis=1) & target.loc[train_mask].notna()
    expected_train_X = features.loc[train_mask].loc[valid_train][FEATURE_COLUMNS].values
    expected_train_y = target.loc[train_mask].loc[valid_train].values

    predict_mask = (features.index >= min_train_rows) & features.notna().all(axis=1)
    n_predict = predict_mask.sum()

    stub = _StubRegressor(predictions=[0.02] * n_predict)
    forecaster = VolatilityForecaster(forward_horizon=10, min_train_rows=min_train_rows, model_factory=lambda: stub)
    predicted = forecaster.fit_predict(df)

    np.testing.assert_array_almost_equal(stub.fit_X, expected_train_X)
    np.testing.assert_array_almost_equal(stub.fit_y, expected_train_y)
    assert predicted.iloc[:min_train_rows].isna().all()
    assert (predicted.iloc[min_train_rows:] == 0.02).sum() == n_predict


def test_volatility_forecaster_insufficient_rows_returns_all_nan():
    df = _make_df(100)
    forecaster = VolatilityForecaster(min_train_rows=252)
    predicted = forecaster.fit_predict(df)
    assert predicted.isna().all()


def test_inverse_vol_position_size_hand_computed():
    # target_vol=0.02, min=0.1, max=0.95
    # predicted_vol=0.01 -> raw=2.0 -> clipped to 0.95
    # predicted_vol=0.02 -> raw=1.0 -> clipped to 0.95
    # predicted_vol=0.04 -> raw=0.5 -> stays 0.5
    # predicted_vol=0.50 -> raw=0.04 -> clipped to 0.10
    predicted = pd.Series([0.01, 0.02, 0.04, 0.50])
    sizes = inverse_vol_position_size(predicted, target_vol=0.02, min_size=0.10, max_size=0.95)
    assert sizes.tolist() == pytest.approx([0.95, 0.95, 0.5, 0.10])


def test_inverse_vol_position_size_propagates_nan():
    predicted = pd.Series([0.02, float("nan"), 0.04])
    sizes = inverse_vol_position_size(predicted)
    assert sizes.iloc[1] != sizes.iloc[1]  # NaN != NaN


def test_inverse_vol_position_size_rejects_invalid_bounds():
    predicted = pd.Series([0.02])
    with pytest.raises(ValueError):
        inverse_vol_position_size(predicted, target_vol=0)
    with pytest.raises(ValueError):
        inverse_vol_position_size(predicted, min_size=0.9, max_size=0.5)
