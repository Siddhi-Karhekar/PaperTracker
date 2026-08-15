"""
Unit tests for strategy/ml_signal.py.

The critical property to verify here isn't prediction accuracy (a model
can be "correctly implemented" and still predict badly -- that's not a
bug) -- it's that the train/predict split never leaks: the fitted model
must only ever see rows before `min_train_rows`, and every signal in that
training span must be forced flat. A deterministic stub classifier
(records exactly what it was fit and predicted on, and returns
caller-controlled probabilities) makes this precisely checkable without
depending on RandomForestClassifier's actual learned behavior.
"""

import numpy as np
import pandas as pd
import pytest

from ml.features import FEATURE_COLUMNS, compute_features, compute_forward_labels
from strategy.ml_signal import MLClassifierStrategy


class _StubClassifier:
    """Records its fit/predict inputs; returns a caller-supplied probability sequence."""

    def __init__(self, proba_for_class_1):
        self.proba_for_class_1 = list(proba_for_class_1)
        self.fit_X = None
        self.fit_y = None
        self.predict_X = None
        self.classes_ = np.array([0.0, 1.0])

    def fit(self, X, y):
        self.fit_X = np.array(X)
        self.fit_y = np.array(y)
        return self

    def predict_proba(self, X):
        self.predict_X = np.array(X)
        n = len(X)
        probs = self.proba_for_class_1[:n]
        return np.array([[1 - p, p] for p in probs])


def _make_df(n=280, seed=7):
    dates = pd.bdate_range("2023-01-01", periods=n)
    rng = np.random.RandomState(seed)
    closes = 100 + np.cumsum(rng.normal(0, 1, size=n))
    closes = np.abs(closes) + 50
    volumes = rng.randint(1000, 5000, size=n).astype(float)
    return pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "high": closes + 1,
            "low": closes - 1,
            "close": closes,
            "volume": volumes,
        }
    )


def _expected_train_and_predict(df, min_train_rows, forward_horizon):
    features = compute_features(df)
    labels = compute_forward_labels(df, horizon=forward_horizon, min_move=0.0)

    train_mask = features.index < min_train_rows
    valid_train = features.loc[train_mask].notna().all(axis=1) & labels.loc[train_mask].notna()
    expected_train_X = features.loc[train_mask].loc[valid_train][FEATURE_COLUMNS].values
    expected_train_y = labels.loc[train_mask].loc[valid_train].values

    predict_mask = (features.index >= min_train_rows) & features.notna().all(axis=1)
    expected_predict_X = features.loc[predict_mask][FEATURE_COLUMNS].values
    predict_index = features.loc[predict_mask].index

    return expected_train_X, expected_train_y, expected_predict_X, predict_index


def test_training_data_never_includes_rows_at_or_after_train_end():
    df = _make_df(280)
    min_train_rows = 230
    expected_train_X, expected_train_y, expected_predict_X, predict_index = _expected_train_and_predict(
        df, min_train_rows, forward_horizon=5
    )

    stub = _StubClassifier(proba_for_class_1=[0.9] * len(expected_predict_X))
    strategy = MLClassifierStrategy(min_train_rows=min_train_rows, forward_horizon=5, model_factory=lambda: stub)
    strategy.run(df)

    assert stub.fit_X.shape == expected_train_X.shape
    np.testing.assert_array_almost_equal(stub.fit_X, expected_train_X)
    np.testing.assert_array_almost_equal(stub.fit_y, expected_train_y)
    # every training row's features came from strictly before min_train_rows
    assert stub.fit_X.shape[0] > 0


def test_predict_only_called_on_rows_from_train_end_onward():
    df = _make_df(280)
    min_train_rows = 230
    _, _, expected_predict_X, predict_index = _expected_train_and_predict(df, min_train_rows, forward_horizon=5)
    assert predict_index.min() >= min_train_rows  # sanity check on the test's own expectation

    stub = _StubClassifier(proba_for_class_1=[0.5] * len(expected_predict_X))
    strategy = MLClassifierStrategy(min_train_rows=min_train_rows, forward_horizon=5, model_factory=lambda: stub)
    strategy.run(df)

    np.testing.assert_array_almost_equal(stub.predict_X, expected_predict_X)


def test_signal_is_always_flat_before_train_end():
    df = _make_df(280)
    min_train_rows = 230
    stub = _StubClassifier(proba_for_class_1=[0.99] * 100)  # would be "long" everywhere if not masked
    strategy = MLClassifierStrategy(min_train_rows=min_train_rows, forward_horizon=5, model_factory=lambda: stub)

    signal = strategy.run(df)

    assert (signal.iloc[:min_train_rows] == 0).all()


def test_signal_thresholds_probability_correctly():
    df = _make_df(280)
    min_train_rows = 230
    _, _, expected_predict_X, predict_index = _expected_train_and_predict(df, min_train_rows, forward_horizon=5)
    n_predict = len(expected_predict_X)

    # First predicted row: high confidence up -> long. Second: neutral -> flat.
    proba_sequence = [0.9, 0.5] + [0.5] * (n_predict - 2)
    stub = _StubClassifier(proba_for_class_1=proba_sequence)
    strategy = MLClassifierStrategy(min_train_rows=min_train_rows, forward_horizon=5, prob_threshold=0.55, model_factory=lambda: stub)

    signal = strategy.run(df)

    first_predict_idx = predict_index[0]
    second_predict_idx = predict_index[1]
    assert signal.loc[first_predict_idx] == 1
    assert signal.loc[second_predict_idx] == 0


def test_long_only_never_emits_short_signal():
    df = _make_df(280)
    min_train_rows = 230
    _, _, expected_predict_X, _ = _expected_train_and_predict(df, min_train_rows, forward_horizon=5)
    stub = _StubClassifier(proba_for_class_1=[0.01] * len(expected_predict_X))  # very confident "down"
    strategy = MLClassifierStrategy(min_train_rows=min_train_rows, forward_horizon=5, long_only=True, model_factory=lambda: stub)

    signal = strategy.run(df)
    assert (signal >= 0).all()


def test_allow_short_emits_negative_signal_on_confident_down_prediction():
    df = _make_df(280)
    min_train_rows = 230
    _, _, expected_predict_X, predict_index = _expected_train_and_predict(df, min_train_rows, forward_horizon=5)
    stub = _StubClassifier(proba_for_class_1=[0.01] * len(expected_predict_X))
    strategy = MLClassifierStrategy(min_train_rows=min_train_rows, forward_horizon=5, long_only=False, model_factory=lambda: stub)

    signal = strategy.run(df)
    assert signal.loc[predict_index[0]] == -1


def test_not_enough_rows_returns_all_flat():
    df = _make_df(100)  # fewer rows than default min_train_rows=252
    strategy = MLClassifierStrategy(min_train_rows=252)
    signal = strategy.run(df)
    assert (signal == 0).all()


def test_invalid_prob_threshold_raises():
    with pytest.raises(ValueError):
        MLClassifierStrategy(prob_threshold=0.5)
    with pytest.raises(ValueError):
        MLClassifierStrategy(prob_threshold=1.0)


def test_signal_passes_base_strategy_contract():
    # Exercises the real RandomForestClassifier default, just to confirm
    # end-to-end wiring produces a contract-valid signal (base.Strategy.run
    # already asserts no NaN / values in {-1,0,1} / correct length).
    df = _make_df(280)
    strategy = MLClassifierStrategy(min_train_rows=230, forward_horizon=5)
    signal = strategy.run(df)
    assert len(signal) == len(df)
    assert set(signal.unique()).issubset({-1, 0, 1})


def test_last_run_artifacts_are_stashed_for_explainability():
    df = _make_df(280)
    strategy = MLClassifierStrategy(min_train_rows=230, forward_horizon=5)
    strategy.run(df)

    assert strategy.last_model is not None
    assert strategy.last_features is not None
    assert strategy.last_proba is not None
    assert strategy.last_train_end == 230
