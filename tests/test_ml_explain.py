"""
Unit tests for ml/explain.py.

The property that actually matters for SHAP to be trustworthy here is
additivity: expected_value + sum(shap values for a row) must exactly
reconstruct that row's real model output. That's checked directly against
real RandomForestClassifier/Regressor predictions (not mocked), for both
model types explain() supports. top_features/global_importance are
checked against a hand-constructed Explanation with known values, so
their ranking logic is verified independent of any real model or SHAP
computation.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ml.explain import Explanation, explain


def _make_classification_data(n=150, n_features=4, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.normal(size=(n, n_features)), columns=[f"f{i}" for i in range(n_features)])
    y = (X["f0"] + X["f1"] > 0).astype(float)
    return X, y


def _make_regression_data(n=150, n_features=4, seed=1):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.normal(size=(n, n_features)), columns=[f"f{i}" for i in range(n_features)])
    y = X["f0"] * 2 + X["f1"] - X["f2"]
    return X, y


def test_classifier_shap_values_reconstruct_predicted_probability():
    X, y = _make_classification_data()
    model = RandomForestClassifier(n_estimators=30, max_depth=4, random_state=0).fit(X.values, y.values)

    explanation = explain(model, X.iloc[:10], positive_class=1.0)

    reconstructed = explanation.expected_value + explanation.shap_values.sum(axis=1).values
    np.testing.assert_allclose(reconstructed, explanation.predictions, atol=1e-8)


def test_regressor_shap_values_reconstruct_prediction():
    X, y = _make_regression_data()
    model = RandomForestRegressor(n_estimators=30, max_depth=4, random_state=0).fit(X.values, y.values)

    explanation = explain(model, X.iloc[:10])

    reconstructed = explanation.expected_value + explanation.shap_values.sum(axis=1).values
    np.testing.assert_allclose(reconstructed, explanation.predictions, atol=1e-8)


def test_explanation_feature_names_and_shape():
    X, y = _make_classification_data(n_features=5)
    model = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=0).fit(X.values, y.values)

    explanation = explain(model, X.iloc[:8])

    assert explanation.feature_names == list(X.columns)
    assert explanation.shap_values.shape == (8, 5)


def test_explain_rejects_empty_input():
    X, y = _make_classification_data()
    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X.values, y.values)
    with pytest.raises(ValueError):
        explain(model, X.iloc[:0])


def _make_hand_constructed_explanation() -> Explanation:
    shap_values = pd.DataFrame(
        {
            "rsi_14": [0.10, -0.02],
            "macd": [-0.05, 0.30],
            "volume_ratio_20d": [0.02, -0.01],
            "momentum_10d": [0.20, 0.05],
        }
    )
    return Explanation(
        feature_names=list(shap_values.columns),
        shap_values=shap_values,
        expected_value=0.5,
        predictions=np.array([0.77, 0.82]),
    )


def test_top_features_ranks_by_absolute_value_signed():
    explanation = _make_hand_constructed_explanation()

    top2_row0 = explanation.top_features(row_index=0, n=2)
    # row 0: |momentum_10d|=0.20 (largest), |rsi_14|=0.10 (second) -- both positive
    assert list(top2_row0.index) == ["momentum_10d", "rsi_14"]
    assert top2_row0["momentum_10d"] == pytest.approx(0.20)
    assert top2_row0["rsi_14"] == pytest.approx(0.10)

    top1_row1 = explanation.top_features(row_index=1, n=1)
    # row 1: |macd|=0.30 is the single largest magnitude, and it's positive
    assert list(top1_row1.index) == ["macd"]
    assert top1_row1["macd"] == pytest.approx(0.30)


def test_global_importance_is_mean_absolute_value_sorted_descending():
    explanation = _make_hand_constructed_explanation()
    importance = explanation.global_importance()

    # mean |value| per column:
    # rsi_14: (0.10+0.02)/2 = 0.06
    # macd: (0.05+0.30)/2 = 0.175
    # volume_ratio_20d: (0.02+0.01)/2 = 0.015
    # momentum_10d: (0.20+0.05)/2 = 0.125
    expected = pd.Series(
        {"macd": 0.175, "momentum_10d": 0.125, "rsi_14": 0.06, "volume_ratio_20d": 0.015}
    )
    pd.testing.assert_series_equal(importance, expected, check_names=False)
