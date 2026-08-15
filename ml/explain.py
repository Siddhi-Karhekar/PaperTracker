"""
ml/explain.py

Explainable AI (XAI) for the ML models in strategy/ml_signal.py and
ml/position_sizing.py: SHAP (SHapley Additive exPlanations) values show,
for a single prediction, how much each input feature pushed the model's
output up or down relative to its average output over the training data
-- decomposing "why did the model say this" into one number per feature,
rather than leaving a random forest's ~200-tree vote as an opaque black
box.

Uses shap.TreeExplainer, which computes *exact* SHAP values for
tree-based models (our RandomForestClassifier/Regressor) in polynomial
time, rather than the sampling-based approximation general-purpose SHAP
explainers need for arbitrary models. The additivity property this relies
on -- expected_value + sum(shap_values for a row) == the model's actual
output for that row, exactly -- is asserted directly in this module's own
tests, not just assumed from SHAP's documentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import shap


@dataclass
class Explanation:
    feature_names: List[str]
    shap_values: pd.DataFrame  # one row per prediction, one column per feature
    expected_value: float      # the model's average output over its training data (the SHAP "base value")
    predictions: np.ndarray    # the model's actual output for each row (probability, for classifiers)

    def top_features(self, row_index=0, n: int = 5) -> pd.Series:
        """
        The n features with the largest |SHAP value| for one prediction
        (by positional row_index into shap_values), signed -- positive
        means that feature pushed this specific prediction up, negative
        means it pushed it down.
        """
        row = self.shap_values.iloc[row_index]
        ranked = row.abs().sort_values(ascending=False).index[:n]
        return row.reindex(ranked)

    def global_importance(self) -> pd.Series:
        """Mean |SHAP value| per feature across all explained rows -- which features matter most overall, not just for one prediction."""
        return self.shap_values.abs().mean(axis=0).sort_values(ascending=False)


def explain(model, X: pd.DataFrame, feature_names=None, positive_class: float = 1.0) -> Explanation:
    """
    model: a fitted tree-based sklearn estimator (RandomForestClassifier
        or RandomForestRegressor -- what strategy.ml_signal and
        ml.position_sizing use).
    X: feature rows to explain, same columns/order the model was trained on.
    positive_class: for classifiers, which class's SHAP values to return
        (ignored for regressors). Defaults to 1.0, matching how
        strategy.ml_signal.MLClassifierStrategy labels "price went up."
    """
    if X.empty:
        raise ValueError("Cannot explain an empty set of rows")

    feature_names = list(feature_names or X.columns)
    explainer = shap.TreeExplainer(model)
    raw_shap = explainer.shap_values(X.values)

    is_classifier = hasattr(model, "predict_proba")
    if is_classifier:
        class_idx = list(model.classes_).index(positive_class)
        shap_matrix = raw_shap[:, :, class_idx]
        expected_value = float(np.asarray(explainer.expected_value)[class_idx])
        predictions = model.predict_proba(X.values)[:, class_idx]
    else:
        shap_matrix = raw_shap
        expected_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])
        predictions = model.predict(X.values)

    shap_df = pd.DataFrame(shap_matrix, columns=feature_names, index=X.index)
    return Explanation(
        feature_names=feature_names,
        shap_values=shap_df,
        expected_value=expected_value,
        predictions=np.asarray(predictions, dtype=float),
    )
