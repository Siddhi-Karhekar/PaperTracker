"""
strategy/ml_signal.py

An ML-based alternative to the rule-based MovingAverageCrossover: a
classifier predicts, from the technical features in ml/features.py,
whether price will be higher `forward_horizon` bars from now, and the
predicted probability is thresholded into the same {-1, 0, 1}
target-position signal every other Strategy produces. Because it
implements the same Strategy interface, it plugs into run_backtest,
walk_forward_validate, and even live paper trading without any of those
modules knowing it's ML-backed.

How this avoids look-ahead bias without a separate training pipeline:
generate_signals(df) internally reserves the *first* `min_train_rows` of
whatever df it's given purely for fitting the model, and forces the
signal to 0 (flat) for that entire span -- those rows are training data,
not a trading record, and reporting "performance" on them would be
in-sample and misleading (the same honesty rule the moving-average
strategy applies to its own warm-up period). Every row from
`min_train_rows` onward is a genuine out-of-sample prediction: the model
never sees that row's features, or anything derived from prices at or
after it, during fitting.

This split is designed to line up with engine.validation.walk_forward_validate:
call it with warmup_days=min_train_rows (or larger) and each window's
"warmup" portion becomes exactly this strategy's training data, while the
real test window is 100% genuinely out-of-sample -- both from the outer
walk-forward split AND from this strategy's own internal split.

A trained model's predictions are only as good as its training data
suggests -- and daily-bar technical features on a single equity are
famously noisy and only weakly predictive at best. This is included as a
genuine extension point and a demonstration of MLOps-adjacent engineering
(feature pipeline, train/predict split without leakage, explainability),
not as a claim that it reliably beats the rule-based strategy. Compare
both honestly with engine.validation before trusting either.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ml.features import FEATURE_COLUMNS, compute_features, compute_forward_labels
from strategy.base import Strategy

logger = logging.getLogger(__name__)


def _default_model_factory():
    return RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=10, random_state=42, n_jobs=-1)


class MLClassifierStrategy(Strategy):
    def __init__(
        self,
        min_train_rows: int = 252,
        forward_horizon: int = 5,
        min_move: float = 0.0,
        prob_threshold: float = 0.55,
        long_only: bool = True,
        model_factory: Optional[Callable[[], object]] = None,
    ):
        if not (0.5 < prob_threshold < 1.0):
            raise ValueError(f"prob_threshold must be in (0.5, 1.0), got {prob_threshold}")
        self.min_train_rows = min_train_rows
        self.forward_horizon = forward_horizon
        self.min_move = min_move
        self.prob_threshold = prob_threshold
        self.long_only = long_only
        self.model_factory = model_factory or _default_model_factory
        self.name = f"ml_classifier_{min_train_rows}_{forward_horizon}"

        # Populated by generate_signals(), read by ml/explain.py and the
        # dashboard to explain the most recent run without recomputing it.
        self.last_model = None
        self.last_features: Optional[pd.DataFrame] = None
        self.last_proba: Optional[pd.Series] = None
        self.last_train_end: Optional[int] = None

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        flat = pd.Series(0, index=df.index, dtype=int)

        if len(df) <= self.min_train_rows:
            logger.warning(
                "%s: only %d rows given, need > min_train_rows=%d to leave anything to predict; returning flat.",
                self.name, len(df), self.min_train_rows,
            )
            return flat

        features = compute_features(df)
        labels = compute_forward_labels(df, horizon=self.forward_horizon, min_move=self.min_move)
        train_end = self.min_train_rows

        train_mask = features.index < train_end
        train_X = features.loc[train_mask]
        train_y = labels.loc[train_mask]
        valid_train = train_X.notna().all(axis=1) & train_y.notna()
        train_X, train_y = train_X.loc[valid_train], train_y.loc[valid_train]

        if len(train_X) < 20 or train_y.nunique() < 2:
            logger.warning(
                "%s: insufficient clean training data (%d usable rows, %d classes); returning flat.",
                self.name, len(train_X), train_y.nunique(),
            )
            return flat

        model = self.model_factory()
        model.fit(train_X[FEATURE_COLUMNS].values, train_y.values)

        predict_mask = (features.index >= train_end) & features.notna().all(axis=1)
        predict_X = features.loc[predict_mask]

        proba = pd.Series(0.5, index=df.index, dtype=float)  # neutral default outside the predicted zone
        if not predict_X.empty:
            up_class_idx = list(model.classes_).index(1.0)
            proba.loc[predict_X.index] = model.predict_proba(predict_X[FEATURE_COLUMNS].values)[:, up_class_idx]

        self.last_model = model
        self.last_features = features
        self.last_proba = proba
        self.last_train_end = train_end

        signal = flat.copy()
        signal.loc[proba > self.prob_threshold] = 1
        if not self.long_only:
            signal.loc[proba < (1 - self.prob_threshold)] = -1

        # Training rows and rows without a full predict-side feature vector
        # (e.g. right at train_end if the slowest feature's warm-up hasn't
        # cleared yet) are never real trading decisions -- force flat.
        signal.loc[signal.index < train_end] = 0
        signal.loc[(signal.index >= train_end) & ~predict_mask] = 0

        return signal
