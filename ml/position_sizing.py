"""
ml/position_sizing.py

Volatility-targeted position sizing: instead of always committing a fixed
fraction of cash to a new position (engine.portfolio.run_backtest's
default `position_size_pct=0.95`), predict how volatile the stock is
likely to be over the next few days and size the position inversely to
that -- commit less when risk looks high, more when it looks low. This is
a standard "vol-targeting" approach in real portfolio construction,
implemented here with an ML volatility forecast instead of just trailing
realized volatility.

`VolatilityForecaster` follows the exact same train/predict split as
strategy.ml_signal.MLClassifierStrategy (fit only on rows before
min_train_rows, predict only from there onward, everything before that
is NaN/unusable) -- same leakage-avoidance reasoning applies here
unchanged, and the two are meant to be read together.

The output of `inverse_vol_position_size()` is a pd.Series you can pass
directly as `run_backtest`'s `position_size_pct` argument (which accepts
either a fixed float or a per-row Series, see engine/portfolio.py).
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from ml.features import FEATURE_COLUMNS, compute_features, compute_forward_volatility

logger = logging.getLogger(__name__)


def _default_regressor_factory():
    return RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=10, random_state=42, n_jobs=-1)


class VolatilityForecaster:
    def __init__(
        self,
        forward_horizon: int = 10,
        min_train_rows: int = 252,
        model_factory: Optional[Callable[[], object]] = None,
    ):
        self.forward_horizon = forward_horizon
        self.min_train_rows = min_train_rows
        self.model_factory = model_factory or _default_regressor_factory

        self.last_model = None
        self.last_features: Optional[pd.DataFrame] = None

    def fit_predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Returns predicted forward daily-return volatility per row of df.
        Rows before min_train_rows (used only to fit) and rows where the
        model wasn't confidently applicable (missing features) are NaN --
        callers (engine.portfolio.run_backtest) are expected to fall back
        to a fixed default sizing for NaN rows, not skip the trade.
        """
        predicted = pd.Series(float("nan"), index=df.index, dtype=float)

        if len(df) <= self.min_train_rows:
            logger.warning(
                "VolatilityForecaster: only %d rows given, need > min_train_rows=%d; returning all-NaN.",
                len(df), self.min_train_rows,
            )
            return predicted

        features = compute_features(df)
        target = compute_forward_volatility(df, horizon=self.forward_horizon)
        train_end = self.min_train_rows

        train_mask = features.index < train_end
        valid_train = features.loc[train_mask].notna().all(axis=1) & target.loc[train_mask].notna()
        train_X = features.loc[train_mask].loc[valid_train][FEATURE_COLUMNS]
        train_y = target.loc[train_mask].loc[valid_train]

        if len(train_X) < 20:
            logger.warning("VolatilityForecaster: insufficient clean training data (%d rows); returning all-NaN.", len(train_X))
            return predicted

        model = self.model_factory()
        model.fit(train_X.values, train_y.values)

        predict_mask = (features.index >= train_end) & features.notna().all(axis=1)
        predict_X = features.loc[predict_mask][FEATURE_COLUMNS]
        if not predict_X.empty:
            predicted.loc[predict_X.index] = model.predict(predict_X.values)

        self.last_model = model
        self.last_features = features
        return predicted


def inverse_vol_position_size(
    predicted_vol: pd.Series,
    target_vol: float = 0.02,
    min_size: float = 0.10,
    max_size: float = 0.95,
) -> pd.Series:
    """
    Converts a predicted daily-volatility series into a position-size
    fraction: size = target_vol / predicted_vol, clipped to [min_size,
    max_size]. A stock predicted to be twice as volatile as the target
    gets roughly half the position size; one predicted to be much calmer
    than target gets capped at max_size rather than an unrealistic
    oversized bet. NaN predictions (warm-up, insufficient data) pass
    through as NaN -- run_backtest falls back to its own default sizing
    for those rows rather than this function inventing a number.
    """
    if target_vol <= 0:
        raise ValueError(f"target_vol must be positive, got {target_vol}")
    if not (0 < min_size <= max_size <= 1):
        raise ValueError(f"require 0 < min_size <= max_size <= 1, got min_size={min_size}, max_size={max_size}")

    raw_size = target_vol / predicted_vol
    return raw_size.clip(lower=min_size, upper=max_size)
