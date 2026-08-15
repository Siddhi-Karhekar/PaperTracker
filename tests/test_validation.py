"""
Unit tests for engine/validation.py.

The end-to-end scenario uses a trivial, test-only ConstantPositionStrategy
(always long, or always flat, regardless of price) on a monotonically
rising, zero-cost price series -- deliberately simple so the walk-forward
loop's own orchestration (windowing, param selection, out-of-sample
evaluation, stitching) can be verified by hand, independent of whatever
the real MovingAverageCrossover math does (that's already covered in
tests/test_strategy.py).
"""

import pandas as pd
import pytest

from engine.execution import CostModel
from engine.validation import WalkForwardResult, _generate_windows, walk_forward_validate
from strategy.base import Strategy

ZERO_COST = CostModel(
    brokerage_pct=0.0, stt_pct=0.0, exchange_txn_pct=0.0, stamp_duty_pct=0.0,
    sebi_turnover_pct=0.0, gst_pct=0.0, dp_charges_flat=0.0,
)


class _ConstantPositionStrategy(Strategy):
    def __init__(self, position: int):
        if position not in (0, 1):
            raise ValueError("test strategy only supports 0 or 1")
        self.position = position
        self.name = f"const_{position}"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(self.position, index=df.index, dtype=int)


def _strategy_factory(**params):
    return _ConstantPositionStrategy(**params)


def _make_rising_df(n=8):
    # opens: [100,102,104,106,108,110,112,114]  closes: [101,103,105,107,109,111,113,115]
    dates = pd.bdate_range("2024-01-01", periods=n)
    opens = [100 + 2 * i for i in range(n)]
    closes = [101 + 2 * i for i in range(n)]
    return pd.DataFrame(
        {
            "date": dates,
            "open": opens,
            "high": [o + 2 for o in opens],
            "low": [o - 2 for o in opens],
            "close": closes,
            "volume": [1000] * n,
        }
    )


def test_generate_windows_matches_hand_computed_boundaries():
    # n=8, train=4, test=2, step=2:
    #   window0: train [0,4), test [4,6)
    #   window1: train [2,6), test [6,8)
    #   next train_start=4 -> train_end=8, test_end=10 > 8 -> stop
    windows = _generate_windows(n_rows=8, train_days=4, test_days=2, step_days=2)
    assert windows == [(0, 4, 4, 6), (2, 6, 6, 8)]


def test_generate_windows_empty_when_data_too_short():
    assert _generate_windows(n_rows=5, train_days=4, test_days=2, step_days=2) == []


def test_walk_forward_raises_when_no_windows_fit():
    df = _make_rising_df(5)
    with pytest.raises(ValueError, match="No windows fit"):
        walk_forward_validate(
            df, param_grid=[{"position": 1}], strategy_factory=_strategy_factory,
            train_days=4, test_days=2, cost_model=ZERO_COST,
        )


def test_walk_forward_picks_long_over_flat_on_rising_market():
    df = _make_rising_df(8)
    result = walk_forward_validate(
        df,
        param_grid=[{"position": 0}, {"position": 1}],
        strategy_factory=_strategy_factory,
        train_days=4,
        test_days=2,
        step_days=2,
        cost_model=ZERO_COST,
        initial_capital=1000.0,
        position_size_pct=1.0,
    )

    assert isinstance(result, WalkForwardResult)
    assert len(result.windows) == 2
    for w in result.windows:
        # On a monotonically rising, zero-cost market, being long always
        # beats staying flat (flat's Sharpe is exactly 0.0 by definition
        # of a constant equity curve), so position=1 must win every window.
        assert w.best_params == {"position": 1}
        assert w.train_metric > 0.0


def test_walk_forward_stitched_equity_curve_matches_hand_computed_values():
    # See module docstring math below -- both windows execute a single BUY
    # at their test window's 2nd bar (index within the 2-bar test slice),
    # since ConstantPositionStrategy(1) sets signal=1 from bar 0 and the
    # look-ahead-bias fix defers execution to the next bar's open.
    #
    # Window0 test_df = df.iloc[4:6] -> opens/closes [108/109, 110/111]
    #   day0: no trade yet -> equity = 1000
    #   day1: BUY qty = floor(1000/110) = 9 -> cash = 1000 - 990 = 10
    #         mark-to-market @ close=111 -> equity = 10 + 9*111 = 1009
    #   window0 equity curve: [1000, 1009]
    #
    # Window1 test_df = df.iloc[6:8] -> opens/closes [112/113, 114/115]
    #   day0: equity = 1000 (fresh backtest, own initial_capital)
    #   day1: BUY qty = floor(1000/114) = 8 -> cash = 1000 - 912 = 88
    #         mark-to-market @ close=115 -> equity = 88 + 8*115 = 1008
    #   window1 equity curve: [1000, 1008]
    #
    # Stitching: window0 contributes its own equity path directly (running
    # capital starts at initial_capital=1000):
    #   stitched so far: [1000, 1009], running_capital = 1009
    # window1's multiplier series = [1000/1000, 1008/1000] = [1.0, 1.008]
    #   stitched segment = 1009 * [1.0, 1.008] = [1009, 1017.072]
    #
    # Full stitched curve: [1000, 1009, 1009, 1017.072]
    df = _make_rising_df(8)
    result = walk_forward_validate(
        df,
        param_grid=[{"position": 0}, {"position": 1}],
        strategy_factory=_strategy_factory,
        train_days=4,
        test_days=2,
        step_days=2,
        cost_model=ZERO_COST,
        initial_capital=1000.0,
        position_size_pct=1.0,
    )

    assert result.stitched_equity_curve is not None
    values = result.stitched_equity_curve["total_equity"].tolist()
    assert values == pytest.approx([1000.0, 1009.0, 1009.0, 1017.072])

    assert result.out_of_sample_report is not None
    expected_total_return = 1017.072 / 1000.0 - 1.0
    assert result.out_of_sample_report.total_return == pytest.approx(expected_total_return)


def test_walk_forward_no_stitched_curve_when_windows_overlap():
    df = _make_rising_df(10)
    result = walk_forward_validate(
        df,
        param_grid=[{"position": 1}],
        strategy_factory=_strategy_factory,
        train_days=4,
        test_days=3,
        step_days=1,  # overlapping test windows
        cost_model=ZERO_COST,
        initial_capital=1000.0,
    )
    assert len(result.windows) > 0
    assert result.stitched_equity_curve is None
    assert result.out_of_sample_report is None


def test_walk_forward_skips_bad_param_combo_without_crashing():
    df = _make_rising_df(8)
    # position=2 is invalid for the test strategy and always raises;
    # position=1 is valid and should still win.
    result = walk_forward_validate(
        df,
        param_grid=[{"position": 2}, {"position": 1}],
        strategy_factory=_strategy_factory,
        train_days=4,
        test_days=2,
        step_days=2,
        cost_model=ZERO_COST,
    )
    assert all(w.best_params == {"position": 1} for w in result.windows)


def test_summary_table_has_one_row_per_window():
    df = _make_rising_df(8)
    result = walk_forward_validate(
        df,
        param_grid=[{"position": 0}, {"position": 1}],
        strategy_factory=_strategy_factory,
        train_days=4,
        test_days=2,
        step_days=2,
        cost_model=ZERO_COST,
    )
    table = result.summary_table()
    assert len(table) == len(result.windows) == 2
    assert "test_sharpe" in table.columns
