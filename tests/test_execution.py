"""
Unit tests for engine/execution.py.

Two things are checked most carefully here, because they're the actual
point of this module:
  1. calculate_fill()'s cost breakdown, hand-computed against the default
     CostModel (see comments below).
  2. simulate_execution() executes at the NEXT bar's open, never the
     signal bar's own close -- the look-ahead bias fix -- and drops signal
     changes that land on the last bar (no future bar to execute at).
"""

import pandas as pd
import pytest

from engine.execution import BUY, SELL, CostModel, calculate_fill, simulate_execution


def test_calculate_fill_buy_matches_hand_computed_breakdown():
    # price=100, qty=10, default CostModel:
    #   gross            = 1000
    #   brokerage        = 0                          (default 0%)
    #   stt              = 1000 * 0.001    = 1.0
    #   exchange_txn     = 1000 * 0.0000297 = 0.0297
    #   stamp_duty       = 1000 * 0.00015  = 0.15      (buy side only)
    #   sebi_charge      = 1000 * 0.000001 = 0.001
    #   gst              = (0 + 0.0297) * 0.18 = 0.005346
    #   dp_charges       = 0                          (buy side: no DP charge)
    #   total_cost       = 1.0 + 0.0297 + 0.15 + 0.001 + 0.005346 = 1.186046
    #   net_cash_flow    = -(1000 + 1.186046) = -1001.186046
    fill = calculate_fill(price=100, quantity=10, side=BUY)
    assert fill.gross_value == pytest.approx(1000.0)
    assert fill.stt == pytest.approx(1.0)
    assert fill.exchange_txn_charge == pytest.approx(0.0297)
    assert fill.stamp_duty == pytest.approx(0.15)
    assert fill.dp_charges == pytest.approx(0.0)
    assert fill.total_cost == pytest.approx(1.186046, rel=1e-6)
    assert fill.net_cash_flow == pytest.approx(-1001.186046, rel=1e-6)


def test_calculate_fill_sell_matches_hand_computed_breakdown():
    # Same trade, SELL side: no stamp duty, but DP charge (flat 15) applies.
    #   total_cost    = 1.0 + 0.0297 + 0 + 0.001 + 0.005346 + 15.0 = 16.036046
    #   net_cash_flow = 1000 - 16.036046 = 983.963954
    fill = calculate_fill(price=100, quantity=10, side=SELL)
    assert fill.stamp_duty == pytest.approx(0.0)
    assert fill.dp_charges == pytest.approx(15.0)
    assert fill.total_cost == pytest.approx(16.036046, rel=1e-6)
    assert fill.net_cash_flow == pytest.approx(983.963954, rel=1e-6)


def test_calculate_fill_rejects_invalid_side():
    with pytest.raises(ValueError):
        calculate_fill(price=100, quantity=10, side="HOLD")


def test_calculate_fill_rejects_non_positive_quantity_or_price():
    with pytest.raises(ValueError):
        calculate_fill(price=100, quantity=0, side=BUY)
    with pytest.raises(ValueError):
        calculate_fill(price=0, quantity=10, side=BUY)


def _make_df():
    # Opens and closes are deliberately different so a test that
    # accidentally executes at the signal bar's close (or the wrong bar's
    # open) fails loudly instead of silently passing.
    dates = pd.bdate_range("2024-01-01", periods=5)
    return pd.DataFrame(
        {
            "date": dates,
            "open": [100, 102, 104, 106, 108],
            "high": [101, 103, 105, 107, 109],
            "low": [99, 101, 103, 105, 107],
            "close": [101, 103, 105, 107, 109],
            "volume": [1000] * 5,
        }
    )


def test_simulate_execution_fills_at_next_bar_open_not_signal_bar_close():
    df = _make_df()
    # position: flat, flat, LONG, LONG, flat
    # -> enters at idx2 (change +1), exits at idx4 (change -1, but idx4 is
    #    the last bar, so that exit has no next bar and must be dropped)
    signal = pd.Series([0, 0, 1, 1, 0], index=df.index)

    trades = simulate_execution(df, signal, quantity_per_unit=10)

    assert len(trades) == 1  # the exit at idx4 is dropped, only entry survives
    entry = trades.iloc[0]
    assert entry["side"] == BUY
    assert entry["signal_date"] == df.loc[2, "date"]
    # Executed at idx3's open (106), NOT idx2's close (105) and NOT idx2's open (104).
    assert entry["execution_date"] == df.loc[3, "date"]
    assert entry["price"] == pytest.approx(106.0)
    assert entry["quantity"] == 10


def test_simulate_execution_drops_signal_change_on_last_bar():
    df = _make_df()
    signal = pd.Series([0, 0, 0, 0, 1], index=df.index)  # change only on the final bar

    trades = simulate_execution(df, signal, quantity_per_unit=10)

    assert trades.empty


def test_simulate_execution_no_changes_returns_empty_frame():
    df = _make_df()
    signal = pd.Series([0, 0, 0, 0, 0], index=df.index)

    trades = simulate_execution(df, signal, quantity_per_unit=10)

    assert trades.empty
    assert list(trades.columns)  # still has the expected schema, just no rows


def test_simulate_execution_applies_cost_model_end_to_end():
    df = _make_df()
    signal = pd.Series([0, 0, 1, 1, 1], index=df.index)  # enter at idx2, hold to the end, no exit

    trades = simulate_execution(df, signal, quantity_per_unit=10)

    assert len(trades) == 1
    row = trades.iloc[0]
    # price=106, qty=10 -> gross=1060, matches calculate_fill's own math
    expected = calculate_fill(price=106.0, quantity=10, side=BUY, cost_model=CostModel())
    assert row["total_cost"] == pytest.approx(expected.total_cost, rel=1e-6)
    assert row["net_cash_flow"] == pytest.approx(expected.net_cash_flow, rel=1e-6)


def test_simulate_execution_handles_position_flip_as_double_quantity():
    df = _make_df()
    signal = pd.Series([0, 1, 1, -1, -1], index=df.index)  # long at idx1, flips to short at idx3

    trades = simulate_execution(df, signal, quantity_per_unit=10)

    assert len(trades) == 2
    flip = trades.iloc[1]
    # change at idx3 = -1 - (+1) = -2 -> quantity = 2 * quantity_per_unit
    assert flip["side"] == SELL
    assert flip["quantity"] == 20
