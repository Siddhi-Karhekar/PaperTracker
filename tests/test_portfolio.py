"""
Unit tests for engine/portfolio.py.

The main scenario uses a zero-cost CostModel so cash/quantity/equity at
every single day can be verified by hand (see the comment above
test_run_backtest_zero_cost_round_trip for the arithmetic) -- this
isolates portfolio bookkeeping from cost-model math, which is already
covered separately in tests/test_execution.py.
"""

import pandas as pd
import pytest

from engine.execution import BUY, SELL, CostModel
from engine.portfolio import run_backtest

ZERO_COST = CostModel(
    brokerage_pct=0.0,
    stt_pct=0.0,
    exchange_txn_pct=0.0,
    stamp_duty_pct=0.0,
    sebi_turnover_pct=0.0,
    gst_pct=0.0,
    dp_charges_flat=0.0,
)


def _make_df(n=7):
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


def test_run_backtest_zero_cost_round_trip():
    # opens: [100,102,104,106,108,110,112]  closes: [101,103,105,107,109,111,113]
    # signal: [0,0,1,1,0,0,0] -> enter at idx2 (executes idx3 open=106),
    #                            exit  at idx4 (executes idx5 open=110)
    # initial_capital=1000, position_size_pct=1.0, zero-cost model:
    #
    # day0-2: no position yet -> cash=1000, qty=0, equity=1000
    # day3:  BUY qty = floor(1000/106) = 9 -> cash = 1000 - 9*106 = 46
    #        mark-to-market at close[3]=107 -> equity = 46 + 9*107 = 1009
    # day4:  no trade yet (exit signal fires at idx4, executes next bar idx5)
    #        cash=46, qty=9, close[4]=109 -> equity = 46 + 9*109 = 1027
    # day5:  SELL all 9 @ open[5]=110 -> cash = 46 + 9*110 = 1036, qty=0
    #        close[5]=111 -> equity = 1036 + 0 = 1036
    # day6:  no trade -> cash=1036, qty=0, equity=1036
    df = _make_df(7)
    signal = pd.Series([0, 0, 1, 1, 0, 0, 0], index=df.index)

    result = run_backtest(df, signal, cost_model=ZERO_COST, initial_capital=1000.0, position_size_pct=1.0)

    expected_cash = [1000, 1000, 1000, 46, 46, 1036, 1036]
    expected_qty = [0, 0, 0, 9, 9, 0, 0]
    expected_equity = [1000, 1000, 1000, 1009, 1027, 1036, 1036]

    assert result.equity_curve["cash"].tolist() == pytest.approx(expected_cash)
    assert result.equity_curve["quantity_held"].tolist() == expected_qty
    assert result.equity_curve["total_equity"].tolist() == pytest.approx(expected_equity)
    assert result.final_cash == pytest.approx(1036.0)
    assert result.final_quantity == 0
    assert result.final_equity == pytest.approx(1036.0)

    assert len(result.trades) == 2
    buy, sell = result.trades.iloc[0], result.trades.iloc[1]
    assert buy["side"] == BUY and buy["quantity"] == 9 and buy["price"] == pytest.approx(106.0)
    assert sell["side"] == SELL and sell["quantity"] == 9 and sell["price"] == pytest.approx(110.0)


def test_run_backtest_drops_exit_signal_on_last_bar():
    # Same setup as engine/execution.py's equivalent test: an exit signal
    # change landing on the final bar has no next-bar open to execute at,
    # so the position stays open through the end of the data.
    df = _make_df(5)
    signal = pd.Series([0, 0, 1, 1, 0], index=df.index)

    result = run_backtest(df, signal, cost_model=ZERO_COST, initial_capital=1000.0, position_size_pct=1.0)

    assert len(result.trades) == 1  # only the entry; the exit was dropped
    assert result.final_quantity == 9  # still holding


def test_run_backtest_cash_never_negative_with_default_cost_model():
    df = _make_df(7)
    signal = pd.Series([0, 0, 1, 1, 0, 0, 0], index=df.index)

    result = run_backtest(df, signal, cost_model=CostModel(), initial_capital=1000.0, position_size_pct=1.0)

    assert (result.equity_curve["cash"] >= 0).all()


def test_run_backtest_rejects_short_signal_by_default():
    df = _make_df(5)
    signal = pd.Series([0, -1, -1, -1, 0], index=df.index)

    with pytest.raises(ValueError, match="allow_short"):
        run_backtest(df, signal, cost_model=ZERO_COST)


def test_run_backtest_rejects_invalid_position_size_pct():
    df = _make_df(5)
    signal = pd.Series([0, 0, 0, 0, 0], index=df.index)

    with pytest.raises(ValueError):
        run_backtest(df, signal, position_size_pct=0)
    with pytest.raises(ValueError):
        run_backtest(df, signal, position_size_pct=1.5)


def test_run_backtest_no_signal_changes_flat_equity_curve():
    df = _make_df(5)
    signal = pd.Series([0, 0, 0, 0, 0], index=df.index)

    result = run_backtest(df, signal, cost_model=ZERO_COST, initial_capital=1000.0)

    assert result.trades.empty
    assert (result.equity_curve["total_equity"] == 1000.0).all()


def test_run_backtest_accepts_dynamic_position_size_pct_series():
    # Same entry as the round-trip test (enters at idx3, open=106), but
    # sized with a per-row Series instead of a fixed fraction: at the
    # execution row (idx3), pct=0.5 -> qty = floor(1000*0.5/106) = floor(4.71..) = 4
    # (versus qty=9 with the fixed 1.0 used in the round-trip test above).
    df = _make_df(7)
    signal = pd.Series([0, 0, 1, 1, 0, 0, 0], index=df.index)
    sizing = pd.Series([1.0] * 7, index=df.index)
    sizing.loc[3] = 0.5

    result = run_backtest(df, signal, cost_model=ZERO_COST, initial_capital=1000.0, position_size_pct=sizing)

    assert len(result.trades) == 2
    buy = result.trades.iloc[0]
    assert buy["quantity"] == 4
    assert buy["price"] == pytest.approx(106.0)


def test_run_backtest_dynamic_sizing_falls_back_to_default_on_nan():
    # Same setup, but the sizing Series is NaN at the execution row -> falls
    # back to default_position_size_pct=0.3: qty = floor(1000*0.3/106) = floor(2.83) = 2
    df = _make_df(7)
    signal = pd.Series([0, 0, 1, 1, 0, 0, 0], index=df.index)
    sizing = pd.Series([float("nan")] * 7, index=df.index)

    result = run_backtest(
        df, signal, cost_model=ZERO_COST, initial_capital=1000.0,
        position_size_pct=sizing, default_position_size_pct=0.3,
    )

    assert len(result.trades) == 2
    buy = result.trades.iloc[0]
    assert buy["quantity"] == 2


def test_run_backtest_rejects_position_size_pct_series_with_mismatched_index():
    df = _make_df(5)
    signal = pd.Series([0, 0, 0, 0, 0], index=df.index)
    bad_sizing = pd.Series([0.5, 0.5, 0.5], index=[0, 1, 2])  # wrong length/index

    with pytest.raises(ValueError, match="index"):
        run_backtest(df, signal, position_size_pct=bad_sizing)
