"""
Unit tests for reporting/metrics.py.

total_return and max_drawdown are checked against hand-computed values
(see the comment above _make_equity_curve). Sharpe ratio is cross-checked
against an independently written formula rather than a hand-derived
number, since verifying floating-point Sharpe arithmetic by hand isn't
practical -- the point of that test is to catch a wrong annualization
factor or a ddof/std mistake, not to re-derive statistics by hand.
"""

from pathlib import Path

import pandas as pd
import pytest

from reporting.metrics import (
    buy_and_hold_benchmark,
    compute_round_trip_pnls,
    generate_report,
    max_drawdown,
    plot_drawdown,
    plot_equity_curve,
    sharpe_ratio,
    total_return,
    turnover,
    win_rate,
)


def _make_equity_curve():
    # total_equity: [1000, 1100, 1050, 1200, 1150]
    #
    # total_return = 1150/1000 - 1 = 0.15
    #
    # max drawdown: running_max = [1000,1100,1100,1200,1200]
    #   drawdown   = [0, 0, 1050/1100-1, 0, 1150/1200-1]
    #              = [0, 0, -0.045454..., 0, -0.041666...]
    #   worst drawdown is idx2 (-0.045454...), peak is idx1 (equity=1100)
    dates = pd.bdate_range("2024-01-01", periods=5)
    return pd.DataFrame({"date": dates, "total_equity": [1000.0, 1100.0, 1050.0, 1200.0, 1150.0]})


def test_total_return_matches_hand_computed_value():
    ec = _make_equity_curve()
    assert total_return(ec) == pytest.approx(0.15)


def test_max_drawdown_matches_hand_computed_value_and_dates():
    ec = _make_equity_curve()
    mdd, peak_date, trough_date = max_drawdown(ec)
    assert mdd == pytest.approx(1050 / 1100 - 1)
    assert peak_date == ec["date"].iloc[1]
    assert trough_date == ec["date"].iloc[2]


def test_max_drawdown_is_zero_for_monotonically_increasing_equity():
    dates = pd.bdate_range("2024-01-01", periods=4)
    ec = pd.DataFrame({"date": dates, "total_equity": [1000.0, 1010.0, 1020.0, 1030.0]})
    mdd, peak_date, trough_date = max_drawdown(ec)
    assert mdd == 0.0
    assert peak_date is None
    assert trough_date is None


def test_sharpe_ratio_matches_independent_calculation():
    ec = _make_equity_curve()
    daily_returns = ec["total_equity"].pct_change().fillna(0.0)
    expected = (daily_returns.mean() / daily_returns.std()) * (252 ** 0.5)
    assert sharpe_ratio(ec, risk_free_rate=0.0) == pytest.approx(expected)


def test_sharpe_ratio_zero_for_constant_equity():
    dates = pd.bdate_range("2024-01-01", periods=4)
    ec = pd.DataFrame({"date": dates, "total_equity": [1000.0] * 4})
    assert sharpe_ratio(ec) == 0.0


def _make_trades():
    # Round trip 1: BUY 10 @ 100 (net -1005), SELL 10 @ 110 (net +1090) -> pnl = +85 (win)
    # Round trip 2: BUY 9 @ 110 (net -995),   SELL 9 @ 100 (net +895)  -> pnl = -100 (loss)
    dates = pd.bdate_range("2024-01-01", periods=4)
    return pd.DataFrame(
        [
            {"execution_date": dates[0], "side": "BUY", "price": 100.0, "quantity": 10, "net_cash_flow": -1005.0},
            {"execution_date": dates[1], "side": "SELL", "price": 110.0, "quantity": 10, "net_cash_flow": 1090.0},
            {"execution_date": dates[2], "side": "BUY", "price": 110.0, "quantity": 9, "net_cash_flow": -995.0},
            {"execution_date": dates[3], "side": "SELL", "price": 100.0, "quantity": 9, "net_cash_flow": 895.0},
        ]
    )


def test_round_trip_pnls_and_win_rate():
    trades = _make_trades()
    pnls = compute_round_trip_pnls(trades)
    assert pnls == pytest.approx([85.0, -100.0])

    wr, n = win_rate(trades)
    assert wr == pytest.approx(0.5)
    assert n == 2


def test_win_rate_none_for_no_trades():
    wr, n = win_rate(pd.DataFrame(columns=["execution_date", "side", "price", "quantity", "net_cash_flow"]))
    assert wr is None
    assert n == 0


def test_turnover_matches_hand_computed_value():
    trades = _make_trades()
    ec = _make_equity_curve()
    # traded_value = 100*10 + 110*10 + 110*9 + 100*9 = 1000+1100+990+900 = 3990
    # avg_equity   = mean([1000,1100,1050,1200,1150]) = 1100
    # n_years      = 5 / 252
    expected = (3990 / 1100) / (5 / 252)
    assert turnover(trades, ec) == pytest.approx(expected)


def test_turnover_zero_for_no_trades():
    ec = _make_equity_curve()
    assert turnover(pd.DataFrame(columns=["price", "quantity"]), ec) == 0.0


def test_buy_and_hold_benchmark_tracks_price_ratio():
    dates = pd.bdate_range("2024-01-01", periods=3)
    df = pd.DataFrame({"date": dates, "close": [100.0, 110.0, 90.0]})
    benchmark = buy_and_hold_benchmark(df, initial_capital=1000.0)
    assert benchmark["total_equity"].tolist() == pytest.approx([1000.0, 1100.0, 900.0])


def test_generate_report_runs_end_to_end():
    ec = _make_equity_curve()
    trades = _make_trades()
    df = pd.DataFrame({"date": ec["date"], "close": [100.0, 108.0, 103.0, 118.0, 113.0]})

    report = generate_report(ec, trades, df, initial_capital=1000.0)

    assert report.total_return == pytest.approx(0.15)
    assert report.win_rate == pytest.approx(0.5)
    assert report.n_round_trips == 2
    assert isinstance(report.summary(), str)
    assert "Sharpe" in report.summary()


def test_plot_functions_produce_files(tmp_path):
    ec = _make_equity_curve()
    df = pd.DataFrame({"date": ec["date"], "close": [100.0, 108.0, 103.0, 118.0, 113.0]})

    equity_png = plot_equity_curve(ec, df, initial_capital=1000.0, output_path=tmp_path / "equity.png")
    drawdown_png = plot_drawdown(ec, output_path=tmp_path / "drawdown.png")

    assert Path(equity_png).exists() and Path(equity_png).stat().st_size > 0
    assert Path(drawdown_png).exists() and Path(drawdown_png).stat().st_size > 0
