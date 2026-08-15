"""
reporting/metrics.py

Performance metrics and charts for a completed backtest.

Takes the `equity_curve` and `trades` DataFrames produced by
engine.portfolio.run_backtest(), computes the standard set of
risk-adjusted performance numbers, and compares against a buy-and-hold
benchmark -- a strategy that only beats holding the stock outright isn't
saying much, so that comparison is always included rather than optional.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


@dataclass
class PerformanceReport:
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_start: Optional[pd.Timestamp]
    max_drawdown_end: Optional[pd.Timestamp]
    win_rate: Optional[float]
    n_round_trips: int
    turnover: float
    benchmark_total_return: float
    benchmark_annualized_return: float

    def summary(self) -> str:
        wr = "n/a (no completed round trips)" if self.win_rate is None else f"{self.win_rate:.1%} ({self.n_round_trips} round trips)"
        lines = [
            "Performance report",
            f"  Total return:            {self.total_return:+.2%}",
            f"  Annualized return:       {self.annualized_return:+.2%}",
            f"  Sharpe ratio:            {self.sharpe_ratio:.2f}",
            f"  Max drawdown:            {self.max_drawdown:.2%}",
            f"  Win rate:                {wr}",
            f"  Turnover (annualized):   {self.turnover:.2f}x",
            f"  Buy-and-hold return:     {self.benchmark_total_return:+.2%} (annualized {self.benchmark_annualized_return:+.2%})",
        ]
        return "\n".join(lines)


def compute_daily_returns(equity_curve: pd.DataFrame) -> pd.Series:
    return equity_curve["total_equity"].pct_change().fillna(0.0)


def total_return(equity_curve: pd.DataFrame) -> float:
    if equity_curve.empty:
        return 0.0
    start = equity_curve["total_equity"].iloc[0]
    end = equity_curve["total_equity"].iloc[-1]
    return float(end / start - 1.0)


def annualized_return(equity_curve: pd.DataFrame) -> float:
    n_days = len(equity_curve)
    if n_days < 2:
        return 0.0
    tr = total_return(equity_curve)
    years = n_days / TRADING_DAYS_PER_YEAR
    return float((1 + tr) ** (1 / years) - 1)


def sharpe_ratio(equity_curve: pd.DataFrame, risk_free_rate: float = 0.0) -> float:
    """
    Annualized Sharpe ratio from daily returns. risk_free_rate is an
    annual rate, converted to a daily rate for the excess-return
    calculation. Returns 0.0 for degenerate inputs (fewer than 2 days, or
    zero return volatility) rather than raising or returning NaN/inf.
    """
    daily_returns = compute_daily_returns(equity_curve)
    if len(daily_returns) < 2 or daily_returns.std() == 0:
        return 0.0
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = daily_returns - daily_rf
    return float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(equity_curve: pd.DataFrame) -> tuple:
    """
    Returns (max_drawdown, peak_date, trough_date). max_drawdown is
    negative (or 0.0 if equity never dips below a prior peak); peak/trough
    dates are None when there's no drawdown to report.
    """
    if equity_curve.empty:
        return 0.0, None, None

    equity = equity_curve["total_equity"]
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0

    trough_idx = drawdown.idxmin()
    mdd = float(drawdown.loc[trough_idx])
    if mdd == 0.0:
        return 0.0, None, None

    peak_idx = equity.loc[:trough_idx].idxmax()
    return mdd, equity_curve.loc[peak_idx, "date"], equity_curve.loc[trough_idx, "date"]


def compute_round_trip_pnls(trades: pd.DataFrame) -> list:
    """
    Pairs each BUY with the SELL that closes it. Assumes sequential,
    non-overlapping single-symbol trades -- which is what
    engine.portfolio.run_backtest() produces (long-only by default: buy,
    then later sell the whole position, repeat). Returns realized P&L per
    round trip, net of all transaction costs, since net_cash_flow already
    includes them.
    """
    if trades.empty:
        return []
    pnls = []
    open_buy = None
    for _, trade in trades.sort_values("execution_date").iterrows():
        if trade["side"] == "BUY":
            open_buy = trade
        elif trade["side"] == "SELL" and open_buy is not None:
            pnls.append(float(trade["net_cash_flow"] + open_buy["net_cash_flow"]))
            open_buy = None
    return pnls


def win_rate(trades: pd.DataFrame) -> tuple:
    """Returns (win_rate, n_round_trips); win_rate is None if there are no completed round trips."""
    pnls = compute_round_trip_pnls(trades)
    if not pnls:
        return None, 0
    wins = sum(1 for p in pnls if p > 0)
    return wins / len(pnls), len(pnls)


def turnover(trades: pd.DataFrame, equity_curve: pd.DataFrame) -> float:
    """
    Annualized turnover: total traded value (both buy and sell legs)
    relative to average portfolio equity, scaled to a 1-year period. A
    turnover of 2.0 means the portfolio's full value was traded over, on
    average, twice per year -- higher turnover means higher total costs
    for the same signal quality.
    """
    if trades.empty or equity_curve.empty:
        return 0.0
    traded_value = float((trades["price"] * trades["quantity"]).sum())
    avg_equity = float(equity_curve["total_equity"].mean())
    n_years = len(equity_curve) / TRADING_DAYS_PER_YEAR
    if avg_equity == 0 or n_years == 0:
        return 0.0
    return (traded_value / avg_equity) / n_years


def buy_and_hold_benchmark(df: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    """
    A naive buy-and-hold benchmark: fully invested from day 1 at the
    first close, no transaction costs, no fractional-share rounding.
    This is a deliberate simplification for a fair-ish comparison point --
    it is not a second backtest run through the execution/cost pipeline.
    """
    first_close = df["close"].iloc[0]
    equity = initial_capital * (df["close"] / first_close)
    return pd.DataFrame({"date": df["date"], "total_equity": equity})


def generate_report(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    df: pd.DataFrame,
    initial_capital: float,
    risk_free_rate: float = 0.0,
) -> PerformanceReport:
    mdd, mdd_start, mdd_end = max_drawdown(equity_curve)
    wr, n_rt = win_rate(trades)
    benchmark = buy_and_hold_benchmark(df, initial_capital)

    return PerformanceReport(
        total_return=total_return(equity_curve),
        annualized_return=annualized_return(equity_curve),
        sharpe_ratio=sharpe_ratio(equity_curve, risk_free_rate),
        max_drawdown=mdd,
        max_drawdown_start=mdd_start,
        max_drawdown_end=mdd_end,
        win_rate=wr,
        n_round_trips=n_rt,
        turnover=turnover(trades, equity_curve),
        benchmark_total_return=total_return(benchmark),
        benchmark_annualized_return=annualized_return(benchmark),
    )


def plot_equity_curve(equity_curve: pd.DataFrame, df: pd.DataFrame, initial_capital: float, output_path) -> Path:
    """Saves a PNG comparing the strategy's equity curve to buy-and-hold. Returns the output path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    benchmark = buy_and_hold_benchmark(df, initial_capital)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(equity_curve["date"], equity_curve["total_equity"], label="Strategy")
    ax.plot(benchmark["date"], benchmark["total_equity"], label="Buy & hold", linestyle="--")
    ax.set_title("Equity curve vs. buy-and-hold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio value")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_drawdown(equity_curve: pd.DataFrame, output_path) -> Path:
    """Saves a PNG of the strategy's drawdown over time. Returns the output path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    equity = equity_curve["total_equity"]
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.fill_between(equity_curve["date"], drawdown, 0, color="firebrick", alpha=0.5)
    ax.set_title("Drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    fig.autofmt_xdate()
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
