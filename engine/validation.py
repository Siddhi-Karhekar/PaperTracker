"""
engine/validation.py

Walk-forward validation: split history into a sequence of rolling
train/test windows, pick strategy parameters using ONLY each window's
train segment (via a grid search scored by a chosen metric), then
evaluate that choice on the following, unseen test segment. Rolling this
forward through the full history and stitching the out-of-sample test
segments together answers "how would this strategy actually have
performed if I'd been re-tuning it periodically and only ever trading on
data I hadn't seen yet?" -- a fundamentally different, more honest
question than "how well does one fixed parameter set fit the whole
history at once," which is how most student backtests get quietly
overfit.

This is what directly guards against picking a moving-average window (or
any other parameter) because it happened to fit one lucky stretch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

from engine.execution import CostModel
from engine.portfolio import run_backtest
from reporting.metrics import PerformanceReport, generate_report
from strategy.base import Strategy

logger = logging.getLogger(__name__)


@dataclass
class WindowResult:
    window_index: int
    train_start_date: pd.Timestamp
    train_end_date: pd.Timestamp
    test_start_date: pd.Timestamp
    test_end_date: pd.Timestamp
    best_params: dict
    train_metric: float
    test_report: PerformanceReport
    test_equity_curve: pd.DataFrame
    test_trades: pd.DataFrame


@dataclass
class WalkForwardResult:
    windows: list
    stitched_equity_curve: Optional[pd.DataFrame]
    out_of_sample_report: Optional[PerformanceReport]

    def summary_table(self) -> pd.DataFrame:
        """One row per window: dates, chosen params, train score, and the out-of-sample metrics that actually matter."""
        rows = [
            {
                "window": w.window_index,
                "train_start": w.train_start_date,
                "train_end": w.train_end_date,
                "test_start": w.test_start_date,
                "test_end": w.test_end_date,
                "best_params": w.best_params,
                "train_metric": w.train_metric,
                "test_total_return": w.test_report.total_return,
                "test_sharpe": w.test_report.sharpe_ratio,
                "test_max_drawdown": w.test_report.max_drawdown,
            }
            for w in self.windows
        ]
        return pd.DataFrame(rows)


def _generate_windows(n_rows: int, train_days: int, test_days: int, step_days: int) -> list:
    windows = []
    train_start = 0
    while True:
        train_end = train_start + train_days
        test_start = train_end
        test_end = test_start + test_days
        if test_end > n_rows:
            break
        windows.append((train_start, train_end, test_start, test_end))
        train_start += step_days
    return windows


def walk_forward_validate(
    df: pd.DataFrame,
    param_grid: list,
    strategy_factory: Callable[..., Strategy],
    train_days: int,
    test_days: int,
    step_days: Optional[int] = None,
    warmup_days: int = 0,
    cost_model: CostModel = CostModel(),
    initial_capital: float = 100_000.0,
    position_size_pct: float = 0.95,
    selection_metric: str = "sharpe_ratio",
) -> WalkForwardResult:
    """
    df: full OHLCV history, sorted ascending.
    param_grid: list of kwarg dicts to try each window, e.g.
        [{"fast_window": 20, "slow_window": 100}, {"fast_window": 50, "slow_window": 200}]
    strategy_factory: builds a Strategy from one param dict, e.g.
        lambda **p: MovingAverageCrossover(**p)
    train_days / test_days: window sizes in trading days (bars).
    step_days: how far to roll forward between windows; defaults to
        test_days (non-overlapping test windows). The stitched equity
        curve is only built when step_days == test_days -- overlapping
        test windows can't be concatenated into one continuous path, so
        it's left as None rather than silently double-counting dates.
    warmup_days: extra bars of history included before each test window
        purely so indicators (e.g. a slow moving average) have real data
        to warm up on -- set this to your largest lookback (e.g.
        slow_window) so the first bars of the test window aren't stuck
        at a forced-flat signal. Warm-up bars are never traded or scored;
        they're trimmed off before the test window is evaluated.
    selection_metric: PerformanceReport attribute used to rank candidate
        params on the train window (e.g. "sharpe_ratio", "total_return").
        A param combo that errors on the train window (e.g. an invalid
        fast/slow pairing) is skipped with a warning, not fatal.

    Returns a WalkForwardResult. stitched_equity_curve chains each
    window's out-of-sample path onto the previous window's ending value
    (every window's own backtest restarts fresh at initial_capital, so
    this rescaling is what turns them into one continuous compounding
    curve). out_of_sample_report is computed from that stitched curve, so
    total/annualized return, Sharpe, and max drawdown are all exact.
    win_rate is also exact (it only depends on the sign of each round
    trip's P&L, which scale doesn't affect). turnover is approximate: it
    pools each window's raw trade notional against the rescaled stitched
    equity without rescaling the trades themselves.
    """
    if step_days is None:
        step_days = test_days

    windows = _generate_windows(len(df), train_days, test_days, step_days)
    if not windows:
        raise ValueError(
            f"No windows fit: need at least train_days + test_days = {train_days + test_days} rows, got {len(df)}"
        )

    results = []
    for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
        train_df = df.iloc[train_start:train_end].reset_index(drop=True)

        best_params, best_metric = None, float("-inf")
        for params in param_grid:
            try:
                strategy = strategy_factory(**params)
                signal = strategy.run(train_df)
                train_result = run_backtest(train_df, signal, cost_model, initial_capital, position_size_pct)
                train_report = generate_report(train_result.equity_curve, train_result.trades, train_df, initial_capital)
                metric = getattr(train_report, selection_metric)
            except Exception as exc:  # noqa: BLE001 - one bad param combo shouldn't kill the whole run
                logger.warning("Window %d: params %s failed on train window (%s), skipping", i, params, exc)
                continue

            if metric is not None and metric == metric and metric > best_metric:  # `metric == metric` excludes NaN
                best_metric, best_params = metric, params

        if best_params is None:
            logger.warning("Window %d: no param combo produced a usable train result, skipping window", i)
            continue

        eval_start = max(0, test_start - warmup_days)
        eval_df = df.iloc[eval_start:test_end].reset_index(drop=True)
        test_strategy = strategy_factory(**best_params)
        full_signal = test_strategy.run(eval_df)

        test_len = test_end - test_start
        test_df = eval_df.iloc[-test_len:].reset_index(drop=True)
        test_signal = full_signal.iloc[-test_len:].reset_index(drop=True)

        test_result = run_backtest(test_df, test_signal, cost_model, initial_capital, position_size_pct)
        test_report = generate_report(test_result.equity_curve, test_result.trades, test_df, initial_capital)

        results.append(
            WindowResult(
                window_index=i,
                train_start_date=train_df["date"].iloc[0],
                train_end_date=train_df["date"].iloc[-1],
                test_start_date=test_df["date"].iloc[0],
                test_end_date=test_df["date"].iloc[-1],
                best_params=best_params,
                train_metric=best_metric,
                test_report=test_report,
                test_equity_curve=test_result.equity_curve,
                test_trades=test_result.trades,
            )
        )

    if not results:
        raise ValueError("No window produced a usable result; check param_grid, train_days, and test_days.")

    stitched = None
    oos_report = None
    if step_days == test_days:
        stitched = _stitch_equity_curves(results, initial_capital)
        non_empty_trades = [r.test_trades for r in results if not r.test_trades.empty]
        pooled_trades = (
            pd.concat(non_empty_trades, ignore_index=True)
            if non_empty_trades
            else pd.DataFrame(columns=["price", "quantity", "side", "net_cash_flow", "execution_date"])
        )
        oos_price_df = stitched[["date"]].merge(df[["date", "close"]], on="date", how="left")
        oos_report = generate_report(stitched, pooled_trades, oos_price_df, initial_capital)
    else:
        logger.info("step_days (%d) != test_days (%d): skipping stitched equity curve (windows overlap).", step_days, test_days)

    return WalkForwardResult(windows=results, stitched_equity_curve=stitched, out_of_sample_report=oos_report)


def _stitch_equity_curves(results: list, initial_capital: float) -> pd.DataFrame:
    """
    Chains each window's out-of-sample equity curve onto the previous
    window's ending value, since every window's own backtest restarted
    fresh at `initial_capital`. Produces one continuous compounding
    equity path made entirely of out-of-sample returns.
    """
    running_capital = initial_capital
    segments = []
    for r in results:
        multiplier = r.test_equity_curve["total_equity"] / initial_capital
        segment = pd.DataFrame({"date": r.test_equity_curve["date"], "total_equity": running_capital * multiplier})
        segments.append(segment)
        running_capital = float(segment["total_equity"].iloc[-1])
    return pd.concat(segments, ignore_index=True)
