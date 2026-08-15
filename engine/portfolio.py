"""
engine/portfolio.py

Runs a full single-symbol backtest: walks through the OHLCV data bar by
bar, executes trades exactly when engine.execution says they're due (next
bar's open, per the look-ahead bias fix from Phase 3), sizes each trade
from real available cash rather than a fixed quantity, and marks the
portfolio to market every single day -- not just on trade days -- to
produce a daily equity curve.

This module deliberately reuses execution.py's shift_to_next_bar_execution
(trade timing) and calculate_fill (cost math) rather than reimplementing
either. The only thing new here is position sizing and cash/position
bookkeeping over time.

Constraints enforced:
  - No negative cash: buy quantities are sized so gross value + all costs
    never exceed available cash. This is a closed-form calculation (all
    of calculate_fill's cost components are proportional to trade value
    for a BUY), not a hopeful guess checked after the fact.
  - Max position size: `position_size_pct` caps how much of current cash
    goes into a single entry (default 0.95, leaving a buffer rather than
    betting every rupee). For this single-symbol, long-only-by-default
    portfolio, that is the position size cap -- there's only ever one
    position to size.
  - Shorting is rejected by default (`allow_short=False`): NSE cash
    equity delivery doesn't support it without separate margin/borrow
    mechanics this project isn't modeling, and the default strategy
    (MovingAverageCrossover with long_only=True) never emits -1 anyway.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Union

import pandas as pd

from engine.execution import BUY, SELL, CostModel, calculate_fill, shift_to_next_bar_execution

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    final_cash: float
    final_quantity: int

    @property
    def final_equity(self) -> float:
        if self.equity_curve.empty:
            return self.final_cash
        return float(self.equity_curve["total_equity"].iloc[-1])


def _buy_cost_multiplier(cost_model: CostModel) -> float:
    """
    For a BUY, every cost component is proportional to gross_value
    (price * quantity) -- there's no flat fee on the buy side (dp_charges
    only applies to SELL). So total cash required for `qty` shares at
    `price` is exactly price * qty * multiplier, letting us solve for the
    max affordable quantity directly instead of guessing and retrying.
    """
    m = cost_model
    return (
        1
        + m.brokerage_pct
        + m.stt_pct
        + m.exchange_txn_pct
        + m.stamp_duty_pct
        + m.sebi_turnover_pct
        + m.gst_pct * (m.brokerage_pct + m.exchange_txn_pct)
    )


def _max_affordable_buy_quantity(cash: float, price: float, cost_model: CostModel) -> int:
    if cash <= 0 or price <= 0:
        return 0
    multiplier = _buy_cost_multiplier(cost_model)
    qty = int(cash // (price * multiplier))
    # Guard against floating-point rounding pushing the exact-fit quantity
    # just over budget -- cheap to double check since qty is always small.
    while qty > 0 and (price * qty + calculate_fill(price, qty, BUY, cost_model).total_cost) > cash + 1e-9:
        qty -= 1
    return max(qty, 0)


def run_backtest(
    df: pd.DataFrame,
    signal: pd.Series,
    cost_model: CostModel = CostModel(),
    initial_capital: float = 100_000.0,
    position_size_pct: Union[float, pd.Series] = 0.95,
    allow_short: bool = False,
    default_position_size_pct: float = 0.95,
) -> BacktestResult:
    """
    df: OHLCV DataFrame (date, open, high, low, close, volume), sorted
        ascending -- same shape Strategy.run() expects.
    signal: target-position series from a Strategy, aligned to df's index.
    position_size_pct: fraction of available cash committed to a new long
        entry (default 0.95). Applies only to entries -- exits always
        close the full held quantity, since the signal's target is 0.

        Can also be a pd.Series aligned to df's index (same length, same
        positional index) for *dynamic* sizing -- e.g. ml.position_sizing's
        volatility-scaled sizer, which shrinks the fraction committed when
        predicted risk is high and grows it when predicted risk is low,
        instead of always committing the same fixed fraction. Each BUY
        looks up the value at that execution row; a NaN there (e.g. during
        the sizer's own model warm-up) falls back to
        `default_position_size_pct` rather than skipping the trade.

    Returns a BacktestResult with a full trade log and a daily
    (not just trade-day) mark-to-market equity curve.
    """
    if isinstance(position_size_pct, pd.Series):
        if not position_size_pct.index.equals(df.index):
            raise ValueError("position_size_pct Series must share df's index")
    elif not (0 < position_size_pct <= 1):
        raise ValueError(f"position_size_pct must be in (0, 1], got {position_size_pct}")

    if not (0 < default_position_size_pct <= 1):
        raise ValueError(f"default_position_size_pct must be in (0, 1], got {default_position_size_pct}")

    if not allow_short and (signal < 0).any():
        raise ValueError(
            "signal contains short positions (-1) but allow_short=False. "
            "Pass allow_short=True if you intend to model shorting."
        )

    scheduled = shift_to_next_bar_execution(df, signal)
    unexecutable = scheduled[scheduled["execution_date"].isna()]
    for _, row in unexecutable.iterrows():
        logger.warning(
            "Signal change on %s (position %s -> %s) has no next bar to execute at; dropped.",
            row["signal_date"].date(), row["position_before"], row["position_after"],
        )
    scheduled = scheduled.dropna(subset=["execution_date"])
    scheduled_by_date = {d: g for d, g in scheduled.groupby("execution_date")}

    cash = float(initial_capital)
    quantity_held = 0
    trades = []
    equity_rows = []

    for row_idx, day in df.iterrows():
        today = day["date"]

        if today in scheduled_by_date:
            for _, sched in scheduled_by_date[today].iterrows():
                position_before = sched["position_before"]
                position_after = sched["position_after"]
                execution_price = float(day["open"])

                if position_after > position_before:
                    if isinstance(position_size_pct, pd.Series):
                        pct = position_size_pct.loc[row_idx]
                        if pd.isna(pct):
                            logger.warning(
                                "position_size_pct is NaN on %s (row %s); falling back to default %.2f.",
                                today.date(), row_idx, default_position_size_pct,
                            )
                            pct = default_position_size_pct
                    else:
                        pct = position_size_pct
                    cash_to_use = cash * pct
                    qty = _max_affordable_buy_quantity(cash_to_use, execution_price, cost_model)
                    if qty <= 0:
                        logger.warning("Insufficient cash to enter a position on %s (cash=%.2f); skipped.", today.date(), cash)
                        continue
                    fill = calculate_fill(execution_price, qty, BUY, cost_model)
                    cash += fill.net_cash_flow
                    quantity_held += qty
                elif position_after < position_before:
                    qty = quantity_held
                    if qty <= 0:
                        logger.warning("Exit signal on %s but no shares held; skipped.", today.date())
                        continue
                    fill = calculate_fill(execution_price, qty, SELL, cost_model)
                    cash += fill.net_cash_flow
                    quantity_held -= qty
                else:
                    continue

                assert cash >= -1e-6, f"cash went negative on {today}: {cash}"
                trades.append(
                    {
                        "signal_date": sched["signal_date"],
                        "execution_date": today,
                        "side": fill.side,
                        "price": fill.price,
                        "quantity": fill.quantity,
                        "total_cost": fill.total_cost,
                        "net_cash_flow": fill.net_cash_flow,
                        "cash_after": cash,
                        "quantity_held_after": quantity_held,
                    }
                )

        position_value = quantity_held * float(day["close"])
        equity_rows.append(
            {
                "date": today,
                "cash": cash,
                "quantity_held": quantity_held,
                "position_value": position_value,
                "total_equity": cash + position_value,
            }
        )

    return BacktestResult(
        trades=pd.DataFrame(trades),
        equity_curve=pd.DataFrame(equity_rows),
        final_cash=cash,
        final_quantity=quantity_held,
    )
