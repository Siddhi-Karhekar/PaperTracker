"""
engine/execution.py

Turns a strategy's target-position signal into realistic filled trades:
executed at the next bar's open (not the signal bar's close), with a
configurable transaction-cost model applied to each fill.

Two things this module deliberately does NOT do, by design (they belong to
engine/portfolio.py in Phase 4):
  - Track running cash or enforce "don't go negative" constraints.
  - Decide position sizing from account equity.
`simulate_execution()` below uses a simple fixed-quantity-per-signal-unit
sizing model so it's runnable and testable standalone. Phase 4's Portfolio
is expected to call the lower-level `calculate_fill()` directly with
real, cash-aware quantities -- so the cost math lives in exactly one place
and both phases stay consistent.

The single most important thing in this file is the next-bar-open
execution timing in `_shift_to_next_bar_execution()`. Executing a signal
computed from bar t's close AT bar t's close is the classic backtest bug
(look-ahead bias): it assumes you could trade on information -- that day's
own closing price -- at the exact moment it became available. Shifting
execution to bar t+1's open is the standard, simple fix.

Cost model note: the default rates below are illustrative, approximate
figures for NSE retail delivery/CNC equity trades (STT, exchange
transaction charges, stamp duty, GST, DP charges) as commonly cited around
when this project was built. Real rates change and vary by broker -- treat
CostModel as a configurable approximation for backtesting purposes, not as
tax/compliance guidance, and update the defaults if you cite exact numbers
in your README or an interview.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

BUY = "BUY"
SELL = "SELL"


@dataclass(frozen=True)
class CostModel:
    brokerage_pct: float = 0.0        # many discount brokers charge 0% on delivery/CNC equity trades
    stt_pct: float = 0.001            # Securities Transaction Tax, ~0.1% -- applied on both buy and sell for delivery equity
    exchange_txn_pct: float = 0.0000297  # NSE transaction charges, approx
    stamp_duty_pct: float = 0.00015   # approx 0.015%, buy side only (India, delivery equity)
    sebi_turnover_pct: float = 0.000001  # SEBI turnover fee, negligible but included for completeness
    gst_pct: float = 0.18             # GST on (brokerage + exchange transaction charges)
    dp_charges_flat: float = 15.0     # flat per-scrip charge on the sell side (approx, varies by depository participant)


@dataclass(frozen=True)
class Fill:
    price: float
    quantity: int
    side: str
    gross_value: float
    brokerage: float
    stt: float
    exchange_txn_charge: float
    stamp_duty: float
    sebi_charge: float
    gst: float
    dp_charges: float
    total_cost: float
    net_cash_flow: float  # negative = cash out (buy), positive = cash in (sell), after costs


def calculate_fill(price: float, quantity: int, side: str, cost_model: CostModel = CostModel()) -> Fill:
    """
    Apply the cost model to a single trade and return the full breakdown.
    This is the one place transaction-cost math happens -- Phase 4's
    Portfolio should call this directly rather than reimplementing it.
    """
    if side not in (BUY, SELL):
        raise ValueError(f"side must be {BUY!r} or {SELL!r}, got {side!r}")
    if quantity <= 0:
        raise ValueError(f"quantity must be positive, got {quantity}")
    if price <= 0:
        raise ValueError(f"price must be positive, got {price}")

    gross_value = price * quantity
    brokerage = gross_value * cost_model.brokerage_pct
    stt = gross_value * cost_model.stt_pct
    exchange_txn_charge = gross_value * cost_model.exchange_txn_pct
    stamp_duty = gross_value * cost_model.stamp_duty_pct if side == BUY else 0.0
    sebi_charge = gross_value * cost_model.sebi_turnover_pct
    gst = (brokerage + exchange_txn_charge) * cost_model.gst_pct
    dp_charges = cost_model.dp_charges_flat if side == SELL else 0.0

    total_cost = brokerage + stt + exchange_txn_charge + stamp_duty + sebi_charge + gst + dp_charges
    net_cash_flow = -(gross_value + total_cost) if side == BUY else (gross_value - total_cost)

    return Fill(
        price=price,
        quantity=quantity,
        side=side,
        gross_value=gross_value,
        brokerage=brokerage,
        stt=stt,
        exchange_txn_charge=exchange_txn_charge,
        stamp_duty=stamp_duty,
        sebi_charge=sebi_charge,
        gst=gst,
        dp_charges=dp_charges,
        total_cost=total_cost,
        net_cash_flow=net_cash_flow,
    )


def shift_to_next_bar_execution(df: pd.DataFrame, signal: pd.Series) -> pd.DataFrame:
    """
    Find every bar where the target position changes, and map it to an
    execution at the *next* bar's open. This is the look-ahead bias fix:
    a signal known at bar t's close cannot be acted on until bar t+1.

    A position change on the very last bar has no next bar to execute at
    -- it's dropped, with the row's `execution_date` left as NaT, since we
    genuinely cannot know what tomorrow's open would have been.
    """
    assumed_prior_position = signal.shift(1, fill_value=0)
    position_change = signal - assumed_prior_position

    changes = position_change[position_change != 0]
    if changes.empty:
        return pd.DataFrame(
            columns=["signal_date", "execution_date", "execution_price", "position_before", "position_after", "position_change"]
        )

    rows = []
    for idx in changes.index:
        next_idx = idx + 1
        signal_date = df.loc[idx, "date"]
        position_before = assumed_prior_position.loc[idx]
        position_after = signal.loc[idx]
        change = position_change.loc[idx]

        if next_idx >= len(df):
            rows.append(
                {
                    "signal_date": signal_date,
                    "execution_date": pd.NaT,
                    "execution_price": None,
                    "position_before": position_before,
                    "position_after": position_after,
                    "position_change": change,
                }
            )
            continue

        rows.append(
            {
                "signal_date": signal_date,
                "execution_date": df.loc[next_idx, "date"],
                "execution_price": df.loc[next_idx, "open"],
                "position_before": position_before,
                "position_after": position_after,
                "position_change": change,
            }
        )

    return pd.DataFrame(rows)


def simulate_execution(
    df: pd.DataFrame,
    signal: pd.Series,
    cost_model: CostModel = CostModel(),
    quantity_per_unit: int = 1,
) -> pd.DataFrame:
    """
    Convert a strategy's signal series into a DataFrame of realistic filled
    trades. Each unit of |position_change| is sized as `quantity_per_unit`
    shares -- e.g. a flip from -1 to +1 trades 2 * quantity_per_unit shares
    in one BUY fill. This fixed sizing is a simplification for this phase;
    Phase 4's Portfolio replaces it with real, cash-constrained sizing by
    calling calculate_fill() directly.

    Signal changes on the final bar of `df` are dropped (no next-bar open
    exists to execute at) and logged rather than silently ignored.
    """
    trade_dates = shift_to_next_bar_execution(df, signal)
    if trade_dates.empty:
        return _empty_trades_frame()

    unexecutable = trade_dates[trade_dates["execution_date"].isna()]
    if not unexecutable.empty:
        for _, row in unexecutable.iterrows():
            print(
                f"WARNING: signal change on {row['signal_date'].date()} "
                f"(position {row['position_before']} -> {row['position_after']}) "
                f"has no next bar to execute at; dropped."
            )
    trade_dates = trade_dates.dropna(subset=["execution_date"]).reset_index(drop=True)
    if trade_dates.empty:
        return _empty_trades_frame()

    records = []
    for _, row in trade_dates.iterrows():
        change = row["position_change"]
        side = BUY if change > 0 else SELL
        quantity = int(round(abs(change) * quantity_per_unit))
        if quantity == 0:
            continue

        fill = calculate_fill(price=row["execution_price"], quantity=quantity, side=side, cost_model=cost_model)

        records.append(
            {
                "signal_date": row["signal_date"],
                "execution_date": row["execution_date"],
                "side": fill.side,
                "price": fill.price,
                "quantity": fill.quantity,
                "gross_value": fill.gross_value,
                "brokerage": fill.brokerage,
                "stt": fill.stt,
                "exchange_txn_charge": fill.exchange_txn_charge,
                "stamp_duty": fill.stamp_duty,
                "sebi_charge": fill.sebi_charge,
                "gst": fill.gst,
                "dp_charges": fill.dp_charges,
                "total_cost": fill.total_cost,
                "net_cash_flow": fill.net_cash_flow,
                "position_before": row["position_before"],
                "position_after": row["position_after"],
            }
        )

    return pd.DataFrame(records)


def _empty_trades_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "signal_date", "execution_date", "side", "price", "quantity", "gross_value",
            "brokerage", "stt", "exchange_txn_charge", "stamp_duty", "sebi_charge", "gst",
            "dp_charges", "total_cost", "net_cash_flow", "position_before", "position_after",
        ]
    )
