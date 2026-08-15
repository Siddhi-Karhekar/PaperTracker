"""
Unit tests for live/paper_trader.py.

Uses a trivial, fully deterministic test-only strategy (long whenever the
latest close is above a fixed threshold) instead of the real MA crossover,
so the scenario below can be verified entirely by hand -- see the comment
above test_on_tick_hand_computed_scenario for the arithmetic. This
isolates PaperTrader's own logic (day-bar aggregation, day-rollover
history commits, execution) from strategy math, which is already covered
separately in tests/test_strategy.py.
"""

from datetime import datetime

import pandas as pd
import pytest

from engine.execution import BUY, SELL, CostModel
from live.feed import Tick
from live.paper_trader import PaperTrader
from strategy.base import Strategy

ZERO_COST = CostModel(
    brokerage_pct=0.0, stt_pct=0.0, exchange_txn_pct=0.0, stamp_duty_pct=0.0,
    sebi_turnover_pct=0.0, gst_pct=0.0, dp_charges_flat=0.0,
)


class _AboveThreshold(Strategy):
    """Long whenever the latest close is above `threshold`, else flat."""

    def __init__(self, threshold: float):
        self.threshold = threshold
        self.name = "above_threshold"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        return (df["close"] > self.threshold).astype(int)


def _history(closes, start="2024-01-01"):
    dates = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [0] * len(closes),
        }
    )


def test_on_tick_hand_computed_scenario():
    # History: one prior day, close=95 (below the threshold=100).
    # threshold=100, initial_capital=1000, position_size_pct=1.0, zero-cost.
    #
    # Tick1 (Jan2 10:00, price=90): new day bar for Jan2 starts (open=90,
    #   since no day_open given). live close=90 <= 100 -> signal=0.
    #   old_position=0 -> no change, no trade.
    #
    # Tick2 (Jan2 10:05, price=110, same day): bar updates -- close=110.
    #   live close=110 > 100 -> signal=1. old_position=0 -> BUY.
    #   cash_to_use = 1000 * 1.0 = 1000; qty = floor(1000/110) = 9
    #   cash = 1000 - 9*110 = 1000 - 990 = 10; quantity_held = 9
    #
    # Tick3 (Jan3 09:16, price=105): day rollover -- Jan2's bar (close=110)
    #   is committed to history. New Jan3 bar starts, close=105 > 100 ->
    #   signal=1. old_position=1 (still long) -> no change, no trade.
    #
    # Tick4 (Jan3 09:20, price=95, same day): bar updates -- close=95.
    #   95 <= 100 -> signal=0. old_position=1 -> SELL all 9 shares @ 95.
    #   cash = 10 + 9*95 = 10 + 855 = 865; quantity_held = 0
    history = _history([95.0], start="2024-01-01")  # single prior day
    strategy = _AboveThreshold(threshold=100.0)
    trader = PaperTrader(
        "TEST", strategy, history,
        initial_capital=1000.0, cost_model=ZERO_COST, position_size_pct=1.0,
    )

    e1 = trader.on_tick(Tick(symbol="TEST", timestamp=datetime(2024, 1, 2, 10, 0), price=90.0))
    assert e1 is None
    assert trader.state.quantity_held == 0
    assert trader.state.cash == pytest.approx(1000.0)

    e2 = trader.on_tick(Tick(symbol="TEST", timestamp=datetime(2024, 1, 2, 10, 5), price=110.0))
    assert e2 is not None
    assert e2.side == BUY
    assert e2.quantity == 9
    assert trader.state.quantity_held == 9
    assert trader.state.cash == pytest.approx(10.0)

    e3 = trader.on_tick(Tick(symbol="TEST", timestamp=datetime(2024, 1, 3, 9, 16), price=105.0))
    assert e3 is None  # still long, no change
    assert trader.state.quantity_held == 9
    # Day rollover happened: Jan2's bar should now be a permanent row in history.
    assert len(trader.history_df) == 2
    committed = trader.history_df.iloc[-1]
    assert committed["date"] == pd.Timestamp("2024-01-02")
    assert committed["close"] == pytest.approx(110.0)

    e4 = trader.on_tick(Tick(symbol="TEST", timestamp=datetime(2024, 1, 3, 9, 20), price=95.0))
    assert e4 is not None
    assert e4.side == SELL
    assert e4.quantity == 9
    assert trader.state.quantity_held == 0
    assert trader.state.cash == pytest.approx(865.0)


def test_paper_trading_banner_is_defined_and_unambiguous():
    from live.paper_trader import PAPER_TRADING_BANNER
    assert "PAPER TRADING" in PAPER_TRADING_BANNER
    assert "NOT REAL FUNDS" in PAPER_TRADING_BANNER


def test_insufficient_cash_does_not_falsely_update_signal():
    # If a BUY can't be afforded (qty=0), the trader must keep treating the
    # position as flat internally and retry the entry, not silently mark
    # itself as "long" without actually holding anything.
    history = _history([95.0])
    strategy = _AboveThreshold(threshold=100.0)
    trader = PaperTrader(
        "TEST", strategy, history,
        initial_capital=1.0,  # not enough to buy even 1 share at 110
        cost_model=ZERO_COST, position_size_pct=1.0,
    )

    event = trader.on_tick(Tick(symbol="TEST", timestamp=datetime(2024, 1, 2, 10, 0), price=110.0))
    assert event is None
    assert trader.state.current_signal == 0  # not falsely marked long
    assert trader.state.quantity_held == 0


def test_equity_property_reflects_cash_and_position():
    history = _history([95.0])
    strategy = _AboveThreshold(threshold=100.0)
    trader = PaperTrader(
        "TEST", strategy, history,
        initial_capital=1000.0, cost_model=ZERO_COST, position_size_pct=1.0,
    )
    trader.on_tick(Tick(symbol="TEST", timestamp=datetime(2024, 1, 2, 10, 0), price=110.0))
    # cash=10, quantity_held=9, last_price=110 -> equity = 10 + 9*110 = 1000
    assert trader.state.equity == pytest.approx(1000.0)
