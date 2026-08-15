"""
live/paper_trader.py

Runs the exact same Strategy code used in the backtest against a live (or
simulated-live) price feed, in paper-trading mode: positions and cash are
tracked in memory using the identical cost model from engine.execution,
but no real order is ever placed anywhere.

*** PAPER TRADING -- NOT REAL FUNDS ***
This module never connects to a broker, never places a real order, and
never moves real money. It exists to demonstrate the project's central
architectural claim: the same strategy engine written for the backtest
runs unchanged against live prices.

How a daily-bar strategy (like the 50/200-day MA crossover) becomes
"live": on startup, historical daily bars are loaded so the strategy's
indicators have real data to warm up on. Each incoming tick updates an
in-progress "today" bar (open/high/low/close aggregated from ticks seen
so far); the strategy re-evaluates its full signal on history + today's
bar-so-far on every tick. When a new day's first tick arrives, the
previous day's finished bar is committed permanently into history before
today's bar starts -- so multi-day indicators (like a 200-day SMA)
genuinely accumulate real trading days as the days pass, exactly like the
backtest would see them.

One honest limitation: because the signal is recomputed against "today's
close so far," it can flicker intraday and only truly settles once the
real session closes. That's an inherent property of adapting a daily-bar
strategy to intraday updates, not a bug -- a live position isn't
finalized until the day the signal was based on actually finishes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd

from engine.execution import BUY, SELL, CostModel, calculate_fill
from live.feed import LiveFeed, Tick
from strategy.base import Strategy

logger = logging.getLogger(__name__)

PAPER_TRADING_BANNER = "*** PAPER TRADING -- NOT REAL FUNDS ***"


@dataclass
class PaperTradeEvent:
    timestamp: datetime
    side: str
    price: float
    quantity: int
    total_cost: float
    net_cash_flow: float
    cash_after: float
    quantity_held_after: float


@dataclass
class PaperTraderState:
    cash: float
    quantity_held: float = 0
    current_signal: int = 0
    trades: list = field(default_factory=list)
    last_price: Optional[float] = None

    @property
    def equity(self) -> float:
        return self.cash + self.quantity_held * (self.last_price or 0.0)


class PaperTrader:
    def __init__(
        self,
        symbol: str,
        strategy: Strategy,
        history_df: pd.DataFrame,
        initial_capital: float = 100_000.0,
        cost_model: CostModel = CostModel(),
        position_size_pct: float = 0.95,
    ):
        self.symbol = symbol
        self.strategy = strategy
        self.history_df = history_df.reset_index(drop=True).copy()
        self.cost_model = cost_model
        self.position_size_pct = position_size_pct
        self.state = PaperTraderState(cash=initial_capital)
        self._current_day_bar: Optional[dict] = None
        logger.info(PAPER_TRADING_BANNER)
        logger.info("Starting paper trader for %s with %.2f initial capital", symbol, initial_capital)

    def _update_current_day_bar(self, tick: Tick) -> None:
        today = pd.Timestamp(tick.timestamp).normalize()

        if self._current_day_bar is not None and self._current_day_bar["date"] != today:
            # Yesterday is over: commit its final aggregated bar to history
            # permanently before today's bar starts accumulating.
            self.history_df = pd.concat(
                [self.history_df, pd.DataFrame([self._current_day_bar])], ignore_index=True
            )
            self._current_day_bar = None

        if self._current_day_bar is None:
            self._current_day_bar = {
                "date": today,
                "open": tick.day_open if tick.day_open is not None else tick.price,
                "high": tick.day_high if tick.day_high is not None else tick.price,
                "low": tick.day_low if tick.day_low is not None else tick.price,
                "close": tick.price,
                "volume": tick.volume if tick.volume is not None else 0,
            }
        else:
            bar = self._current_day_bar
            bar["high"] = max(bar["high"], tick.day_high if tick.day_high is not None else tick.price)
            bar["low"] = min(bar["low"], tick.day_low if tick.day_low is not None else tick.price)
            bar["close"] = tick.price
            if tick.volume is not None:
                bar["volume"] = tick.volume

    def _live_df(self) -> pd.DataFrame:
        return pd.concat([self.history_df, pd.DataFrame([self._current_day_bar])], ignore_index=True)

    def on_tick(self, tick: Tick) -> Optional[PaperTradeEvent]:
        self.state.last_price = tick.price
        self._update_current_day_bar(tick)
        live_df = self._live_df()

        signal = self.strategy.run(live_df)
        new_position = int(signal.iloc[-1])
        old_position = self.state.current_signal
        if new_position == old_position:
            return None

        event = self._execute(old_position, new_position, tick)
        # Only adopt the new target position if we actually traded into it,
        # or if there was genuinely nothing to sell (already flat) -- an
        # unaffordable buy is left as a pending mismatch so the trader
        # keeps retrying on the next tick instead of silently giving up.
        if event is not None or (new_position < old_position and self.state.quantity_held == 0):
            self.state.current_signal = new_position
        return event

    def _execute(self, old_position: int, new_position: int, tick: Tick) -> Optional[PaperTradeEvent]:
        if new_position > old_position:
            cash_to_use = self.state.cash * self.position_size_pct
            quantity = int(cash_to_use // tick.price)
            side = BUY
        else:
            quantity = int(self.state.quantity_held)
            side = SELL

        if quantity <= 0:
            logger.warning(
                "Signal changed (%s -> %s) but nothing to trade (qty=%s); skipped.",
                old_position, new_position, quantity,
            )
            return None

        fill = calculate_fill(price=tick.price, quantity=quantity, side=side, cost_model=self.cost_model)
        self.state.cash += fill.net_cash_flow
        self.state.quantity_held += fill.quantity if side == BUY else -fill.quantity

        event = PaperTradeEvent(
            timestamp=tick.timestamp,
            side=fill.side,
            price=fill.price,
            quantity=fill.quantity,
            total_cost=fill.total_cost,
            net_cash_flow=fill.net_cash_flow,
            cash_after=self.state.cash,
            quantity_held_after=self.state.quantity_held,
        )
        self.state.trades.append(event)
        logger.info(
            "%s %s %d %s @ %.2f | cash=%.2f qty=%.0f equity=%.2f",
            PAPER_TRADING_BANNER, fill.side, fill.quantity, self.symbol, fill.price,
            self.state.cash, self.state.quantity_held, self.state.equity,
        )
        return event

    async def run(self, feed: LiveFeed) -> None:
        async for tick in feed.stream(self.symbol):
            self.on_tick(tick)


if __name__ == "__main__":
    import argparse
    import asyncio
    from datetime import date, timedelta

    from data.loader import fetch_ohlcv
    from live.feed import NSEPollingFeed, SimulatedFeed
    from strategy.moving_average import MovingAverageCrossover

    parser = argparse.ArgumentParser(description=f"{PAPER_TRADING_BANNER} Run the strategy against a live or simulated feed.")
    parser.add_argument("symbol", help="NSE symbol, e.g. SBIN")
    parser.add_argument("--live", action="store_true", help="Poll real NSE quotes (market hours only). Default: simulate.")
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--fast-window", type=int, default=50)
    parser.add_argument("--slow-window", type=int, default=200)
    parser.add_argument("--lookback-days", type=int, default=400, help="Calendar days of history to fetch for warm-up.")
    parser.add_argument("--sim-speed", type=float, default=0.2, help="Seconds between simulated ticks.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger.info(PAPER_TRADING_BANNER)

    end = date.today()
    start = end - timedelta(days=args.lookback_days)
    full_history = fetch_ohlcv(args.symbol, start, end)

    strategy = MovingAverageCrossover(fast_window=args.fast_window, slow_window=args.slow_window)

    if args.live:
        # Warm up on everything fetched so far; live ticks arrive from here on.
        warmup_df = full_history
        feed = NSEPollingFeed()
    else:
        # Simulate: hold back the most recent slice of real history and
        # replay it as if it were arriving live, so the demo actually
        # shows signal transitions instead of just idling.
        split = max(len(full_history) - 60, args.slow_window + 1)
        warmup_df = full_history.iloc[:split]
        replay_df = full_history.iloc[split:]
        feed = SimulatedFeed(replay_df, interval_seconds=args.sim_speed)
        logger.info("Simulating: replaying %d days of held-back history as live ticks.", len(replay_df))

    trader = PaperTrader(args.symbol, strategy, warmup_df, initial_capital=args.capital)
    asyncio.run(trader.run(feed))
