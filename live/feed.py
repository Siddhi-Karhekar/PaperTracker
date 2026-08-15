"""
live/feed.py

Normalizes live market data into one event shape regardless of source,
and streams it to subscribers. This is the "event dispatcher" from the
architecture diagram: everything downstream (the strategy, the paper
trader) only ever sees a Tick, never a provider's raw response format.

Two feed implementations:

  - NSEPollingFeed: polls NSE's public quote endpoint (via jugaad-data's
    NSELive) on a fixed interval. NSE does not offer free public
    WebSocket tick data for retail use without a broker account (Zerodha
    Kite Connect, Upstox, etc. require a funded trading account and paid
    API access) -- polling a REST endpoint is the honest, free
    alternative, not a corner cut silently. Only returns live data during
    NSE market hours (9:15-15:30 IST, Mon-Fri).

  - SimulatedFeed: replays historical OHLCV bars as a tick stream at a
    configurable speed. This is what makes the live pipeline demoable
    outside market hours -- an interviewer opening a link at 9pm on a
    Sunday should still see something move.

Both implement the same LiveFeed.stream() async generator interface, so
swapping in a real broker WebSocket later (a true push feed instead of a
polling adapter) requires no changes to anything downstream -- that
substitutability is the actual point of this abstraction.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncIterator, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Tick:
    """One normalized price update, regardless of where it came from."""

    symbol: str
    timestamp: datetime
    price: float
    day_open: Optional[float] = None
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    prev_close: Optional[float] = None
    volume: Optional[float] = None


class LiveFeed(ABC):
    @abstractmethod
    def stream(self, symbol: str) -> AsyncIterator[Tick]:
        """Async-generator: yield Tick events for `symbol`."""
        raise NotImplementedError


class NSEPollingFeed(LiveFeed):
    def __init__(self, poll_interval_seconds: float = 5.0):
        # 5s default: fast enough to feel live, slow enough not to hammer
        # NSE's public site with a scraping-style client.
        self.poll_interval_seconds = poll_interval_seconds

    async def stream(self, symbol: str) -> AsyncIterator[Tick]:
        from jugaad_data.nse import NSELive  # lazy import: network dependency

        nse = NSELive()
        while True:
            try:
                quote = nse.stock_quote(symbol)
                price_info = quote.get("priceInfo", {})
                high_low = price_info.get("intraDayHighLow", {}) or {}
                yield Tick(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    price=_safe_float(price_info.get("lastPrice")),
                    day_open=_safe_float(price_info.get("open")),
                    day_high=_safe_float(high_low.get("max")),
                    day_low=_safe_float(high_low.get("min")),
                    prev_close=_safe_float(price_info.get("previousClose")),
                )
            except Exception as exc:  # noqa: BLE001 - one failed poll shouldn't kill the stream
                logger.warning("NSE quote poll failed for %s: %s", symbol, exc)
            await asyncio.sleep(self.poll_interval_seconds)


class SimulatedFeed(LiveFeed):
    """
    Replays a historical OHLCV DataFrame as a tick stream, one bar per
    `interval_seconds`, using each bar's close as the tick price and its
    open/high/low/volume as that tick's day_* fields. Meant for demoing
    the live pipeline without depending on NSE market hours or a network
    connection.
    """

    def __init__(self, df: pd.DataFrame, interval_seconds: float = 1.0):
        self.df = df.reset_index(drop=True)
        self.interval_seconds = interval_seconds

    async def stream(self, symbol: str) -> AsyncIterator[Tick]:
        for _, row in self.df.iterrows():
            ts = row["date"]
            ts = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
            yield Tick(
                symbol=symbol,
                timestamp=ts,
                price=float(row["close"]),
                day_open=float(row["open"]),
                day_high=float(row["high"]),
                day_low=float(row["low"]),
                volume=float(row["volume"]),
            )
            if self.interval_seconds:
                await asyncio.sleep(self.interval_seconds)


def _safe_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
