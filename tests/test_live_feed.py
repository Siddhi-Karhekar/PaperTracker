"""
Unit tests for live/feed.py's SimulatedFeed.

NSEPollingFeed hits the real network (nseindia.com) and can't be tested
offline -- consistent with how data/loader.py's actual NSE calls aren't
unit tested either, only its pure logic. SimulatedFeed has no such
dependency and is fully testable.
"""

import asyncio

import pandas as pd

from live.feed import SimulatedFeed


def _make_df(n=3):
    dates = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame(
        {
            "date": dates,
            "open": [100.0 + i for i in range(n)],
            "high": [101.0 + i for i in range(n)],
            "low": [99.0 + i for i in range(n)],
            "close": [100.5 + i for i in range(n)],
            "volume": [1000 + i for i in range(n)],
        }
    )


async def _collect(feed: SimulatedFeed, symbol: str):
    ticks = []
    async for tick in feed.stream(symbol):
        ticks.append(tick)
    return ticks


def test_simulated_feed_replays_bars_in_order_as_ticks():
    df = _make_df(3)
    feed = SimulatedFeed(df, interval_seconds=0)  # no delay -- keep the test fast

    ticks = asyncio.run(_collect(feed, "TEST"))

    assert len(ticks) == 3
    for i, tick in enumerate(ticks):
        assert tick.symbol == "TEST"
        assert tick.price == df["close"].iloc[i]
        assert tick.day_open == df["open"].iloc[i]
        assert tick.day_high == df["high"].iloc[i]
        assert tick.day_low == df["low"].iloc[i]
        assert tick.volume == df["volume"].iloc[i]


def test_simulated_feed_empty_dataframe_yields_no_ticks():
    df = _make_df(0)
    feed = SimulatedFeed(df, interval_seconds=0)

    ticks = asyncio.run(_collect(feed, "TEST"))

    assert ticks == []
