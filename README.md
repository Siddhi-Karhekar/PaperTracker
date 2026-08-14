# Algo Backtester — NSE Equities

A backtesting engine for NSE (Nifty 50) equities that models realistic transaction costs and slippage, reports risk-adjusted performance, and extends into live paper trading using the same strategy code.

> Status: Phase 0 (setup) complete. This README will be filled in properly in Phase 9.

## The problem

*(Fill in at the end — plain-English framing of what this solves and why a naive backtest is misleading.)*

## Architecture

```
Historical data ─┐
                  ├─→ Event dispatcher → Strategy engine → Execution simulator → Portfolio tracker → Performance report
Live market feed ─┘                                              │
                                                                   └─→ (live mode) Paper trade executor → Dashboard
```

The strategy engine is written once and used identically in both backtest and live modes.

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## How to run

Fetch and validate NSE OHLCV data (tries the NSE historical API first, falls back to yfinance's `.NS` ticker if NSE errors out):

```bash
python -m data.loader SBIN --start 2020-01-01 --end 2024-12-31
```

Cached data lives per-symbol under `data/cache/<SYMBOL>.csv`; repeat calls only fetch the date ranges not already cached. Use `--force-refresh` to ignore the cache.

Generate signals from a strategy:

```python
from datetime import date
from data.loader import fetch_ohlcv
from strategy.moving_average import MovingAverageCrossover

df = fetch_ohlcv("SBIN", date(2015, 1, 1), date(2024, 12, 31))
strategy = MovingAverageCrossover(fast_window=50, slow_window=200)
signal = strategy.run(df)  # pd.Series of {-1, 0, 1}, aligned to df's index
```

`Strategy` (in `strategy/base.py`) is the interface every strategy plugs into: implement `generate_signals(df) -> pd.Series` and `run()` handles input/output validation for you. Signals represent *target position* (long/flat/short) at each bar, not one-shot trade instructions -- the execution simulator (Phase 3) is what turns position changes into actual orders.

Run tests:

```bash
pytest
```

## Results

*(Fill in at the end — Sharpe ratio, max drawdown, equity curve vs. buy-and-hold benchmark, with real numbers.)*

## Limitations

*(Fill in at the end — honest caveats: no backtest guarantees future performance, data/strategy selection bias, etc.)*

## Build plan / progress

- [x] Phase 0 — Setup
- [x] Phase 1 — Data layer
- [x] Phase 2 — Strategy engine
- [ ] Phase 3 — Execution simulator
- [ ] Phase 4 — Portfolio and risk tracker
- [ ] Phase 5 — Performance reporting
- [ ] Phase 6 — Walk-forward validation
- [ ] Phase 7 — Live data extension (optional)
- [ ] Phase 8 — Dashboard
- [ ] Phase 9 — Documentation and polish

## Project structure

```
algo-backtester/
├── data/
│   ├── loader.py
│   └── cache/
├── strategy/
│   ├── base.py
│   └── moving_average.py
├── engine/
│   ├── execution.py
│   ├── portfolio.py
│   └── validation.py
├── live/
│   ├── feed.py
│   └── paper_trader.py
├── reporting/
│   └── metrics.py
├── dashboard/
│   └── app.py
├── notebooks/
│   └── exploration.ipynb
├── tests/
├── README.md
└── requirements.txt
```
