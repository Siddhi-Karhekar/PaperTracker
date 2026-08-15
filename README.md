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

Turn signals into realistic filled trades:

```python
from engine.execution import simulate_execution, CostModel

trades = simulate_execution(df, signal, cost_model=CostModel(), quantity_per_unit=10)
```

`simulate_execution` fixes the most common backtest bug (look-ahead bias) by executing every position change at the **next bar's open**, never the signal bar's own close -- a signal known at bar t's close can't be acted on until bar t+1. Each fill also gets a realistic NSE delivery-equity cost breakdown (STT, exchange transaction charges, stamp duty, GST, DP charges) via `CostModel` -- all rates are approximate defaults and configurable; verify current rates before citing exact numbers. Position sizing here is a simple fixed-quantity-per-signal-unit model; Phase 4's portfolio tracker will size trades from real, cash-constrained capital by calling `engine.execution.calculate_fill()` directly.

Run a full backtest with real cash-aware position sizing and a daily equity curve:

```python
from engine.portfolio import run_backtest
from engine.execution import CostModel

result = run_backtest(df, signal, cost_model=CostModel(), initial_capital=100_000, position_size_pct=0.95)
result.equity_curve   # date, cash, quantity_held, position_value, total_equity -- one row per trading day
result.trades         # full trade log
result.final_equity   # portfolio value on the last day
```

`run_backtest` reuses the same next-bar-open execution timing and cost model from Phase 3 -- it only adds position sizing (from real available cash, capped by `position_size_pct`) and never lets cash go negative. Shorting is rejected unless `allow_short=True` is passed explicitly.

Score the backtest and chart it against buy-and-hold:

```python
from reporting.metrics import generate_report, plot_equity_curve, plot_drawdown

report = generate_report(result.equity_curve, result.trades, df, initial_capital=100_000)
print(report.summary())

plot_equity_curve(result.equity_curve, df, initial_capital=100_000, output_path="reports/equity.png")
plot_drawdown(result.equity_curve, output_path="reports/drawdown.png")
```

`generate_report` computes total/annualized return, Sharpe ratio, max drawdown (with peak/trough dates), win rate (from realized round-trip P&L, net of costs), annualized turnover, and a buy-and-hold benchmark for comparison -- a strategy that doesn't beat holding the stock outright isn't saying much, so that comparison is always included, never optional.

Validate out-of-sample with walk-forward validation, instead of trusting one parameter set fit to the whole history:

```python
from engine.validation import walk_forward_validate
from strategy.moving_average import MovingAverageCrossover

param_grid = [{"fast_window": f, "slow_window": s} for f, s in [(20, 100), (50, 200), (30, 150)]]

result = walk_forward_validate(
    df,
    param_grid=param_grid,
    strategy_factory=lambda **p: MovingAverageCrossover(**p),
    train_days=252 * 2,   # 2 years to pick params
    test_days=126,        # ~6 months out-of-sample
    warmup_days=200,      # >= largest slow_window, so the test window isn't stuck flat during warm-up
)

print(result.summary_table())          # one row per window: chosen params + out-of-sample metrics
print(result.out_of_sample_report.summary())  # the metrics that actually matter: pure out-of-sample performance
```

Each window picks parameters using *only* its train segment, then is scored on the following unseen test segment -- rolling forward through the full history. The stitched `out_of_sample_report` is the honest answer to "how would this have performed with periodic re-tuning on data I hadn't seen yet," which is a fundamentally different question than fitting one parameter set to the whole history at once.

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
- [x] Phase 3 — Execution simulator
- [x] Phase 4 — Portfolio and risk tracker
- [x] Phase 5 — Performance reporting
- [x] Phase 6 — Walk-forward validation
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
