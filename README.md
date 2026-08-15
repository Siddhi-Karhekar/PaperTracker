# Algo Backtester — NSE Equities

A backtesting engine for NSE (Nifty 50) equities that models realistic transaction costs and slippage, reports risk-adjusted performance, and extends into live paper trading using the same strategy code.

## The problem

"Would this trading idea have actually worked?" is a much harder question than it sounds. Most first-attempt backtests quietly answer a different, easier question -- "does this idea look good on a chart if I let it cheat a little?" -- because they let a strategy trade at a price it couldn't have known yet (look-ahead bias), ignore brokerage/taxes/slippage entirely, and get tuned once against the full history until the numbers look good, without ever being checked against data the strategy didn't get to see.

This project answers the harder question properly: signals are computed only from information available at the time, every trade executes at the next bar's open (not the signal bar's own close) with a real NSE cost breakdown applied, and strategy parameters are validated with walk-forward testing -- picked on a training window, scored on the following unseen window, rolled forward through history -- rather than fit once to the whole dataset. The same strategy code that runs in the backtest also runs, unchanged, against live (or simulated-live) prices in paper-trading mode, which is the architectural claim the rest of this README backs up with actual code, not just a diagram.

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

### Live paper trading (`*** PAPER TRADING -- NOT REAL FUNDS ***`)

Run the identical strategy code against a live or simulated price feed:

```bash
# Simulated (default) -- replays real held-back history as "live" ticks, works any time, no network needed
python -m live.paper_trader SBIN

# Live -- polls real NSE quotes, only during market hours (9:15-15:30 IST, Mon-Fri)
python -m live.paper_trader SBIN --live
```

NSE doesn't offer free public WebSocket tick data for retail use without a broker account, so `NSEPollingFeed` polls NSE's public quote endpoint on an interval instead -- an honest, documented simplification rather than a silently cut corner. `SimulatedFeed` replays historical bars as ticks so the pipeline is demoable any time, not just during Indian market hours.

`live/feed.py` normalizes both sources into the same `Tick` event; `live/paper_trader.py`'s `PaperTrader` maintains an in-progress "today" bar from incoming ticks, commits it to history once the day rolls over, and re-runs the *same* `Strategy` and `engine.execution.calculate_fill` used in the backtest -- no separate live-only logic path. No real broker connection, no real orders, ever.

### Dashboard

```bash
streamlit run dashboard/app.py
```

Pick a symbol, date range, and strategy parameters in the sidebar and click **Run backtest** to see the equity curve vs. buy-and-hold, a metrics table, the trade log, and (optionally) a full walk-forward validation panel with its own out-of-sample equity curve. Below that, **Start live demo** runs the simulated paper-trading ticker from Phase 7 right in the browser -- this is the page to actually pull up in an interview.

Run tests:

```bash
pytest
```

## Results

SBIN, 50/200-day MA crossover, 2021-08-15 to 2026-08-14 (5 years), Rs. 100,000 starting capital, real NSE cost model applied. Generated with `python -m scripts.generate_results` -- see that script to reproduce these numbers or run them against a different symbol/period.

**Full-period backtest** (one parameter set fit once to the whole 5 years):

| Metric | Strategy | Buy & hold |
|---|---|---|
| Total return | +55.99% | +151.70% |
| Annualized return | +9.46% | +20.63% |
| Sharpe ratio | 0.57 | -- |
| Max drawdown | -22.75% | -- |
| Win rate | 50.0% (4 round trips) | -- |
| Turnover | 1.70x/yr | -- |

**Walk-forward validation** (params re-picked every ~2 years on a train window, tested on the following ~6 months, out-of-sample, 5 rolling windows):

| Metric | Strategy (out-of-sample) | Buy & hold |
|---|---|---|
| Total return | +79.60% | +94.38% |
| Annualized return | +26.39% | +30.46% |
| Sharpe ratio | 1.21 | -- |
| Max drawdown | -19.14% | -- |
| Win rate | 50.0% (2 round trips) | -- |

**Honest read of these numbers:** neither version beat buy-and-hold over this specific window -- SBIN was in a strong, fairly persistent uptrend for most of these 5 years, and a trend-following crossover strategy structurally gives up some of a strong monotonic run by design (it enters after a trend is confirmed and exits after it reverses, never catching the exact top or bottom). That's an expected, well-understood property of this strategy family, not a bug in the implementation.

What the walk-forward result *does* show is the value of the validation step itself: re-tuning periodically and testing only on unseen data more than doubled the Sharpe ratio (0.57 -> 1.21) and reduced the max drawdown (-22.75% -> -19.14%) versus fitting one parameter set to the whole history and calling it done. That's the actual point of Phase 6 -- it doesn't promise the strategy beats the market, it makes the "how good is this really" number honest.

**Demo video:** not recorded yet -- a 2-3 minute walkthrough of `streamlit run dashboard/app.py` (run a backtest, toggle walk-forward validation, run the live paper-trading demo) is worth adding to this README and your resume/LinkedIn before applying anywhere.

## Limitations

- **No backtest guarantees future performance.** These results are specific to SBIN over this one 5-year window; a different symbol, period, or market regime (sideways/choppy markets in particular) would produce different, possibly much worse, numbers. Trend-following strategies like this one are known to underperform buy-and-hold during strong sustained trends and outperform during choppy or declining markets -- the numbers above reflect that.
- **Data source is a public website scraper, not a licensed feed.** `jugaad-data` wraps NSE's public historical API, which can change format or rate-limit without notice; the yfinance fallback exists for exactly that reason. Cache what you use for anything you plan to rely on.
- **Cost model rates are approximate.** STT, exchange transaction charges, stamp duty, GST, and DP charges in `CostModel` are illustrative defaults for NSE delivery-equity trades, not live-verified current rates -- update them before treating exact rupee figures as authoritative.
- **Split/bonus adjustment is heuristic, not exact.** `data.loader.validate_ohlcv` flags large overnight price jumps as *potential* splits but doesn't auto-adjust historical prices. A real corporate action during the backtest period could distort signals or returns if the price series isn't manually checked.
- **Single symbol, single strategy family at a time.** The portfolio tracker (Phase 4) supports one instrument per backtest; there's no multi-asset allocation or correlation modeling. The stretch goal of combining two uncorrelated strategies isn't implemented.
- **No shorting mechanics.** `allow_short=True` exists but models shorting as simple negative inventory with no borrow cost, margin requirement, or availability constraint -- unrealistic for real short-selling, which is why the default strategy is long-only.
- **Walk-forward stitching assumes non-overlapping test windows** (`step_days == test_days`); overlapping windows return per-window results but no single combined out-of-sample curve, since overlapping periods can't be concatenated into one path without double-counting.
- **Live paper trading is a polling adapter, not true tick data**, and its signal can flicker intraday since it's evaluated against "today's close so far" rather than a finalized bar -- documented in detail in `live/paper_trader.py`.
- **This is a paper-trading and research project, not investment advice or a production trading system.** Nothing here should be read as a recommendation to trade SBIN or any other security.

## Build plan / progress

- [x] Phase 0 — Setup
- [x] Phase 1 — Data layer
- [x] Phase 2 — Strategy engine
- [x] Phase 3 — Execution simulator
- [x] Phase 4 — Portfolio and risk tracker
- [x] Phase 5 — Performance reporting
- [x] Phase 6 — Walk-forward validation
- [x] Phase 7 — Live data extension (optional)
- [x] Phase 8 — Dashboard
- [x] Phase 9 — Documentation and polish (demo video/GIF still pending -- see note below)

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
├── scripts/
│   └── generate_results.py
├── tests/
├── README.md
└── requirements.txt
```
