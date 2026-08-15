# Algo Backtester — Full Project Explanation

This document explains the project twice, on purpose: first in plain English with no assumed background, then again in engineering terms for a technical audience (e.g. interview prep). Read Part 1 if you want to understand *what this does and why*; read Part 2 if you want to understand *how it's built*.

---

## Part 1: Plain-English walkthrough

### The big picture

Imagine you have an idea: "when a stock's short-term average price crosses above its long-term average price, that usually means it's starting an uptrend — I should buy it. When it crosses back below, I should sell." That's a **trading strategy** — a rule for deciding when to own a stock and when not to.

The question is: would that idea have actually made money in the past, after accounting for real-world friction like broker fees and taxes? Testing an idea against historical data to answer that is called **backtesting**. This project is a backtesting engine, built in layers, where each layer solves one specific problem that naive backtests usually get wrong. Then it goes one step further: it lets the exact same rule watch real-time prices and pretend to trade with fake money, called **paper trading** — no real money is ever involved, it's a rehearsal.

The full pipeline:

```
historical prices → strategy decides buy/sell → simulate realistic trades → track money over time → score the results → validate honestly → (optionally) watch live prices → dashboard to see it all
```

### Trading basics, quickly

A **stock** is a small ownership slice of a company (e.g., SBIN is State Bank of India). Its **price** moves throughout the trading day as people buy and sell it. Each trading day, four prices matter most: the **open** (price at the start of the day), **close** (price at the end), and the **high**/**low** (the extremes hit during the day). Together with **volume** (how many shares traded), this is called **OHLCV** data — Open, High, Low, Close, Volume. It's the basic building block everything else in this project is made from.

**NSE** is the National Stock Exchange of India — where Indian stocks like SBIN trade. This project focuses on NSE stocks specifically.

### Step 1: Getting the data — `data/loader.py`

Before you can test any idea, you need history: what did the stock's price do every day for the last several years? `data/loader.py` fetches this from NSE (with a backup source if NSE's servers are down or blocking requests), cleans it up (checks for missing days, impossible values like a negative price, or a stock split that would make old prices misleading), and saves it to a local file so you don't have to re-download it every time.

Think of this step as: "build a big spreadsheet of Date, Open, High, Low, Close, Volume, one row per trading day."

### Step 2: Deciding when to buy or sell — `strategy/`

This is where your "idea" becomes code. The strategy looks at the price history and, for every single day, outputs one of three signals: **+1 (be long / own the stock)**, **0 (stay out, hold cash)**, or **-1 (short the stock**, betting it'll go down — not used by default here, since it's riskier and this project keeps things realistic by disabling it by default).

The one strategy currently built is a **moving average crossover**. A "moving average" is just the average closing price over the last N days — e.g., a 50-day moving average is the average of the last 50 closes, recalculated fresh every day as time moves forward. The idea: compute a *fast* one (short window, e.g. 50 days — reacts quickly) and a *slow* one (long window, e.g. 200 days — reacts slowly, represents the bigger trend). When the fast average is above the slow average, prices have been rising recently relative to the longer trend — read as "uptrend, be long." When fast dips below slow, read as "downtrend, get out."

Crucially, this strategy code is written once and reused unchanged for both backtesting on old data and later watching live prices — that reuse is a core design goal of the whole project (real trading systems work this way too).

### Step 3: Making the trades realistic — `engine/execution.py`

Here's where most beginner backtests quietly cheat, and this is the part of the project built specifically to not cheat.

**Problem 1 — knowing the future.** Say the strategy looks at Monday's *closing* price and decides "buy." A naive backtest might pretend you bought *at* that same Monday close — but that's impossible in real life, because you can't know Monday's final closing price until the market has already closed for the day. This project fixes that: every trade actually executes at the *next* day's opening price. This is called avoiding **look-ahead bias**, and it's the single most important correctness fix in the whole project.

**Problem 2 — pretending trading is free.** In reality, every trade costs money: brokerage fees, government taxes on securities transactions (called **STT** in India), exchange fees, stamp duty, GST, and small per-trade charges from your broker's back-office system (called **DP charges**). This project models all of these (using approximate real NSE rates, clearly documented as approximate since exact rates change) and subtracts them from every trade, so the final numbers reflect what you'd *actually* keep, not an unrealistic "free trading" fantasy.

### Step 4: Tracking the money — `engine/portfolio.py`

This layer plays "accountant." It starts with a pretend pile of cash (e.g., Rs. 100,000), and every time the strategy says "buy," it works out how many shares you can actually afford (never letting cash go negative — you can't spend money you don't have), executes the trade using Step 3's realistic pricing and costs, and updates your cash and share count. Every single day — not just on trading days — it also calculates your **equity**: cash you're holding plus the current market value of any shares you own. Plotting equity day by day over the years gives you the **equity curve** — the line chart of "how much is this pretend portfolio worth over time," the single most important visual in any backtest.

### Step 5: Scoring the results — `reporting/metrics.py`

Once you have an equity curve, you need numbers to judge it by, since "the line went up" isn't precise enough.

**Total return** is simply: (ending money − starting money) ÷ starting money, as a percentage. **Annualized return** converts that into "what would this be per year on average" so you can compare a 2-year test to a 5-year test fairly.

**Sharpe ratio** measures return *relative to how bumpy the ride was*. Two strategies can both end up +20%, but if one got there smoothly and the other got there through wild swings, the smooth one is arguably "better" for the same reward — Sharpe captures that. Higher is better; above 1 is generally considered decent.

**Max drawdown** is the worst peak-to-trough loss you'd have had to sit through — e.g., if your portfolio hit Rs. 150,000 at some point and later fell to Rs. 120,000 before recovering, that's a 20% drawdown. This tells you the worst pain a real person would have had to survive holding this strategy, which matters because a strategy that's profitable long-term but has terrifying dips is a strategy most people would panic-sell out of.

**Win rate** is simply what fraction of completed trades (buy followed by a later sell) made money versus lost money.

**Turnover** measures how often the strategy trades. High turnover means more transaction costs eating into returns for the same underlying idea.

Finally, every result is compared against **buy-and-hold** — what you'd have gotten by just buying the stock on day one and never touching it. A "clever" strategy that underperforms simply owning the stock the whole time isn't actually adding value — this comparison is *always* shown, never optional, specifically to keep the project honest.

### Step 6: Making sure it's not just luck — `engine/validation.py`

This is the most important safeguard in the whole project. Say you test 10 different combinations of "fast/slow window lengths" against 5 years of history, and pick whichever one made the most money. That sounds smart, but there's a trap: with enough combinations, *something* will look great purely by coincidence, fitted to quirks specific to that exact stretch of history. If you then claim "this strategy works," you're fooling yourself — this is called **overfitting**, and it's the single most common flaw in beginner backtests.

**Walk-forward validation** is the fix. Instead of picking parameters using the entire history, you split time into chunks: a "training" chunk (say, 2 years) where you're allowed to try different parameter combinations and pick whichever did best, and then a "test" chunk immediately after (say, 6 months) where you lock in that choice and see how it does on data the selection process never got to look at. Then you slide both windows forward in time and repeat — train on a new 2-year chunk, test on the next unseen 6 months, and so on through the whole history. Stitching all those test-only results together gives you an honest answer to "if I had actually been running and periodically re-tuning this strategy, live, only ever using information available at the time — how would it really have done?"

In this project's own SBIN results, this showed its value directly: the naive "pick one parameter set for the whole 5 years" approach had a Sharpe ratio of 0.57, but the walk-forward, out-of-sample version had a Sharpe of 1.21 with a smaller max drawdown — a materially more honest and better-looking result, obtained by *not* cheating.

### Step 7: Watching it happen live — `live/`

Once you trust a strategy on history, the natural next question is "would the same code work watching real, current prices?" This project answers yes by literally reusing Step 2's strategy code unchanged, but feeding it live price updates instead of a finished historical file.

A **tick** here just means "one price update as it happens." Getting genuinely live tick-by-tick data for NSE stocks for free (without a paid broker account) isn't really possible, so this project is upfront about a practical compromise: it *polls* (repeatedly re-checks, every few seconds) a public NSE price-quote page instead of getting a true instant push feed. There's also a "simulated" mode that replays old historical data *as if* it were arriving live — useful so you (or an interviewer) can see the whole pipeline work anytime, not just during Indian market hours (9:15am–3:30pm IST, weekdays).

Whenever the live signal changes (e.g., flips from "flat" to "long"), the system "pretends" to place that trade — updating fake cash and fake share counts using the exact same realistic cost math from Step 3 — and logs it clearly labeled **"PAPER TRADING — NOT REAL FUNDS"** everywhere, since no real broker connection or real money is ever involved.

### Step 8: The dashboard — `dashboard/app.py`

Everything above is Python code you could technically run line-by-line, but that's not demoable to another person. The dashboard (built with Streamlit, a tool for turning Python scripts into simple web apps) wraps it all in a clickable interface: type in a stock symbol and date range, pick strategy settings, hit "Run backtest," and see the equity curve chart, the scorecard of metrics, and the list of individual trades. There's also a button to run walk-forward validation and see its out-of-sample chart, and a button to run the simulated live paper-trading ticker right in the browser. This is the piece meant to actually be opened and clicked through in an interview, rather than reading code.

### Tying it together with a real result

Running this on SBIN, 2021–2026: the loader pulled ~5 years of daily SBIN prices; the 50/200-day crossover strategy looked at that history and produced a long/flat signal for every day; the execution engine turned each signal change into a real trade at the next day's open with real costs deducted; the portfolio tracker followed cash and shares day-by-day starting from Rs. 100,000; the metrics module scored the result (+55.99% total return, Sharpe 0.57) and compared it to just buying and holding SBIN the whole time (+151.70%) — showing the strategy actually lagged a simple buy-and-hold in this particular strong uptrend, which is a real, honest, explainable finding, not a failure of the code. Walk-forward validation then re-ran the whole process with periodic re-tuning on unseen data only, producing a more credible Sharpe of 1.21. That's the entire system, working exactly as designed, on real data.

---

## Part 2: Engineering deep-dive

### Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.10+ (targets 3.11+ per spec) | Standard for quant/data tooling |
| Data manipulation | pandas, numpy | DataFrame-based OHLCV pipeline throughout |
| Data source (primary) | `jugaad-data` (`jugaad_data.nse.stock_df`, `jugaad_data.nse.NSELive`) | Wraps NSE's public historical + quote REST endpoints |
| Data source (fallback) | `yfinance` | Hits Yahoo Finance with `.NS`-suffixed tickers |
| HTTP | `requests` (transitive, via the above libraries) | — |
| Local persistence | Flat CSV files (`data/cache/<SYMBOL>.csv`) | No DB; simplest thing that works for a single-user backtester |
| Visualization | matplotlib (static PNGs) + Plotly (interactive, in-dashboard) | matplotlib for `reporting/metrics.py`'s file output, Plotly for the dashboard's interactive charts |
| Web UI | Streamlit | Turns a plain Python script into a stateful web app with minimal boilerplate |
| Async runtime | `asyncio` (stdlib) | Drives the live tick feed as an async generator |
| Testing | `pytest` | 62 tests, no mocking framework needed |
| Version control | Git + GitHub | — |
| ML/statistics libraries | None in active use | See "ML models" below |

No database, no message queue, no containerization — a single-process, file-cached, synchronous-by-default system, deliberately, since the point is engine correctness, not infrastructure.

### Data flow / module boundaries

```
data.loader.fetch_ohlcv(symbol, start, end) -> pd.DataFrame[date, open, high, low, close, volume]
        │
        ▼
strategy.base.Strategy.run(df) -> pd.Series[int] ∈ {-1, 0, 1}   (target position per bar)
        │
        ▼
engine.execution.simulate_execution(df, signal) -> trades DataFrame   (standalone, fixed-qty)
   OR
engine.portfolio.run_backtest(df, signal) -> BacktestResult(equity_curve, trades)   (cash-aware, production path)
        │
        ▼
reporting.metrics.generate_report(equity_curve, trades, df) -> PerformanceReport
        │
        ▼
engine.validation.walk_forward_validate(df, param_grid, strategy_factory, ...) -> WalkForwardResult
        │
        ▼
live.paper_trader.PaperTrader.on_tick(Tick) -> reuses Strategy + engine.execution.calculate_fill directly
        │
        ▼
dashboard.app -> Streamlit UI over all of the above
```

Each arrow is a hard module boundary with a typed dataclass or DataFrame contract crossing it — no module reaches into another's internals. This is what lets each phase be built and unit-tested independently without later phases needing rewrites.

### Module-by-module, with actual interfaces

**`data/loader.py`** — `fetch_ohlcv(symbol, start_date, end_date, series="EQ", force_refresh=False) -> pd.DataFrame`. Internally: tries `_fetch_from_nse` (jugaad-data), catches any exception or empty result, falls back to `_fetch_from_yfinance`. Column names from either provider are normalized through `_normalize_columns()`, which maps a table of lowercased/stripped aliases (`"close"`, `"ltp"` → `close`, etc.) onto a fixed schema — defensive because NSE's raw column names have drifted across library versions, so column mapping is alias-based rather than hardcoded to one exact schema. Caching is incremental: `_load_cache`/`_save_cache` read/write one CSV per symbol, and `fetch_ohlcv` only fetches the date sub-ranges not already on disk (diffing `cached_min`/`cached_max` against the requested range), not a full re-pull. `validate_ohlcv(df, symbol) -> ValidationReport` is a separate pure function — checks weekday-calendar gaps, duplicate dates, OHLC consistency (`high < low`, non-positive prices), and flags >40%/60% overnight close ratios as *potential* unadjusted stock splits (heuristic, not auto-corrected).

**`strategy/base.py`** — `Strategy` is an `ABC` with one abstract method, `generate_signals(df) -> pd.Series`. The public entry point is `Strategy.run(df)`, which wraps `generate_signals` with input validation (`REQUIRED_COLUMNS`, monotonic date order) and output validation (`_validate_signals`: no NaN, values ⊆ `{-1, 0, 1}`, index/length match). Template Method pattern — subclasses only implement the decision logic, never the contract-checking.

**`strategy/moving_average.py`** — `MovingAverageCrossover(fast_window=50, slow_window=200, long_only=True)`. Uses `pandas.Series.rolling(window, min_periods=window).mean()` for both SMAs — `min_periods=window` is what forces NaN (→ signal 0) during warm-up rather than a partially-filled average. Signal is level-based (state at every bar: `fast > slow` → +1), not edge-triggered, since the execution layer needs a continuous target-position series, not discrete "buy here" events.

**`engine/execution.py`** — Two-tier API. `calculate_fill(price, quantity, side, cost_model) -> Fill` is the single source of truth for cost math: `CostModel` is a frozen dataclass (STT %, exchange txn %, stamp duty %, SEBI turnover %, GST % applied to brokerage+exchange, flat DP charge on sell). `shift_to_next_bar_execution(df, signal) -> DataFrame` is the look-ahead-bias fix, implemented as `signal - signal.shift(1, fill_value=0)` to find position deltas, each mapped to `df.loc[idx+1]`'s open (positional index arithmetic, not date arithmetic — `df` must be `reset_index(drop=True)`'d before use). `simulate_execution()` composes both for a standalone, fixed-quantity-per-signal-unit run — a deliberately simpler sibling to `portfolio.run_backtest`, kept so cost math has one canonical implementation both callers share.

**`engine/portfolio.py`** — `run_backtest(df, signal, cost_model, initial_capital, position_size_pct=0.95, allow_short=False) -> BacktestResult`. `_max_affordable_buy_quantity()` derives a closed-form cost multiplier — every BUY-side cost component in `CostModel` is proportional to `price*quantity` (no flat fee on buys), so `total_cash_needed = price*qty*(1 + Σpct_costs + gst_pct*(brokerage+exchange))`, solved directly for `max_qty = cash // (price*multiplier)`, with a defensive `while` loop only for floating-point edge cases — O(1) instead of trial-and-error per trade. The backtest loop is a single bar-by-bar pass (`for _, day in df.iterrows()`) checking a `scheduled_by_date` dict for due trades before marking to market — daily equity computed for *every* row, not just trade days, which is what makes the equity curve continuous rather than a sparse trade log.

**`reporting/metrics.py`** — Pure functions over the `equity_curve`/`trades` DataFrames: `sharpe_ratio` guards div-by-zero (constant equity → 0.0, not NaN/inf); `max_drawdown` uses `equity.cummax()` then `idxmin()` on the drawdown series, with `equity.loc[:trough_idx].idxmax()` to recover the actual peak; `compute_round_trip_pnls` pairs sequential BUY→SELL trades by scanning `trades.sort_values("execution_date")` and matching each SELL to the most recent open BUY (works because the portfolio is single-symbol, sequential, non-overlapping by construction). `turnover` is annualized: `(Σ trade notional / avg_equity) / (n_days/252)`.

**`engine/validation.py`** — `_generate_windows(n_rows, train_days, test_days, step_days)` produces `(train_start, train_end, test_start, test_end)` index tuples via a sliding-window loop, index-based rather than date-offset arithmetic (keeps it provider-agnostic). `walk_forward_validate()` does, per window: grid search over `param_grid` on the train slice only (each candidate scored via `generate_report(...).<selection_metric>`, default `"sharpe_ratio"`, exceptions from invalid param combos caught and logged rather than fatal), then evaluates the winning params on a `test_start - warmup_days` to `test_end` slice, trimming the warm-up rows off *after* running the strategy (so indicators see real history) but *before* scoring (so warm-up bars are never counted as performance). `_stitch_equity_curves()` chains windows into one continuous curve by treating each window's own equity path as a multiplier (`equity / initial_capital`) applied to the running capital carried over from the previous window — necessary because each window's `run_backtest` restarts fresh at `initial_capital`, so naively concatenating raw equity values would be wrong at every window boundary.

**`live/feed.py`** — `LiveFeed(ABC)` with `async def stream(symbol) -> AsyncIterator[Tick]`. `NSEPollingFeed` wraps `jugaad_data.nse.NSELive().stock_quote()` in a `while True: ... await asyncio.sleep(interval)` loop — REST polling, not a WebSocket, because NSE doesn't expose free public tick-level WebSocket access without a funded broker API account (Zerodha Kite Connect, Upstox, etc.). `SimulatedFeed` replays a DataFrame through the same interface. Both satisfy the same ABC, so `PaperTrader.run()` doesn't know or care which one it's consuming — that substitutability is the actual design goal, not the polling implementation itself.

**`live/paper_trader.py`** — `PaperTrader.on_tick(tick)` is fully synchronous (only the outer `run()` loop is async); it maintains `_current_day_bar: dict` as an in-progress OHLC aggregate, committing it into `self.history_df` on date rollover (`pd.concat`), then re-runs `self.strategy.run(live_df)` on history+today every tick and diffs the last signal value against `self.state.current_signal`. A 200-day SMA strategy run live genuinely re-evaluates a 200+ row rolling computation on every tick — fine at human-scale tick rates (seconds), not something you'd want at HFT tick rates, which is an explicit non-goal here.

**`dashboard/app.py`** — Streamlit script-rerun model: the whole file re-executes top-to-bottom on every widget interaction; state that must survive a rerun (`df`, `symbol`) is held in `st.session_state`. `@st.cache_data` memoizes `_load_data()` so switching UI controls doesn't re-hit the network/cache-CSV path unnecessarily. One non-obvious fix: `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))` at the top, because `streamlit run` doesn't reliably put the project root on `sys.path` depending on the invoking cwd — without it, `from data.loader import ...` fails with `ModuleNotFoundError`.

### ML models — there aren't any

Worth being direct: nothing in this project is machine learning. The signal generator (`MovingAverageCrossover`) is a deterministic rule over two rolling means — no training, no fitting, no learned parameters, no train/test split in the ML sense. The "train/test" language in `engine/validation.py` refers to walk-forward *parameter selection* (grid search over a small discrete set of window lengths, scored by backtested Sharpe), closer to hyperparameter tuning than to model training, with no generalization guarantee the way there would be with cross-validated regression.

If you wanted to genuinely add ML, the natural seams are already there: `Strategy.generate_signals(df) -> pd.Series` is the exact interface a model-backed strategy would implement too (e.g., a classifier predicting next-bar direction from engineered features, its `.predict()` output mapped to `{-1,0,1}`), and `engine.validation.walk_forward_validate`'s `strategy_factory` callable would just need to build/fit that model on each train window instead of instantiating `MovingAverageCrossover(**params)`. This wasn't built because the project's scope was a rule-based backtester with an execution/cost/validation engine as the differentiator, not a forecasting model — but the architecture doesn't block it.

### Testing approach

62 tests, no `unittest.mock` anywhere. Every numeric function (`calculate_fill`, `max_drawdown`, the MA crossover, the walk-forward stitching) is tested against values computed by hand in the test file's own docstring/comments and checked with `pytest.approx`, rather than against the implementation's own output — this catches logic bugs a "assert it returns something" test wouldn't. Network-dependent code (`_fetch_from_nse`, `NSEPollingFeed`) is deliberately *not* unit-tested (would require hitting real NSE servers or building a mock HTTP layer); those paths were instead smoke-tested manually against live data once, with output verified directly. `SimulatedFeed` and `test_live_feed.py` exist specifically to get equivalent coverage of the async-generator *contract* without a network dependency, using `asyncio.run()` inside otherwise-synchronous pytest test functions rather than pulling in `pytest-asyncio`.
