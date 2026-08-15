# Algo Backtester — Full Project Explanation

This document explains the project twice, on purpose: first in plain English with no assumed background, then again in engineering terms for a technical audience (e.g. interview prep). Read Part 1 if you want to understand *what this does and why*; read Part 2 if you want to understand *how it's built*. Part 3 covers the ML, explainable AI, and NLP extensions added after the original 9-phase build, in both styles.

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

### ML models — see Part 3

The original build had no machine learning at all: the signal generator (`MovingAverageCrossover`) is a deterministic rule over two rolling means, with no training, fitting, or learned parameters. Real ML (a classifier strategy, ML-driven position sizing, SHAP explainability, and an NLP chatbot) was added afterward — see Part 3 below for the full explanation of what was added and how, in both plain English and engineering terms.

### Testing approach

Every numeric function across the project (`calculate_fill`, `max_drawdown`, the MA crossover, the walk-forward stitching, the RSI/MACD feature formulas, SHAP additivity, the NLP intent classifier's held-out accuracy) is tested against values computed by hand or independently cross-checked, rather than against the implementation's own output — this catches logic bugs a "assert it returns something" test wouldn't. Network-dependent code (`_fetch_from_nse`, `NSEPollingFeed`) is deliberately *not* unit-tested (would require hitting real NSE servers or building a mock HTTP layer); those paths were instead smoke-tested manually against live data once, with output verified directly. `SimulatedFeed` and `test_live_feed.py` exist specifically to get equivalent coverage of the async-generator *contract* without a network dependency, using `asyncio.run()` inside otherwise-synchronous pytest test functions rather than pulling in `pytest-asyncio`.

---

## Part 3: ML, Explainable AI, and NLP extensions

These were added after the original 9-phase build, on request, to demonstrate ML/NLP/XAI engineering on top of the existing backtester — not because the rule-based strategy needed replacing. Same rule as everywhere else in this project: build it properly, test it against real or hand-verified numbers, and be honest about what it can't promise.

### In plain English

**The ML strategy.** Instead of a human-written rule ("buy when the fast average crosses above the slow average"), `MLClassifierStrategy` looks at a basket of technical indicators every day — momentum, volatility, RSI, MACD, how far price is from its moving averages — and asks a trained model: "based on patterns like this one in the past, is the price more likely to be higher in 5 days?" If the model is confident enough (above a set probability threshold), it signals "buy." This is genuinely learned from data rather than hand-specified, but that comes with a real cost: a rule like "50-day average above 200-day average" is something you can reason about and trust for understandable reasons, while a trained model's reasoning is opaque unless you specifically go looking for it — which is what the explainability piece is for.

**Explainable AI (XAI).** Because "trust me, the model said so" isn't good enough, every ML prediction can be broken down feature-by-feature using a technique called SHAP: for one specific day's prediction, SHAP shows exactly how many percentage points each individual indicator (say, RSI being unusually low, or momentum being strongly positive) pushed the prediction up or down. The dashboard shows this as a bar chart — you can literally see "the model went long today mostly because of strong 10-day momentum and an oversold RSI reading."

**Volatility-scaled position sizing.** Separately from deciding whether to buy, there's a question of how much to buy. The original approach was simple: always risk the same fixed percentage of your cash. The ML addition predicts how choppy/volatile the stock is likely to be over the next couple of weeks, and sizes the position smaller when things look risky and larger when they look calm — a standard professional risk-management technique (called volatility targeting) with a machine-learned forecast feeding it instead of just looking backward at recent volatility.

**The NLP chatbot.** After running a backtest, you can type a plain-English question into the dashboard — "what was the Sharpe ratio," "did it beat buy and hold," "what does drawdown mean," or anything else in your own words — and get a real answer pulled from that actual backtest's numbers, not a canned response. Under the hood, the primary path sends your question and that run's real numbers to a real AI model (Claude Haiku, a small and fast Claude model) with instructions to answer in one or two sentences using only those numbers — this is what handles open-ended questions and rephrasings well. That path needs an API key; if you don't provide one (or the call fails for any reason, e.g. no internet), the chatbot silently drops to a small, self-contained backup technique where your question is compared against a library of example questions it already knows how to handle, and it answers using whichever known question yours most closely resembles. That backup works instantly, for free, with no internet dependency, but only covers the specific set of questions it was built for.

**The honest caveat.** The rule-based strategy's key properties — no cheating by seeing the future, real costs applied, tested against unseen data — are true by construction; you can point at the code and prove it. A trained model's *usefulness* isn't provable the same way: it might have found a genuine pattern, or it might have just memorized noise in a few years of data that happens not to repeat. Both are included here to show real engineering (not toy code), not as a claim that the ML version is actually a better way to trade.

### In technical terms

**`ml/features.py`** computes fifteen backward-looking technical features per bar (1/5/10-day returns, 10/20-day realized volatility, 14-day RSI, MACD line/signal/histogram, SMA ratios, 10-day momentum, 20-day volume ratio, Bollinger %B) via pure pandas `rolling`/`ewm` operations — no external TA library. `compute_forward_labels()` and `compute_forward_volatility()` generate the (necessarily forward-looking) training targets for the classifier and the volatility regressor respectively; both are NaN for the last `horizon` rows where the future window doesn't exist yet, and both are only ever consumed as *training* targets, never as features (features and labels are computed by separate functions specifically so a caller can't accidentally feed a label in as a feature).

**`strategy/ml_signal.py`**'s `MLClassifierStrategy.generate_signals(df)` reserves the first `min_train_rows` of whatever `df` it receives purely for fitting (default: `RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=10)`), forces the signal to 0 across that entire span, and predicts genuinely out-of-sample for every row after it. This is verified directly in `tests/test_ml_signal.py` using a stub classifier that records exactly which feature rows it was fit and predicted on — the test asserts those arrays match an independently recomputed "rows before/after `min_train_rows`, with complete features and labels" slice, so a leakage bug (e.g. an off-by-one letting one future row into training) would fail the test on array-shape or array-content mismatch, not just "looks plausible."

**`ml/position_sizing.py`**'s `VolatilityForecaster` follows the identical train/predict split pattern (a `RandomForestRegressor` predicting `compute_forward_volatility`), and `inverse_vol_position_size()` converts a predicted daily-vol series into a `target_vol / predicted_vol` fraction, clipped to `[min_size, max_size]`. `engine/portfolio.run_backtest`'s `position_size_pct` parameter was extended to accept either the original fixed `float` or a `pd.Series` aligned to `df`'s index (backward compatible — all 6 original portfolio tests pass unchanged) — each BUY looks up that row's sizing value, falling back to `default_position_size_pct` on NaN (e.g. during the forecaster's own warm-up) rather than skipping the trade.

**`ml/explain.py`** wraps `shap.TreeExplainer`, which computes *exact* Shapley values for tree ensembles in polynomial time (not the sampling approximation general-purpose SHAP needs). The installed SHAP version (0.49.1) returns `shap_values(X)` as a `(n_samples, n_features, n_classes)` array for classifiers with a per-class `expected_value` array — this shape is version-dependent and was verified empirically against the actual installed library rather than assumed from documentation, since it has changed across SHAP releases. The additivity property (`expected_value + sum(shap values for a row) == that row's real model output`, exactly) is asserted in `tests/test_ml_explain.py` against real fitted `RandomForestClassifier`/`RandomForestRegressor` models, with `np.testing.assert_allclose(..., atol=1e-8)` — floating-point-exact, not approximate.

**`nlp/qa_engine.py`**'s `QAEngine.answer()` tries an LLM-backed path first: `_try_llm_answer()` calls `anthropic.Anthropic().messages.create()` with `model="claude-haiku-4-5-20251001"`, `max_tokens=200`, and a system prompt instructing 1-2 sentence answers with no preamble, grounded in a compact one-line context string (`_report_context()`) built from the run's real `PerformanceReport` fields and trade count — never the raw DataFrame, to keep the prompt small and latency low. That function returns `None` (not raising) whenever `ANTHROPIC_API_KEY` is unset, the `anthropic` package isn't installed, or the API call raises for any reason (network error, rate limit, bad key), and `answer()` only falls through to the offline path when it gets `None` back — so the LLM path is tried unconditionally on every question, not just low-confidence ones. The offline fallback is TF-IDF vectorization + nearest-neighbor cosine similarity against ~90 curated example phrasings across 14 intents, *not* a trained probabilistic classifier — a `MultinomialNB` classifier was tried first and empirically discarded: with only 5-10 short examples per class, its predicted probabilities came out poorly calibrated (correct predictions scored as low as 0.10-0.27 confidence, indistinguishable from genuinely out-of-domain questions). Switching to "return the label of the single most similar training example, using cosine similarity as the confidence score" measurably fixed both problems (verified in the same ad-hoc test used to make the decision, then formalized into `tests/test_nlp_qa_engine.py`): ~90% accuracy on held-out phrasings never seen in training, and every tested out-of-domain question scoring ~0.0 similarity against every trained intent. Within that fallback, anything below `confidence_threshold=0.25` (chosen with margin below the weakest genuine match observed, 0.32) gets a canned "I don't understand" message instead of a guessed intent. Answers for matched offline intents are template strings interpolating real fields from the `PerformanceReport` dataclass, never model-generated text, so every number in a fallback-path answer is exactly the number the backtest actually produced; LLM-path answers are model-generated text but are grounded in that same real data via the prompt context. Priority order (LLM tried first, offline fallback only on `None`) is asserted directly in `tests/test_nlp_qa_engine.py` by mocking `_try_llm_answer` to return a fixed string for a question the offline classifier would score low-confidence, and separately mocking it to return `None` to confirm the offline path still activates.

**Dashboard wiring** (`dashboard/app.py`): the strategy selector now branches on `is_ml_strategy` to show ML-specific sidebar controls (training rows, prediction horizon, confidence threshold) instead of MA windows, a "Use ML volatility-scaled position sizing" checkbox builds the `pd.Series` path through `run_backtest` described above, a SHAP panel renders (only when `strategy.last_model` is populated, i.e. only for the ML strategy after a real fit) both global feature importance and the most recent prediction's breakdown as Plotly bar charts, and the chatbot panel constructs a fresh `QAEngine` from that run's actual `report`/`trades` on every backtest, with `st.session_state["chat_history"]` persisting the conversation across Streamlit's rerun-on-every-interaction execution model. Walk-forward validation in the dashboard UI remains wired to the MA strategy only (grid-search × rolling-windows × model-refit for the ML strategy is slow enough that it doesn't belong in an interactive control) — `engine.validation.walk_forward_validate` supports the ML strategy generically via its `strategy_factory` callable, just not exposed as a one-click dashboard toggle; `scripts/generate_results.py` is the template for running it from a script instead.
