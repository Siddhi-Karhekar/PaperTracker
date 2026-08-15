"""
dashboard/app.py

Streamlit dashboard: pick a symbol, date range, and strategy, run a
backtest through the real cost model / next-open execution / cash-aware
portfolio sizing from Phases 3-4, see the equity curve against
buy-and-hold, the metrics table, the trade log, optional walk-forward
validation (Phase 6), and a simulated live paper-trading ticker (Phase
7). This is the thing to actually demo in an interview -- a working link
beats a GitHub repo nobody runs.

Run with:
    streamlit run dashboard/app.py

*** The live panel below replays real held-back historical data as a
simulated tick stream by default, and is clearly labeled PAPER TRADING --
NOT REAL FUNDS throughout. No real broker connection, no real orders,
ever. ***
"""

from __future__ import annotations

import time
from datetime import date, timedelta

import plotly.graph_objects as go
import streamlit as st

from data.loader import DataFetchError, fetch_ohlcv, validate_ohlcv
from engine.execution import CostModel
from engine.portfolio import run_backtest
from engine.validation import walk_forward_validate
from live.feed import Tick
from live.paper_trader import PAPER_TRADING_BANNER, PaperTrader
from reporting.metrics import buy_and_hold_benchmark, generate_report
from strategy.moving_average import MovingAverageCrossover

st.set_page_config(page_title="Algo Backtester", layout="wide")

STRATEGIES = {"Moving Average Crossover": MovingAverageCrossover}


@st.cache_data(show_spinner=False)
def _load_data(symbol: str, start: date, end: date):
    return fetch_ohlcv(symbol, start, end)


# ---------------------------------------------------------------------------
# Sidebar: inputs
# ---------------------------------------------------------------------------
st.sidebar.title("Algo Backtester")
st.sidebar.caption("NSE equities -- backtest + walk-forward validation + paper trading")

symbol = st.sidebar.text_input("NSE symbol", value="SBIN").strip().upper()

default_end = date.today() - timedelta(days=1)
default_start = default_end - timedelta(days=365 * 3)
start_date = st.sidebar.date_input("Start date", value=default_start)
end_date = st.sidebar.date_input("End date", value=default_end)

st.sidebar.subheader("Strategy")
strategy_name = st.sidebar.selectbox("Strategy", list(STRATEGIES.keys()))
fast_window = st.sidebar.number_input("Fast window", min_value=2, max_value=400, value=50)
slow_window = st.sidebar.number_input("Slow window", min_value=3, max_value=500, value=200)

st.sidebar.subheader("Portfolio")
initial_capital = st.sidebar.number_input("Initial capital (Rs.)", min_value=1000.0, value=100_000.0, step=10_000.0)
position_size_pct = st.sidebar.slider("Position size (% of cash per entry)", min_value=10, max_value=100, value=95) / 100.0

st.sidebar.subheader("Walk-forward validation")
run_wfv = st.sidebar.checkbox("Also run walk-forward validation", value=False)
train_days = st.sidebar.number_input("Train window (days)", min_value=50, value=504, disabled=not run_wfv)
test_days = st.sidebar.number_input("Test window (days)", min_value=20, value=126, disabled=not run_wfv)

run_clicked = st.sidebar.button("Run backtest", type="primary")

st.title("Algo Backtester")
st.caption(f"{PAPER_TRADING_BANNER} This dashboard never places real orders.")

if fast_window >= slow_window:
    st.sidebar.error("Fast window must be smaller than slow window.")
    st.stop()

# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
if run_clicked:
    with st.spinner(f"Fetching {symbol} data..."):
        try:
            df = _load_data(symbol, start_date, end_date)
        except DataFetchError as exc:
            st.error(f"Could not fetch data for {symbol}: {exc}")
            st.stop()
    st.session_state["df"] = df
    st.session_state["symbol"] = symbol

if "df" in st.session_state:
    df = st.session_state["df"]
    active_symbol = st.session_state["symbol"]

    with st.expander(f"Data quality report -- {active_symbol} ({len(df)} rows)"):
        st.text(validate_ohlcv(df, active_symbol).summary())

    strategy = STRATEGIES[strategy_name](fast_window=fast_window, slow_window=slow_window)
    signal = strategy.run(df)
    result = run_backtest(df, signal, cost_model=CostModel(), initial_capital=initial_capital, position_size_pct=position_size_pct)
    report = generate_report(result.equity_curve, result.trades, df, initial_capital)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total return", f"{report.total_return:+.2%}")
    col2.metric("Annualized return", f"{report.annualized_return:+.2%}")
    col3.metric("Sharpe ratio", f"{report.sharpe_ratio:.2f}")
    col4.metric("Max drawdown", f"{report.max_drawdown:.2%}")
    wr_label = "n/a" if report.win_rate is None else f"{report.win_rate:.1%}"
    col5.metric("Win rate", wr_label, help=f"{report.n_round_trips} round trips")
    st.caption(f"Buy & hold total return: {report.benchmark_total_return:+.2%}  |  Turnover: {report.turnover:.2f}x/yr")

    benchmark = buy_and_hold_benchmark(df, initial_capital)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.equity_curve["date"], y=result.equity_curve["total_equity"], name="Strategy"))
    fig.add_trace(go.Scatter(x=benchmark["date"], y=benchmark["total_equity"], name="Buy & hold", line=dict(dash="dash")))
    fig.update_layout(title="Equity curve vs. buy-and-hold", xaxis_title="Date", yaxis_title="Portfolio value (Rs.)", height=450)
    st.plotly_chart(fig, use_container_width=True)

    equity = result.equity_curve["total_equity"]
    drawdown = equity / equity.cummax() - 1.0
    dd_fig = go.Figure()
    dd_fig.add_trace(go.Scatter(x=result.equity_curve["date"], y=drawdown, fill="tozeroy", name="Drawdown", line=dict(color="firebrick")))
    dd_fig.update_layout(title="Drawdown", xaxis_title="Date", yaxis_title="Drawdown", height=250)
    st.plotly_chart(dd_fig, use_container_width=True)

    with st.expander(f"Trade log ({len(result.trades)} trades)"):
        st.dataframe(result.trades, use_container_width=True)

    if run_wfv:
        st.subheader("Walk-forward validation")
        param_grid = [
            {"fast_window": f, "slow_window": s}
            for f, s in {(20, 100), (50, 200), (30, 150), (int(fast_window), int(slow_window))}
            if f < s
        ]
        with st.spinner("Running walk-forward validation (grid search per window)..."):
            try:
                wfv_result = walk_forward_validate(
                    df,
                    param_grid=param_grid,
                    strategy_factory=lambda **p: MovingAverageCrossover(**p),
                    train_days=int(train_days),
                    test_days=int(test_days),
                    warmup_days=int(slow_window),
                    cost_model=CostModel(),
                    initial_capital=initial_capital,
                    position_size_pct=position_size_pct,
                )
            except ValueError as exc:
                st.warning(f"Walk-forward validation couldn't run: {exc}")
                wfv_result = None

        if wfv_result is not None:
            st.dataframe(wfv_result.summary_table(), use_container_width=True)
            if wfv_result.out_of_sample_report is not None:
                oos = wfv_result.out_of_sample_report
                st.caption(
                    f"Out-of-sample: total return {oos.total_return:+.2%}, "
                    f"annualized {oos.annualized_return:+.2%}, Sharpe {oos.sharpe_ratio:.2f}, "
                    f"max drawdown {oos.max_drawdown:.2%}"
                )
            if wfv_result.stitched_equity_curve is not None:
                oos_fig = go.Figure()
                oos_fig.add_trace(
                    go.Scatter(
                        x=wfv_result.stitched_equity_curve["date"],
                        y=wfv_result.stitched_equity_curve["total_equity"],
                        name="Out-of-sample equity",
                    )
                )
                oos_fig.update_layout(title="Walk-forward out-of-sample equity curve", height=350)
                st.plotly_chart(oos_fig, use_container_width=True)
else:
    st.info("Set your parameters in the sidebar and click **Run backtest** to get started.")

# ---------------------------------------------------------------------------
# Live paper trading ticker (simulated by default)
# ---------------------------------------------------------------------------
st.divider()
st.subheader(PAPER_TRADING_BANNER)
st.caption("Replays real held-back historical data as a live tick stream. No real broker connection, no real orders.")

if "live_symbol" not in st.session_state:
    st.session_state["live_symbol"] = st.session_state.get("symbol", "SBIN")

live_symbol = st.text_input("Symbol for live demo", key="live_symbol")
n_ticks = st.slider("Number of simulated ticks", min_value=10, max_value=100, value=30)
tick_delay = st.slider("Delay between ticks (seconds)", min_value=0.0, max_value=1.0, value=0.15)

if st.button("Start live demo"):
    with st.spinner(f"Fetching {live_symbol} data for the live demo..."):
        try:
            live_source_df = _load_data(live_symbol, date.today() - timedelta(days=500), date.today() - timedelta(days=1))
        except DataFetchError as exc:
            st.error(f"Could not fetch data for {live_symbol}: {exc}")
            st.stop()

    warmup_len = max(len(live_source_df) - n_ticks, min(210, len(live_source_df)))
    warmup_df = live_source_df.iloc[:warmup_len]
    replay_df = live_source_df.iloc[warmup_len:warmup_len + n_ticks].reset_index(drop=True)

    if replay_df.empty:
        st.warning("Not enough history to run the live demo -- try a symbol with a longer trading history.")
    else:
        live_strategy = STRATEGIES[strategy_name](fast_window=fast_window, slow_window=slow_window)
        trader = PaperTrader(live_symbol, live_strategy, warmup_df, initial_capital=initial_capital, position_size_pct=position_size_pct)

        metrics_placeholder = st.empty()
        chart_placeholder = st.empty()
        log_placeholder = st.empty()

        price_history, equity_history, dates_history = [], [], []

        for _, row in replay_df.iterrows():
            tick = Tick(
                symbol=live_symbol,
                timestamp=row["date"].to_pydatetime(),
                price=float(row["close"]),
                day_open=float(row["open"]),
                day_high=float(row["high"]),
                day_low=float(row["low"]),
                volume=float(row["volume"]),
            )
            event = trader.on_tick(tick)

            price_history.append(tick.price)
            dates_history.append(tick.timestamp)
            equity_history.append(trader.state.equity)

            with metrics_placeholder.container():
                c1, c2, c3 = st.columns(3)
                c1.metric("Last price", f"Rs. {tick.price:,.2f}")
                c2.metric("Position", "LONG" if trader.state.quantity_held > 0 else "FLAT")
                c3.metric("Paper equity", f"Rs. {trader.state.equity:,.2f}")

            live_fig = go.Figure()
            live_fig.add_trace(go.Scatter(x=dates_history, y=equity_history, name="Paper equity"))
            live_fig.update_layout(title=f"{PAPER_TRADING_BANNER} -- live equity", height=300)
            chart_placeholder.plotly_chart(live_fig, use_container_width=True)

            if event is not None:
                log_placeholder.success(f"{event.side} {event.quantity} {live_symbol} @ Rs. {event.price:.2f}")

            if tick_delay:
                time.sleep(tick_delay)

        st.success("Live demo finished replaying held-back history.")
