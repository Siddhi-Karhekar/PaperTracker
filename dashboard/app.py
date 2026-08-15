"""
dashboard/app.py

Streamlit dashboard: pick a symbol, date range, and strategy (rule-based
MA crossover or an ML classifier), run a backtest through the real cost
model / next-open execution / cash-aware portfolio sizing from Phases
3-4 (optionally with ML volatility-scaled position sizing), see the
equity curve against buy-and-hold, the metrics table, the trade log, a
SHAP explainability panel when the ML strategy is used, a natural-language
Q&A panel over the actual results, optional walk-forward validation
(Phase 6), and a simulated live paper-trading ticker (Phase 7). This is
the thing to actually demo in an interview -- a working link beats a
GitHub repo nobody runs.

Run with:
    streamlit run dashboard/app.py

*** The live panel below replays real held-back historical data as a
simulated tick stream by default, and is clearly labeled PAPER TRADING --
NOT REAL FUNDS throughout. No real broker connection, no real orders,
ever. ***
"""

from __future__ import annotations

import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

# Streamlit doesn't reliably put the project root on sys.path (it depends
# on the working directory `streamlit run` was launched from), so the
# project's own packages (data, engine, strategy, ...) can fail to import
# even when run correctly. Force it explicitly rather than relying on cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.graph_objects as go
import streamlit as st

from data.loader import DataFetchError, fetch_ohlcv, validate_ohlcv
from engine.execution import CostModel
from engine.portfolio import run_backtest
from engine.validation import walk_forward_validate
from live.feed import Tick
from live.paper_trader import PAPER_TRADING_BANNER, PaperTrader
from ml.explain import explain as shap_explain
from ml.position_sizing import VolatilityForecaster, inverse_vol_position_size
from nlp.qa_engine import QAEngine
from reporting.metrics import buy_and_hold_benchmark, generate_report
from strategy.ml_signal import MLClassifierStrategy
from strategy.moving_average import MovingAverageCrossover

st.set_page_config(page_title="Algo Backtester", layout="wide")

STRATEGIES = {
    "Moving Average Crossover": MovingAverageCrossover,
    "ML Classifier (Random Forest)": MLClassifierStrategy,
}


@st.cache_data(show_spinner=False)
def _load_data(symbol: str, start: date, end: date):
    return fetch_ohlcv(symbol, start, end)


def _build_strategy(name: str, **params):
    """Constructs the selected Strategy with only the kwargs it actually accepts."""
    if name == "ML Classifier (Random Forest)":
        return MLClassifierStrategy(
            min_train_rows=params["ml_min_train_rows"],
            forward_horizon=params["ml_forward_horizon"],
            prob_threshold=params["ml_prob_threshold"],
        )
    return MovingAverageCrossover(fast_window=params["fast_window"], slow_window=params["slow_window"])


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
is_ml_strategy = strategy_name == "ML Classifier (Random Forest)"

if is_ml_strategy:
    st.sidebar.caption("Predicts next-N-day direction from technical features (RSI, MACD, moving averages, volatility, ...). See the Limitations section in the README before trusting this over the rule-based strategy.")
    ml_min_train_rows = st.sidebar.number_input("Training rows (warm-up + fit)", min_value=60, max_value=2000, value=252)
    ml_forward_horizon = st.sidebar.number_input("Prediction horizon (days)", min_value=1, max_value=60, value=5)
    ml_prob_threshold = st.sidebar.slider("Confidence threshold to go long", min_value=0.51, max_value=0.95, value=0.55)
    fast_window, slow_window = 50, 200  # unused for the ML strategy, kept defined for the walk-forward param grid below
else:
    fast_window = st.sidebar.number_input("Fast window", min_value=2, max_value=400, value=50)
    slow_window = st.sidebar.number_input("Slow window", min_value=3, max_value=500, value=200)

st.sidebar.subheader("Portfolio")
initial_capital = st.sidebar.number_input("Initial capital (Rs.)", min_value=1000.0, value=100_000.0, step=10_000.0)
use_ml_sizing = st.sidebar.checkbox("Use ML volatility-scaled position sizing", value=False, help="Predicts near-term volatility and sizes positions inversely to it, instead of always committing the same fixed %.")
position_size_pct = st.sidebar.slider("Position size (% of cash per entry)", min_value=10, max_value=100, value=95, disabled=use_ml_sizing) / 100.0

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

    strategy_kwargs = dict(fast_window=fast_window, slow_window=slow_window)
    if is_ml_strategy:
        strategy_kwargs = dict(
            ml_min_train_rows=ml_min_train_rows, ml_forward_horizon=ml_forward_horizon, ml_prob_threshold=ml_prob_threshold
        )
    strategy = _build_strategy(strategy_name, **strategy_kwargs)
    signal = strategy.run(df)

    sizing = position_size_pct
    if use_ml_sizing:
        with st.spinner("Forecasting volatility for position sizing..."):
            forecaster = VolatilityForecaster(min_train_rows=ml_min_train_rows if is_ml_strategy else 252)
            predicted_vol = forecaster.fit_predict(df)
            sizing = inverse_vol_position_size(predicted_vol, target_vol=0.02, min_size=0.10, max_size=0.95)

    result = run_backtest(df, signal, cost_model=CostModel(), initial_capital=initial_capital, position_size_pct=sizing)
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

    # --- Explainable AI: SHAP feature importance (ML strategy only) ---
    if is_ml_strategy and getattr(strategy, "last_model", None) is not None:
        st.subheader("Why the model predicted what it did (SHAP)")
        st.caption(
            "SHAP values decompose each prediction into a per-feature contribution: expected_value + "
            "sum(shap values) reconstructs the model's actual predicted probability exactly (verified in this "
            "project's own tests, not just assumed)."
        )
        predicted_rows = strategy.last_features.loc[
            (strategy.last_features.index >= strategy.last_train_end) & strategy.last_features.notna().all(axis=1)
        ]
        if predicted_rows.empty:
            st.info("No genuinely out-of-sample predictions to explain (not enough rows after the training window).")
        else:
            explanation = shap_explain(strategy.last_model, predicted_rows)

            importance = explanation.global_importance()
            imp_fig = go.Figure(go.Bar(x=importance.values, y=importance.index, orientation="h"))
            imp_fig.update_layout(
                title="Global feature importance (mean |SHAP value| across all predicted days)",
                xaxis_title="Mean |SHAP value|", height=400, yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(imp_fig, use_container_width=True)

            latest_date = df.loc[predicted_rows.index[-1], "date"]
            top = explanation.top_features(row_index=-1, n=6)
            top_fig = go.Figure(go.Bar(
                x=top.values, y=top.index, orientation="h",
                marker_color=["seagreen" if v > 0 else "firebrick" for v in top.values],
            ))
            top_fig.update_layout(
                title=f"Why the model's most recent prediction ({latest_date.date()}) was what it was",
                xaxis_title="SHAP value (pushes prediction up / down)", height=320, yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(top_fig, use_container_width=True)

    # --- NLP chatbot over the actual results ---
    st.subheader("Ask about these results")
    if os.environ.get("ANTHROPIC_API_KEY"):
        st.caption("Answered by Claude Haiku, grounded on this run's real numbers -- fast, 1-2 sentence answers, open-ended questions welcome.")
    else:
        st.caption(
            "Set ANTHROPIC_API_KEY for fast, open-ended LLM answers (Claude Haiku). Without it, this runs a "
            "free offline Q&A over a fixed set of questions: return, Sharpe, drawdown, win rate, turnover, "
            "trade count, or what any of those terms mean."
        )

    qa_engine = QAEngine(report=report, trades=result.trades, symbol=active_symbol)
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    question = st.text_input("Ask a question about this backtest", key="chat_question")
    if st.button("Ask") and question.strip():
        answer = qa_engine.answer(question)
        st.session_state["chat_history"].append((question, answer))

    for q, a in reversed(st.session_state["chat_history"][-5:]):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Answer:** {a}")

    if run_wfv and is_ml_strategy:
        st.subheader("Walk-forward validation")
        st.info(
            "Walk-forward validation in this dashboard is wired up for the Moving Average Crossover strategy "
            "only (grid search x rolling windows x model refit gets slow for the ML strategy in an interactive "
            "UI). engine.validation.walk_forward_validate supports any Strategy generically via strategy_factory "
            "-- run it from a script for the ML strategy, e.g. scripts/generate_results.py as a template."
        )
    elif run_wfv:
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
        live_strategy_kwargs = dict(fast_window=fast_window, slow_window=slow_window)
        if is_ml_strategy:
            live_strategy_kwargs = dict(
                ml_min_train_rows=min(ml_min_train_rows, max(len(warmup_df) - 5, 10)),
                ml_forward_horizon=ml_forward_horizon, ml_prob_threshold=ml_prob_threshold,
            )
        live_strategy = _build_strategy(strategy_name, **live_strategy_kwargs)
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
