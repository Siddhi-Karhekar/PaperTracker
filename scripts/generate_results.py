"""
scripts/generate_results.py

Reproducibility script for the README's "Results" section: fetches real
NSE history for one symbol, runs a full-period backtest, runs walk-forward
validation on the same data, prints both reports, and saves the equity
and drawdown charts to reports/. Run this yourself and use the printed
numbers -- don't hand-write results into a README.

Usage:
    python -m scripts.generate_results
    python -m scripts.generate_results --symbol TCS --years 5
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta

from data.loader import fetch_ohlcv, validate_ohlcv
from engine.execution import CostModel
from engine.portfolio import run_backtest
from engine.validation import walk_forward_validate
from reporting.metrics import generate_report, plot_drawdown, plot_equity_curve
from strategy.moving_average import MovingAverageCrossover


def main():
    parser = argparse.ArgumentParser(description="Generate real backtest + walk-forward results for the README.")
    parser.add_argument("--symbol", default="SBIN")
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--fast-window", type=int, default=50)
    parser.add_argument("--slow-window", type=int, default=200)
    parser.add_argument("--capital", type=float, default=100_000.0)
    args = parser.parse_args()

    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=365 * args.years)

    print(f"Fetching {args.symbol} from {start} to {end}...")
    df = fetch_ohlcv(args.symbol, start, end)
    print(validate_ohlcv(df, args.symbol).summary())
    print(f"{len(df)} rows fetched.\n")

    # --- Full-period backtest (one parameter set, fit once to everything) ---
    strategy = MovingAverageCrossover(fast_window=args.fast_window, slow_window=args.slow_window)
    signal = strategy.run(df)
    result = run_backtest(df, signal, cost_model=CostModel(), initial_capital=args.capital)
    report = generate_report(result.equity_curve, result.trades, df, args.capital)

    print("=" * 70)
    print(f"FULL-PERIOD BACKTEST -- {args.symbol}, {strategy.name}, {start} to {end}")
    print("=" * 70)
    print(report.summary())
    print()

    plot_equity_curve(result.equity_curve, df, args.capital, output_path="reports/equity.png")
    plot_drawdown(result.equity_curve, output_path="reports/drawdown.png")
    print("Saved reports/equity.png and reports/drawdown.png\n")

    # --- Walk-forward validation (the honest, out-of-sample number) ---
    param_grid = [
        {"fast_window": f, "slow_window": s}
        for f, s in {(20, 100), (50, 200), (30, 150), (args.fast_window, args.slow_window)}
        if f < s
    ]
    train_days, test_days = 252 * 2, 126
    print("=" * 70)
    print(f"WALK-FORWARD VALIDATION -- train={train_days}d, test={test_days}d, params tried={param_grid}")
    print("=" * 70)
    try:
        wfv = walk_forward_validate(
            df,
            param_grid=param_grid,
            strategy_factory=lambda **p: MovingAverageCrossover(**p),
            train_days=train_days,
            test_days=test_days,
            warmup_days=max(s for _, s in [(p["fast_window"], p["slow_window"]) for p in param_grid]),
            cost_model=CostModel(),
            initial_capital=args.capital,
        )
        print(wfv.summary_table().to_string(index=False))
        print()
        if wfv.out_of_sample_report is not None:
            print("Out-of-sample (stitched across all windows):")
            print(wfv.out_of_sample_report.summary())
    except ValueError as exc:
        print(f"Walk-forward validation skipped: {exc}")

    print("\nDone. Copy the numbers above into the README's Results section.")


if __name__ == "__main__":
    main()
