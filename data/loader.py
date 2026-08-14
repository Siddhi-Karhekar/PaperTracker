"""
data/loader.py

Fetches, caches, and validates historical NSE equity OHLCV data.

Primary source: jugaad-data (wraps NSE's historical equity API).
Fallback source: yfinance with the ".NS" suffix, used automatically if the
NSE API errors out (it does this fairly often -- rate limiting, schema
changes, or the well-known requirement that a browser-like session hit the
NSE homepage before the data endpoint will respond).

Cache: one CSV per symbol under data/cache/. Each call only fetches the
date ranges not already on disk, then merges and re-saves.

Usage:
    from datetime import date
    from data.loader import fetch_ohlcv, validate_ohlcv

    df = fetch_ohlcv("SBIN", date(2020, 1, 1), date(2024, 12, 31))
    report = validate_ohlcv(df, "SBIN")
    print(report.summary())

Or from the command line:
    python -m data.loader SBIN --start 2020-01-01 --end 2024-12-31
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Standard schema used by the rest of the system, regardless of provider.
STANDARD_COLUMNS = ["date", "open", "high", "low", "close", "volume"]

# Raw provider column name (lowercased, punctuation stripped) -> standard name.
# jugaad-data's exact column casing has shifted across versions, so we match
# defensively instead of hardcoding one schema.
_COLUMN_ALIASES = {
    "date": "date",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "ltp": "close",  # used as a close fallback if CLOSE isn't present
    "volume": "volume",
    "totaltradedqty": "volume",
    "prevclose": "prev_close",
}


class DataFetchError(RuntimeError):
    """Raised when no provider could return data for a symbol/range."""


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map a raw provider DataFrame onto STANDARD_COLUMNS."""
    rename = {}
    for col in df.columns:
        key = "".join(ch for ch in str(col).lower() if ch.isalnum())
        if key in _COLUMN_ALIASES and _COLUMN_ALIASES[key] not in rename.values():
            rename[col] = _COLUMN_ALIASES[key]
    df = df.rename(columns=rename)

    missing = [c for c in STANDARD_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Provider response is missing expected columns {missing}. "
            f"Raw columns were: {list(df.columns)}"
        )

    keep = STANDARD_COLUMNS + (["prev_close"] if "prev_close" in df.columns else [])
    df = df[keep].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    numeric_cols = ["open", "high", "low", "close", "volume"] + (
        ["prev_close"] if "prev_close" in df.columns else []
    )
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["date"])
    df = df.sort_values("date").drop_duplicates(subset="date", keep="last").reset_index(drop=True)
    return df


def _cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol.upper()}.csv"


def _load_cache(symbol: str) -> pd.DataFrame | None:
    path = _cache_path(symbol)
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    return df if not df.empty else None


def _save_cache(symbol: str, df: pd.DataFrame) -> None:
    df.to_csv(_cache_path(symbol), index=False)


def _fetch_from_nse(symbol: str, start: date, end: date, series: str = "EQ") -> pd.DataFrame:
    from jugaad_data.nse import stock_df  # lazy import: network dependency

    raw = stock_df(symbol=symbol, from_date=start, to_date=end, series=series)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    return _normalize_columns(raw)


def _fetch_from_yfinance(symbol: str, start: date, end: date) -> pd.DataFrame:
    import yfinance as yf  # lazy import: network dependency

    ticker = f"{symbol.upper()}.NS"
    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    raw = raw.reset_index()
    # yfinance sometimes returns MultiIndex columns for a single ticker
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]
    return _normalize_columns(raw)


def _fetch_range(symbol: str, start: date, end: date, series: str = "EQ") -> pd.DataFrame:
    """Try NSE first, fall back to yfinance if it errors or returns nothing."""
    try:
        df = _fetch_from_nse(symbol, start, end, series)
        if not df.empty:
            return df
        logger.warning("NSE returned no rows for %s [%s, %s], trying yfinance fallback", symbol, start, end)
    except Exception as exc:  # noqa: BLE001 - deliberately broad, this is a network call
        logger.warning("NSE fetch failed for %s [%s, %s] (%s), trying yfinance fallback", symbol, start, end, exc)

    try:
        return _fetch_from_yfinance(symbol, start, end)
    except Exception as exc:  # noqa: BLE001
        logger.error("yfinance fallback also failed for %s [%s, %s]: %s", symbol, start, end, exc)
        return pd.DataFrame(columns=STANDARD_COLUMNS)


def fetch_ohlcv(
    symbol: str,
    start_date: date,
    end_date: date,
    series: str = "EQ",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Return a clean OHLCV DataFrame for `symbol` between start_date and
    end_date (inclusive), sourced from local cache where possible and
    the network for anything missing. Raises DataFetchError if nothing
    could be retrieved at all.
    """
    if start_date > end_date:
        raise ValueError(f"start_date {start_date} is after end_date {end_date}")

    cached = None if force_refresh else _load_cache(symbol)

    ranges_to_fetch: list[tuple[date, date]] = []
    if cached is None or cached.empty:
        ranges_to_fetch.append((start_date, end_date))
    else:
        cached_min = cached["date"].min().date()
        cached_max = cached["date"].max().date()
        if start_date < cached_min:
            ranges_to_fetch.append((start_date, cached_min))
        if end_date > cached_max:
            ranges_to_fetch.append((cached_max, end_date))

    fetched_frames = []
    for r_start, r_end in ranges_to_fetch:
        logger.info("Fetching %s: %s to %s", symbol, r_start, r_end)
        fetched_frames.append(_fetch_range(symbol, r_start, r_end, series))

    all_frames = [f for f in ([cached] if cached is not None else []) + fetched_frames if f is not None and not f.empty]
    if not all_frames:
        raise DataFetchError(f"No data returned for {symbol} between {start_date} and {end_date}")

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.sort_values("date").drop_duplicates(subset="date", keep="last").reset_index(drop=True)

    _save_cache(symbol, combined)

    mask = (combined["date"] >= pd.Timestamp(start_date)) & (combined["date"] <= pd.Timestamp(end_date))
    result = combined.loc[mask].reset_index(drop=True)
    if result.empty:
        raise DataFetchError(f"No data in requested range for {symbol}: {start_date} to {end_date}")
    return result


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    symbol: str
    n_rows: int
    missing_business_days: list = field(default_factory=list)
    duplicate_dates: list = field(default_factory=list)
    bad_rows: list = field(default_factory=list)  # (date, reason)
    potential_splits: list = field(default_factory=list)  # (date, close_ratio)

    @property
    def is_clean(self) -> bool:
        return not (self.missing_business_days or self.duplicate_dates or self.bad_rows)

    def summary(self) -> str:
        lines = [f"Validation report for {self.symbol}: {self.n_rows} rows"]
        lines.append(f"  Missing business days: {len(self.missing_business_days)}")
        lines.append(f"  Duplicate dates: {len(self.duplicate_dates)}")
        lines.append(f"  Bad rows: {len(self.bad_rows)}")
        lines.append(f"  Potential unadjusted splits: {len(self.potential_splits)}")
        return "\n".join(lines)


def validate_ohlcv(df: pd.DataFrame, symbol: str = "UNKNOWN") -> ValidationReport:
    """
    Flag common data-quality issues without mutating or auto-correcting
    anything: missing trading days, duplicate dates, nonsensical OHLC rows,
    and close-to-close jumps consistent with an unadjusted stock split.

    Note on missing days: this checks against a plain weekday calendar, not
    the actual NSE trading holiday calendar, so real NSE holidays will show
    up as expected/false-positive gaps. Good enough to catch real data
    problems; not a substitute for an exchange calendar if that matters
    later (e.g. for walk-forward validation in Phase 6).
    """
    report = ValidationReport(symbol=symbol, n_rows=len(df))
    if df.empty:
        return report

    d = df.sort_values("date").reset_index(drop=True)

    full_range = pd.bdate_range(d["date"].min(), d["date"].max())
    present = set(d["date"])
    report.missing_business_days = [dt.date() for dt in full_range if dt not in present]

    dup_mask = d["date"].duplicated(keep=False)
    report.duplicate_dates = sorted(set(d.loc[dup_mask, "date"].dt.date))

    for _, row in d.iterrows():
        reasons = []
        if row[["open", "high", "low", "close"]].le(0).any() or pd.isna(row[["open", "high", "low", "close"]]).any():
            reasons.append("non-positive or missing price")
        else:
            if row["high"] < row["low"]:
                reasons.append("high < low")
            if row["high"] < max(row["open"], row["close"]):
                reasons.append("high < open/close")
            if row["low"] > min(row["open"], row["close"]):
                reasons.append("low > open/close")
        if pd.notna(row["volume"]) and row["volume"] < 0:
            reasons.append("negative volume")
        if reasons:
            report.bad_rows.append((row["date"].date(), ", ".join(reasons)))

    close = d["close"]
    ratio = close / close.shift(1)
    for i in range(1, len(d)):
        r = ratio.iloc[i]
        if pd.isna(r):
            continue
        # Crude threshold: ordinary daily volatility rarely exceeds ~40%;
        # jumps beyond that are more consistent with a 1:2, 1:5, 1:10 style
        # split than a single bad trading day.
        if r <= 0.6 or r >= 1.8:
            report.potential_splits.append((d["date"].iloc[i].date(), round(float(r), 3)))

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch and validate NSE OHLCV data.")
    parser.add_argument("symbol", help="NSE symbol, e.g. SBIN, RELIANCE, TCS")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()

    data = fetch_ohlcv(args.symbol, start, end, force_refresh=args.force_refresh)
    print(data.head())
    print(f"... {len(data)} rows total")

    report = validate_ohlcv(data, args.symbol)
    print(report.summary())
