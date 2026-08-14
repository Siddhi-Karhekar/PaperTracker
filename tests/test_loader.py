"""
Unit tests for data/loader.py's validation logic.

These use synthetic DataFrames only -- no network calls -- so they run
fast and deterministically in CI or offline. Fetch-path integration
(actually hitting NSE/yfinance) is smoke-tested manually per the README,
not covered here.
"""

from datetime import date

import pandas as pd
import pytest

from data.loader import _normalize_columns, validate_ohlcv


def _make_clean_df(n_days: int = 10, start: str = "2024-01-01") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=n_days)
    close = [100 + i for i in range(n_days)]
    rows = []
    for dt, c in zip(dates, close):
        rows.append(
            {
                "date": dt,
                "open": c - 0.5,
                "high": c + 1,
                "low": c - 1,
                "close": c,
                "volume": 1000 + 10 * len(rows),
            }
        )
    df = pd.DataFrame(rows)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df


def test_validate_clean_data_has_no_flags():
    df = _make_clean_df()
    report = validate_ohlcv(df, "TEST")
    assert report.is_clean
    assert report.n_rows == len(df)
    assert report.missing_business_days == []
    assert report.duplicate_dates == []
    assert report.bad_rows == []


def test_validate_detects_missing_business_day():
    df = _make_clean_df(n_days=10)
    df = df.drop(index=5).reset_index(drop=True)  # remove one weekday in the middle
    report = validate_ohlcv(df, "TEST")
    assert len(report.missing_business_days) == 1


def test_validate_detects_duplicate_dates():
    df = _make_clean_df(n_days=5)
    dup = df.iloc[[2]].copy()
    df = pd.concat([df, dup], ignore_index=True)
    report = validate_ohlcv(df, "TEST")
    assert len(report.duplicate_dates) == 1


def test_validate_detects_high_less_than_low():
    df = _make_clean_df(n_days=5)
    df.loc[2, "high"] = df.loc[2, "low"] - 1
    report = validate_ohlcv(df, "TEST")
    assert any("high < low" in reason for _, reason in report.bad_rows)


def test_validate_detects_non_positive_price():
    df = _make_clean_df(n_days=5)
    df.loc[1, "close"] = 0
    report = validate_ohlcv(df, "TEST")
    assert any(d == df.loc[1, "date"].date() for d, _ in report.bad_rows)


def test_validate_detects_negative_volume():
    df = _make_clean_df(n_days=5)
    df.loc[3, "volume"] = -100
    report = validate_ohlcv(df, "TEST")
    assert any("negative volume" in reason for _, reason in report.bad_rows)


def test_validate_flags_potential_split():
    df = _make_clean_df(n_days=6)
    # Simulate a 1:2 split: from day 3 onward, prices continue trending from
    # a halved base (the post-split regime), not just one anomalous day --
    # that's what an unadjusted split actually looks like in raw data.
    split_ratio = 2.0
    for i in range(3, len(df)):
        for col in ["open", "high", "low", "close"]:
            df.loc[i, col] = df.loc[i, col] / split_ratio
    report = validate_ohlcv(df, "TEST")
    assert len(report.potential_splits) == 1
    assert report.potential_splits[0][0] == df.loc[3, "date"].date()


def test_validate_empty_dataframe_returns_empty_report():
    df = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    report = validate_ohlcv(df, "TEST")
    assert report.n_rows == 0
    assert report.is_clean


def test_normalize_columns_maps_common_aliases():
    raw = pd.DataFrame(
        {
            "DATE": ["01-Jan-2024", "02-Jan-2024"],
            "OPEN": [100, 101],
            "HIGH": [102, 103],
            "LOW": [99, 100],
            "CLOSE": [101, 102],
            "VOLUME": [1000, 1100],
        }
    )
    normalized = _normalize_columns(raw)
    assert list(normalized.columns[:6]) == ["date", "open", "high", "low", "close", "volume"]
    assert normalized["date"].iloc[0] == pd.Timestamp("2024-01-01")


def test_normalize_columns_raises_on_missing_fields():
    raw = pd.DataFrame({"DATE": ["01-Jan-2024"], "OPEN": [100]})
    with pytest.raises(ValueError):
        _normalize_columns(raw)
