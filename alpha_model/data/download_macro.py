"""Download macro market data: sector ETFs, VIX, bonds, commodities, FRED yields.

Usage:
    python -m alpha_model.data.download_macro
"""

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf
from fredapi import Fred

from backend.app.config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent / "raw"
START_DATE = "2009-01-01"
END_DATE = "2026-03-16"

# Sector ETFs (11 GICS sectors)
SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLB", "XLU", "XLRE", "XLC"]

# Macro tickers
MACRO_TICKERS = ["SPY", "^VIX", "TLT", "GLD", "HYG", "LQD"]

# FRED series
FRED_SERIES = {
    "DGS10": "10Y Treasury Yield",
    "DGS2": "2Y Treasury Yield",
    "DGS3MO": "3M Treasury Yield",
    "FEDFUNDS": "Federal Funds Rate",
}


def download_yfinance_macro(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download daily close prices for macro tickers."""
    data = yf.download(tickers, start=start, end=end, group_by="ticker", progress=False)

    records = []
    for ticker in tickers:
        try:
            if ticker in data.columns.get_level_values(0):
                close = data[ticker]["Close"].dropna()
                for date, value in close.items():
                    records.append({"date": date, "ticker": ticker, "close": float(value)})
        except Exception as e:
            logger.warning(f"  {ticker} failed: {e}")

    df = pd.DataFrame(records)
    logger.info(f"yfinance macro: {df['ticker'].nunique()} tickers, {len(df)} rows")
    return df


def download_fred_yields(start: str) -> pd.DataFrame:
    """Download FRED yield data."""
    settings = get_settings()
    fred = Fred(api_key=settings.fred_api_key)

    records = []
    for series_id, description in FRED_SERIES.items():
        try:
            data = fred.get_series(series_id, observation_start=start)
            data = data.dropna()
            for date, value in data.items():
                records.append({"date": date, "series": series_id, "value": float(value)})
            logger.info(f"  {series_id}: {len(data)} obs")
        except Exception as e:
            logger.error(f"  {series_id} FAILED: {e}")

    df = pd.DataFrame(records)
    # Pivot: one column per series
    if not df.empty:
        df = df.pivot(index="date", columns="series", values="value").reset_index()
    return df


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Sector ETFs
    logger.info("Downloading sector ETFs...")
    sector_data = download_yfinance_macro(SECTOR_ETFS, START_DATE, END_DATE)
    sector_data.to_parquet(OUTPUT_DIR / "sector_etfs.parquet", index=False)

    # Macro tickers
    logger.info("Downloading macro tickers...")
    macro_data = download_yfinance_macro(MACRO_TICKERS, START_DATE, END_DATE)
    macro_data.to_parquet(OUTPUT_DIR / "macro_tickers.parquet", index=False)

    # FRED yields
    logger.info("Downloading FRED yields...")
    fred_data = download_fred_yields(START_DATE)
    fred_data.to_parquet(OUTPUT_DIR / "fred_yields.parquet", index=False)

    logger.info("All macro data downloaded")
