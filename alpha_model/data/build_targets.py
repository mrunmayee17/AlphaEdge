"""Build sector-neutralized forward return targets for PatchTST training.

Targets are excess returns vs sector ETF at 4 horizons: 1d, 5d, 21d, 63d.
All targets use FORWARD-looking returns (shifted negative) — strictly out-of-sample.

XLC launched 2018-06-18. Pre-2018: use equal-weighted proxy of comm stocks.

Usage:
    python -m alpha_model.data.build_targets
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent / "processed"
RAW_DIR = Path(__file__).resolve().parent / "raw"

HORIZONS = [1, 5, 21, 63]

# GICS sector → ETF mapping
SECTOR_ETF_MAP = {
    "Technology": "XLK", "Information Technology": "XLK",
    "Financial Services": "XLF", "Financials": "XLF",
    "Healthcare": "XLV", "Health Care": "XLV",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Consumer Cyclical": "XLY", "Consumer Discretionary": "XLY",
    "Consumer Defensive": "XLP", "Consumer Staples": "XLP",
    "Basic Materials": "XLB", "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}

# XLC proxy tickers (pre-2018-06-18)
XLC_PROXY_TICKERS = ["GOOGL", "META", "NFLX", "DIS", "T", "VZ", "CMCSA"]
XLC_LAUNCH_DATE = pd.Timestamp("2018-06-18")


def compute_forward_returns(close: pd.Series, horizon: int) -> pd.Series:
    """Compute forward return over horizon days.

    forward_return_t = close_{t+horizon} / close_t - 1
    """
    return close.shift(-horizon) / close - 1


def build_xlc_proxy(prices_df: pd.DataFrame) -> pd.Series:
    """Build equal-weighted XLC proxy from constituent stocks for pre-2018 dates."""
    proxy_returns = []
    for ticker in XLC_PROXY_TICKERS:
        ticker_data = prices_df[prices_df["ticker"] == ticker]
        if ticker_data.empty:
            continue
        close = ticker_data.set_index("date")["close"].sort_index()
        ret = close.pct_change()
        proxy_returns.append(ret)

    if not proxy_returns:
        return pd.Series(dtype=float)

    # Equal-weighted daily return
    combined = pd.concat(proxy_returns, axis=1).mean(axis=1)
    # Convert to price level (cumulative return)
    proxy_price = (1 + combined).cumprod() * 100  # arbitrary starting level
    return proxy_price


def get_sector_etf_close(
    sector_etf: str,
    sector_etfs_df: pd.DataFrame,
    prices_df: pd.DataFrame,
) -> pd.Series:
    """Get sector ETF close prices, with XLC proxy for pre-2018."""
    etf_data = sector_etfs_df[sector_etfs_df["ticker"] == sector_etf]
    etf_close = etf_data.set_index("date")["close"].sort_index()

    if sector_etf == "XLC":
        # Build proxy for pre-launch dates
        proxy = build_xlc_proxy(prices_df)
        if not proxy.empty:
            pre_launch = proxy[proxy.index < XLC_LAUNCH_DATE]
            post_launch = etf_close[etf_close.index >= XLC_LAUNCH_DATE]
            etf_close = pd.concat([pre_launch, post_launch]).sort_index()

    return etf_close


def build_targets_for_ticker(
    ticker: str,
    ticker_close: pd.Series,
    sector_etf_close: pd.Series,
) -> pd.DataFrame | None:
    """Build sector-neutralized forward return targets for one ticker.

    Returns DataFrame with columns: alpha_1d, alpha_5d, alpha_21d, alpha_63d
    """
    if len(ticker_close) < 100:
        return None

    targets = pd.DataFrame(index=ticker_close.index)

    for horizon in HORIZONS:
        stock_fwd = compute_forward_returns(ticker_close, horizon)
        etf_fwd = compute_forward_returns(
            sector_etf_close.reindex(ticker_close.index, method="ffill"),
            horizon,
        )
        targets[f"alpha_{horizon}d"] = stock_fwd - etf_fwd

    return targets


def build_all_targets(
    prices_df: pd.DataFrame,
    sector_etfs_df: pd.DataFrame,
    ticker_sector_map: dict[str, str],
) -> pd.DataFrame:
    """Build targets for all tickers.

    Returns long-format DataFrame: (ticker, date, alpha_1d, alpha_5d, alpha_21d, alpha_63d)
    """
    all_targets = []
    skipped = []

    for ticker in prices_df["ticker"].unique():
        sector = ticker_sector_map.get(ticker)
        if not sector:
            skipped.append((ticker, "no sector mapping"))
            continue

        sector_etf = SECTOR_ETF_MAP.get(sector)
        if not sector_etf:
            skipped.append((ticker, f"no ETF for sector '{sector}'"))
            continue

        ticker_close = (
            prices_df[prices_df["ticker"] == ticker]
            .set_index("date")["close"]
            .sort_index()
        )

        etf_close = get_sector_etf_close(sector_etf, sector_etfs_df, prices_df)

        targets = build_targets_for_ticker(ticker, ticker_close, etf_close)
        if targets is None:
            skipped.append((ticker, "insufficient data"))
            continue

        targets["ticker"] = ticker
        all_targets.append(targets.reset_index())

    if skipped:
        logger.warning(f"Skipped {len(skipped)} tickers")
        for t, r in skipped[:10]:
            logger.warning(f"  {t}: {r}")

    if not all_targets:
        raise RuntimeError("No targets built — check sector mappings and data")

    result = pd.concat(all_targets, ignore_index=True)
    logger.info(
        f"Built targets: {result['ticker'].nunique()} tickers, "
        f"{len(result)} rows, horizons={HORIZONS}"
    )
    return result


TARGET_COLUMNS = [f"alpha_{h}d" for h in HORIZONS]


if __name__ == "__main__":
    import yfinance as yf

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prices = pd.read_parquet(RAW_DIR / "prices.parquet")
    sector_etfs = pd.read_parquet(RAW_DIR / "sector_etfs.parquet")

    # Build ticker → sector mapping from yfinance
    logger.info("Fetching sector mappings from yfinance...")
    ticker_sector_map = {}
    for ticker in prices["ticker"].unique():
        try:
            info = yf.Ticker(ticker).info
            sector = info.get("sector", "")
            if sector:
                ticker_sector_map[ticker] = sector
        except Exception:
            pass

    logger.info(f"Mapped {len(ticker_sector_map)} tickers to sectors")

    targets = build_all_targets(prices, sector_etfs, ticker_sector_map)
    targets.to_parquet(OUTPUT_DIR / "targets.parquet", index=False)
    logger.info(f"Saved to {OUTPUT_DIR / 'targets.parquet'}")
