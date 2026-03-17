"""Download OHLCV price data for all S&P 500 constituents via yfinance.

Downloads from 2009-01-01 (250d warm-up before 2010 training start).
Forward-fills max 2 days, drops tickers with insufficient data.

Usage:
    python -m alpha_model.data.download_prices
"""

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent / "raw"
START_DATE = "2009-01-01"
END_DATE = "2026-03-16"
MIN_ROWS = 100  # Skip tickers with fewer rows (delisted/insufficient)
MAX_FFILL_DAYS = 2  # Forward-fill limit (was 5, tightened per review)


def download_prices(tickers: list[str], start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    """Bulk download OHLCV for all tickers.

    Downloads in chunks of 50 to avoid yfinance rate limits.
    """
    chunk_size = 50
    all_data = []

    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        logger.info(f"Downloading chunk {i // chunk_size + 1}/{(len(tickers) + chunk_size - 1) // chunk_size}: "
                    f"{len(chunk)} tickers")

        try:
            data = yf.download(
                chunk,
                start=start,
                end=end,
                group_by="ticker",
                progress=False,
                threads=True,
            )
            all_data.append(data)
        except Exception as e:
            logger.error(f"Chunk {i // chunk_size + 1} failed: {e}")
            # Try individual downloads as fallback
            for ticker in chunk:
                try:
                    single = yf.download(ticker, start=start, end=end, progress=False)
                    if not single.empty:
                        # Add ticker level to match group_by format
                        single.columns = pd.MultiIndex.from_product([[ticker], single.columns])
                        all_data.append(single)
                except Exception as e2:
                    logger.warning(f"  {ticker} failed: {e2}")

    if not all_data:
        raise RuntimeError("No price data downloaded")

    # Combine all chunks
    combined = pd.concat(all_data, axis=1)
    return combined


def process_prices(raw_data: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Process raw multi-ticker data into clean long-format DataFrame.

    - Forward-fills max 2 days
    - Drops tickers with <100 rows
    - Returns long format: (ticker, date, open, high, low, close, volume)
    """
    records = []
    skipped = []

    for ticker in tickers:
        try:
            if ticker in raw_data.columns.get_level_values(0):
                df = raw_data[ticker].copy()
            else:
                skipped.append((ticker, "not in downloaded data"))
                continue

            if df.empty or len(df.dropna(subset=["Close"])) < MIN_ROWS:
                skipped.append((ticker, f"insufficient data ({len(df)} rows)"))
                continue

            # Forward-fill max 2 days
            df = df.ffill(limit=MAX_FFILL_DAYS)

            # Drop remaining NaN rows
            df = df.dropna(subset=["Close"])

            if len(df) < MIN_ROWS:
                skipped.append((ticker, f"insufficient after ffill ({len(df)} rows)"))
                continue

            df = df.reset_index()
            df["ticker"] = ticker
            df = df.rename(columns={"Date": "date", "Open": "open", "High": "high",
                                     "Low": "low", "Close": "close", "Volume": "volume"})
            records.append(df[["ticker", "date", "open", "high", "low", "close", "volume"]])

        except Exception as e:
            skipped.append((ticker, str(e)))

    if skipped:
        logger.warning(f"Skipped {len(skipped)} tickers:")
        for ticker, reason in skipped[:20]:
            logger.warning(f"  {ticker}: {reason}")
        if len(skipped) > 20:
            logger.warning(f"  ... and {len(skipped) - 20} more")

    result = pd.concat(records, ignore_index=True)
    logger.info(f"Processed {result['ticker'].nunique()} tickers, {len(result)} total rows")
    return result


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load universe
    universe_path = OUTPUT_DIR / "sp500_unique_tickers.parquet"
    if not universe_path.exists():
        logger.info("Universe file not found, running download_universe first...")
        from alpha_model.data.download_universe import get_current_sp500, build_yearly_constituents, get_all_unique_tickers
        current, changes = get_current_sp500()
        df = build_yearly_constituents(current, changes)
        df.to_parquet(OUTPUT_DIR / "sp500_constituents.parquet", index=False)
        tickers = get_all_unique_tickers(df)
        pd.DataFrame({"ticker": tickers}).to_parquet(universe_path, index=False)
    else:
        tickers = pd.read_parquet(universe_path)["ticker"].tolist()

    logger.info(f"Downloading prices for {len(tickers)} tickers...")
    raw = download_prices(tickers)

    logger.info("Processing prices...")
    prices = process_prices(raw, tickers)

    output_path = OUTPUT_DIR / "prices.parquet"
    prices.to_parquet(output_path, index=False)
    logger.info(f"Saved to {output_path}")
