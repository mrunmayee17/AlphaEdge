"""Download S&P 500 constituents — current list + historical changes.

Two approaches:
1. Current Wikipedia scrape for latest constituents
2. Wikipedia "changes" table for historical additions/removals → reconstruct at any date

This is more reliable than Wayback Machine (which has timeout issues).

Usage:
    python -m alpha_model.data.download_universe
"""

import logging
from datetime import datetime
from pathlib import Path

import httpx
import pandas as pd
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
OUTPUT_DIR = Path(__file__).resolve().parent / "raw"


def get_current_sp500() -> tuple[list[str], pd.DataFrame]:
    """Scrape current S&P 500 list and historical changes from Wikipedia.

    Returns:
        current_tickers: list of current S&P 500 tickers
        changes_df: DataFrame with columns [date, added, removed]
    """
    headers = {"User-Agent": "BAM/1.0 (research project; Python httpx)"}
    # Use pandas read_html which handles Wikipedia tables reliably
    tables = pd.read_html(WIKI_URL, storage_options={"User-Agent": "Mozilla/5.0"})

    # Table 0: Current constituents (has 'Symbol' column)
    constituents_df = tables[0]
    symbol_col = [c for c in constituents_df.columns if "symbol" in c.lower() or "ticker" in c.lower()][0]
    current_tickers = (
        constituents_df[symbol_col]
        .str.strip()
        .str.replace(".", "-", regex=False)
        .tolist()
    )
    current_tickers = [t for t in current_tickers if t and len(t) <= 10]
    logger.info(f"Current S&P 500: {len(current_tickers)} constituents")

    # Table 1: Historical changes
    changes = []
    if len(tables) >= 2:
        changes_df_raw = tables[1]
        # Find date, added, removed columns
        cols = changes_df_raw.columns.tolist()
        date_col = cols[0]
        added_col = cols[1] if len(cols) > 1 else None
        removed_col = cols[3] if len(cols) > 3 else None

        for _, row in changes_df_raw.iterrows():
            try:
                date = pd.to_datetime(row[date_col])
                added = str(row[added_col]).strip().replace(".", "-") if added_col else None
                removed = str(row[removed_col]).strip().replace(".", "-") if removed_col else None
                if added == "nan":
                    added = None
                if removed == "nan":
                    removed = None
                changes.append({"date": date, "added": added, "removed": removed})
            except Exception:
                continue

    changes_df = pd.DataFrame(changes)
    if not changes_df.empty:
        changes_df = changes_df.sort_values("date").reset_index(drop=True)
        logger.info(f"Historical changes: {len(changes_df)} events "
                    f"({changes_df['date'].min().strftime('%Y-%m-%d')} to "
                    f"{changes_df['date'].max().strftime('%Y-%m-%d')})")

    return current_tickers, changes_df


def reconstruct_at_date(
    current_tickers: list[str],
    changes_df: pd.DataFrame,
    target_date: str,
) -> list[str]:
    """Reconstruct S&P 500 constituents at a past date.

    Starts from current list and reverses changes back to target_date.
    """
    target = pd.to_datetime(target_date)
    tickers = set(current_tickers)

    if changes_df.empty:
        return sorted(tickers)

    # Walk backwards through changes: reverse each change after target_date
    future_changes = changes_df[changes_df["date"] > target].sort_values("date", ascending=False)

    for _, row in future_changes.iterrows():
        added = row["added"]
        removed = row["removed"]
        # Reverse: if it was added after target, remove it
        if added and isinstance(added, str) and added in tickers:
            tickers.discard(added)
        # Reverse: if it was removed after target, add it back
        if removed and isinstance(removed, str):
            tickers.add(removed)

    # Clean up any non-string entries
    tickers = {t for t in tickers if isinstance(t, str) and t.strip()}
    return sorted(tickers)


def build_yearly_constituents(
    current_tickers: list[str],
    changes_df: pd.DataFrame,
    start_year: int = 2010,
    end_year: int = 2024,
) -> pd.DataFrame:
    """Build yearly constituent snapshots."""
    all_records = []

    for year in range(start_year, end_year + 1):
        target = f"{year}-01-15"
        tickers = reconstruct_at_date(current_tickers, changes_df, target)
        logger.info(f"  {year}: {len(tickers)} constituents")

        for ticker in tickers:
            all_records.append({"year": year, "ticker": ticker})

    return pd.DataFrame(all_records)


def get_all_unique_tickers(df: pd.DataFrame) -> list[str]:
    """Get union of all tickers across all years for bulk download."""
    return sorted(df["ticker"].unique().tolist())


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    current, changes = get_current_sp500()

    df = build_yearly_constituents(current, changes)
    output_path = OUTPUT_DIR / "sp500_constituents.parquet"
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(df)} records to {output_path}")

    # Save changes table
    if not changes.empty:
        changes.to_parquet(OUTPUT_DIR / "sp500_changes.parquet", index=False)

    unique = get_all_unique_tickers(df)
    logger.info(f"Total unique tickers across all years: {len(unique)}")

    pd.DataFrame({"ticker": unique}).to_parquet(
        OUTPUT_DIR / "sp500_unique_tickers.parquet", index=False
    )
