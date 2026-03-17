"""Download Fama-French 5 factors + Momentum from Ken French's website.

Data is in percentage — we divide by 100 to get decimal returns.

Usage:
    python -m alpha_model.data.download_factors
"""

import io
import logging
import zipfile
from pathlib import Path

import httpx
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent / "raw"

FF5_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
MOM_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"


def download_and_parse_ff_zip(url: str) -> pd.DataFrame:
    """Download a French data zip, extract CSV, parse into DataFrame."""
    logger.info(f"Downloading {url.split('/')[-1]}...")
    resp = httpx.get(url, timeout=30.0, follow_redirects=True)
    resp.raise_for_status()

    z = zipfile.ZipFile(io.BytesIO(resp.content))
    fname = z.namelist()[0]
    content = z.read(fname).decode("utf-8")

    # Parse: find data rows (start with YYYYMMDD digit pattern)
    lines = content.split("\n")
    data_lines = []
    header = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Header line (contains column names, comes before data)
        if stripped[0].isalpha() and "," in stripped:
            header = [c.strip() for c in stripped.split(",")]
            continue
        # Data lines start with digits (YYYYMMDD)
        if stripped[:4].isdigit():
            data_lines.append(stripped)
        # Stop at footer (Annual/Copyright notices)
        elif data_lines and not stripped[0].isdigit():
            break

    if not data_lines:
        raise ValueError(f"No data lines found in {fname}")

    # Parse into DataFrame
    rows = []
    for line in data_lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            rows.append(parts)

    df = pd.DataFrame(rows)
    # First column is date
    df[0] = pd.to_datetime(df[0], format="%Y%m%d")
    df = df.rename(columns={0: "date"})

    # Convert remaining columns to float and divide by 100
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce") / 100.0

    return df


def download_ff5_factors() -> pd.DataFrame:
    """Download Fama-French 5 factors (daily)."""
    df = download_and_parse_ff_zip(FF5_URL)
    # Expected columns: date, Mkt-RF, SMB, HML, RMW, CMA, RF
    if len(df.columns) == 7:
        df.columns = ["date", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    logger.info(f"FF5: {len(df)} rows, {df['date'].min()} to {df['date'].max()}")
    return df


def download_momentum_factor() -> pd.DataFrame:
    """Download Momentum factor (daily)."""
    df = download_and_parse_ff_zip(MOM_URL)
    if len(df.columns) == 2:
        df.columns = ["date", "Mom"]
    elif len(df.columns) >= 2:
        df = df.iloc[:, :2]
        df.columns = ["date", "Mom"]
    logger.info(f"Momentum: {len(df)} rows, {df['date'].min()} to {df['date'].max()}")
    return df


def download_all_factors() -> pd.DataFrame:
    """Download and merge FF5 + Momentum."""
    ff5 = download_ff5_factors()
    mom = download_momentum_factor()

    # Merge on date
    factors = ff5.merge(mom, on="date", how="outer").sort_values("date").reset_index(drop=True)
    logger.info(f"Combined factors: {len(factors)} rows, columns={list(factors.columns)}")
    return factors


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    factors = download_all_factors()
    output_path = OUTPUT_DIR / "ff_factors.parquet"
    factors.to_parquet(output_path, index=False)
    logger.info(f"Saved to {output_path}")
