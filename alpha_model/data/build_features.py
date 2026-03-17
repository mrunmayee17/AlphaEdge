"""Feature engineering pipeline — builds 23-channel time series features.

All features are strictly point-in-time (no lookahead).

Channels:
  1-4:   Raw OHLC as % change from prior close
  5-6:   Volume features (log volume, volume ratio)
  7-9:   Volatility (realized 21d, Garman-Klass, vol-of-vol)
  10-15: Factor betas (rolling 63d OLS: Mkt-RF, SMB, HML, RMW, CMA, Mom)
  16-21: Cross-asset (SPY return, VIX level, VIX change, 10Y yield change, HYG-LQD spread change, TLT return)
  22-23: Sector-neutralized returns (1d and 5d) — prevents sector beta leakage

Usage:
    python -m alpha_model.data.build_features
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent / "processed"
RAW_DIR = Path(__file__).resolve().parent / "raw"

# Factor beta rolling window
FACTOR_WINDOW = 63
FACTOR_MIN_PERIODS = 50  # Require 50/63 valid obs (review fix C3)


def build_ohlc_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Channels 1-4: OHLC as % change from prior close."""
    df = prices.copy()
    prev_close = df["close"].shift(1)
    df["pct_open"] = (df["open"] - prev_close) / prev_close
    df["pct_high"] = (df["high"] - prev_close) / prev_close
    df["pct_low"] = (df["low"] - prev_close) / prev_close
    df["pct_close"] = (df["close"] - prev_close) / prev_close  # daily return
    return df[["pct_open", "pct_high", "pct_low", "pct_close"]]


def build_volume_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Channels 5-6: Log volume and volume ratio."""
    df = prices.copy()
    df["log_volume"] = np.log(df["volume"] + 1)
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(20, min_periods=10).mean()
    return df[["log_volume", "volume_ratio"]]


def build_volatility_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Channels 7-9: Realized vol, Garman-Klass vol, vol-of-vol."""
    df = prices.copy()
    returns = df["close"].pct_change()

    # Realized volatility (21d)
    df["realized_vol_21d"] = returns.rolling(21, min_periods=15).std() * np.sqrt(252)

    # Garman-Klass volatility
    log_hl = np.log(df["high"] / df["low"])
    log_co = np.log(df["close"] / df["open"])
    gk_daily = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
    df["garman_klass_vol"] = np.sqrt(252 * gk_daily.rolling(21, min_periods=15).mean())

    # Vol of vol
    df["vol_of_vol"] = df["realized_vol_21d"].rolling(21, min_periods=15).std()

    return df[["realized_vol_21d", "garman_klass_vol", "vol_of_vol"]]


def build_factor_betas(
    returns: pd.Series,
    factors: pd.DataFrame,
    risk_free: pd.Series,
) -> pd.DataFrame:
    """Channels 10-15: Rolling 63d OLS factor betas."""
    factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
    excess_returns = returns - risk_free.reindex(returns.index).fillna(0)

    betas = pd.DataFrame(index=returns.index)

    for factor in factor_cols:
        if factor not in factors.columns:
            betas[f"beta_{factor}"] = np.nan
            continue

        factor_series = factors[factor].reindex(returns.index).fillna(0)

        # Rolling OLS: beta = cov(r, f) / var(f)
        cov = excess_returns.rolling(FACTOR_WINDOW, min_periods=FACTOR_MIN_PERIODS).cov(factor_series)
        var = factor_series.rolling(FACTOR_WINDOW, min_periods=FACTOR_MIN_PERIODS).var()
        betas[f"beta_{factor}"] = cov / var.replace(0, np.nan)

    return betas


def build_cross_asset_features(
    dates: pd.DatetimeIndex,
    macro_data: dict[str, pd.Series],
) -> pd.DataFrame:
    """Channels 16-21: Cross-asset features (SPY, VIX, yields, credit, bonds)."""
    df = pd.DataFrame(index=dates)

    # SPY return
    if "SPY" in macro_data:
        spy = macro_data["SPY"].reindex(dates)
        df["spy_return"] = spy.pct_change()

    # VIX level and change
    if "^VIX" in macro_data:
        vix = macro_data["^VIX"].reindex(dates)
        df["vix_level"] = vix / 100  # Normalize to ~0.15-0.30 range
        df["vix_change"] = vix.pct_change()

    # 10Y yield change
    if "DGS10" in macro_data:
        y10 = macro_data["DGS10"].reindex(dates)
        df["yield_10y_change"] = y10.diff() / 100  # Basis points to decimal

    # HYG-LQD spread change (credit)
    if "HYG" in macro_data and "LQD" in macro_data:
        hyg = macro_data["HYG"].reindex(dates)
        lqd = macro_data["LQD"].reindex(dates)
        spread = hyg / lqd
        df["credit_spread_change"] = spread.pct_change()

    # TLT return (bond proxy)
    if "TLT" in macro_data:
        tlt = macro_data["TLT"].reindex(dates)
        df["tlt_return"] = tlt.pct_change()

    # Fill any missing columns with zeros and enforce column order
    ordered_cols = ["spy_return", "vix_level", "vix_change", "yield_10y_change",
                    "credit_spread_change", "tlt_return"]
    for col in ordered_cols:
        if col not in df.columns:
            df[col] = 0.0

    return df[ordered_cols]


def build_sector_neutral_features(
    stock_returns: pd.Series,
    sector_etf_returns: pd.Series,
) -> pd.DataFrame:
    """Channels 22-23: Sector-neutralized returns (1d and 5d).

    CRITICAL: prevents model from learning sector beta instead of alpha.
    """
    df = pd.DataFrame(index=stock_returns.index)
    aligned_etf = sector_etf_returns.reindex(stock_returns.index).fillna(0)

    df["sector_neutral_return_1d"] = stock_returns - aligned_etf
    df["sector_neutral_return_5d"] = (
        stock_returns.rolling(5).sum() - aligned_etf.rolling(5).sum()
    )
    return df


def build_features_for_ticker(
    ticker: str,
    prices_df: pd.DataFrame,
    factors_df: pd.DataFrame,
    macro_data: dict[str, pd.Series],
    sector_etf_returns: pd.Series,
) -> pd.DataFrame | None:
    """Build all 23 feature channels for a single ticker.

    Returns DataFrame with 23 columns, indexed by date.
    Returns None if insufficient data.
    """
    # Filter prices for this ticker
    if "ticker" in prices_df.columns:
        ticker_prices = prices_df[prices_df["ticker"] == ticker].set_index("date").sort_index()
    else:
        ticker_prices = prices_df.sort_index()

    if len(ticker_prices) < 313:  # 250 context + 63 factor warm-up
        logger.debug(f"Skipping {ticker}: only {len(ticker_prices)} rows")
        return None

    # Daily returns
    returns = ticker_prices["close"].pct_change()

    # Build each feature group
    ohlc = build_ohlc_features(ticker_prices)
    volume = build_volume_features(ticker_prices)
    volatility = build_volatility_features(ticker_prices)

    # Factor betas
    rf = factors_df.set_index("date")["RF"] if "RF" in factors_df.columns else pd.Series(dtype=float)
    factors_indexed = factors_df.set_index("date")
    betas = build_factor_betas(returns, factors_indexed, rf)

    # Cross-asset
    cross_asset = build_cross_asset_features(ticker_prices.index, macro_data)

    # Sector-neutral returns
    sector_neutral = build_sector_neutral_features(returns, sector_etf_returns)

    # Combine all features
    features = pd.concat([
        ohlc,                # 4 channels
        volume,              # 2 channels
        volatility,          # 3 channels
        betas,               # 6 channels
        cross_asset,         # 6 channels
        sector_neutral,      # 2 channels
    ], axis=1)

    assert features.shape[1] == 23, f"Expected 23 channels, got {features.shape[1]}: {list(features.columns)}"

    # Validate NaN rate
    nan_rate = features.isna().mean().mean()
    if nan_rate > 0.20:
        logger.warning(f"{ticker}: {nan_rate:.1%} NaN rate — may be problematic")

    return features


FEATURE_COLUMNS = [
    "pct_open", "pct_high", "pct_low", "pct_close",       # 1-4
    "log_volume", "volume_ratio",                            # 5-6
    "realized_vol_21d", "garman_klass_vol", "vol_of_vol",   # 7-9
    "beta_Mkt-RF", "beta_SMB", "beta_HML",                  # 10-12
    "beta_RMW", "beta_CMA", "beta_Mom",                     # 13-15
    "spy_return", "vix_level", "vix_change",                 # 16-18
    "yield_10y_change", "credit_spread_change", "tlt_return",# 19-21
    "sector_neutral_return_1d", "sector_neutral_return_5d",  # 22-23
]
