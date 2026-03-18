"""Chronos-2 time-series inference — loads pretrained foundation model from
HuggingFace and generates probabilistic alpha predictions.

Uses amazon/chronos-bolt-base (or configured variant) to forecast close-price
returns at 1d, 5d, 21d, and 63d horizons with quantile bands.
"""

import logging
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

# Cached pipeline to avoid reloading on every request
_pipeline_cache: dict = {}

# Forecast horizons (trading days)
HORIZONS = [1, 5, 21, 63]
HORIZON_LABELS = ["1d", "5d", "21d", "63d"]

# Chronos-Bolt outputs 9 quantiles: [0.1, 0.2, ..., 0.9]
# We use indices 0 (q10), 4 (q50/median), 8 (q90)
Q10_IDX, Q50_IDX, Q90_IDX = 0, 4, 8


def _load_pipeline(model_id: str):
    """Load Chronos pipeline from HuggingFace (cached after first call)."""
    if model_id in _pipeline_cache:
        return _pipeline_cache[model_id]

    from chronos import ChronosBoltPipeline

    pipeline = ChronosBoltPipeline.from_pretrained(
        model_id,
        device_map="cpu",
        dtype=torch.float32,
    )
    _pipeline_cache[model_id] = pipeline
    logger.info(f"Loaded Chronos pipeline: {model_id}")
    return pipeline


def _fetch_returns(ticker: str, context_length: int = 512) -> np.ndarray:
    """Fetch recent daily close-price returns for a ticker.

    Returns a 1-D numpy array of daily log-returns (last `context_length` days).
    Uses retry logic to handle Yahoo Finance rate limits on cloud IPs.
    """
    import yfinance as yf

    end = date.today()
    start = end - timedelta(days=int(context_length * 1.8))  # buffer for non-trading days

    for attempt in range(3):
        try:
            df = yf.download(ticker, start=str(start), end=str(end), progress=False)
            break
        except Exception as e:
            if "Rate" in str(e) and attempt < 2:
                wait = 2 ** (attempt + 1)
                logger.warning(f"Yahoo Finance rate limited in inference, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise

    if df.empty or len(df) < context_length:
        raise ValueError(f"Insufficient price data for {ticker}: {len(df)} rows (need {context_length})")

    # Flatten MultiIndex columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"].values.astype(np.float64)
    returns = np.diff(np.log(close))  # log-returns
    return returns[-context_length:]


async def run_inference(ticker: str, sector: str, sector_etf: str, model_id: str) -> dict:
    """Run Chronos-2 inference for a single ticker. Returns AlphaPrediction dict."""
    start_time = time.time()

    pipeline = _load_pipeline(model_id)

    returns = _fetch_returns(ticker)
    context = torch.tensor(returns, dtype=torch.float32)

    # Forecast out to the longest horizon
    max_horizon = max(HORIZONS)
    forecast = pipeline.predict(
        context,
        prediction_length=max_horizon,
    )  # shape: (1, 9, max_horizon) — 9 quantiles

    quantiles = forecast.squeeze(0).numpy()  # (9, max_horizon)

    # For each horizon, compute cumulative return over that window
    # then extract q10, q50 (median), q90 from the quantile forecasts
    result = {
        "ticker": ticker,
        "prediction_date": str(date.today()),
        "sector": sector,
        "sector_etf": sector_etf,
    }

    for horizon, label in zip(HORIZONS, HORIZON_LABELS):
        # Cumulative return = sum of daily log-return forecasts over the horizon
        q10 = float(quantiles[Q10_IDX, :horizon].sum())
        q50 = float(quantiles[Q50_IDX, :horizon].sum())
        q90 = float(quantiles[Q90_IDX, :horizon].sum())
        result[f"alpha_{label}"] = q50
        result[f"q10_{label}"] = q10
        result[f"q90_{label}"] = q90

    latency_ms = (time.time() - start_time) * 1000

    result.update({
        "patch_attention": [],
        "top_features": [],
        "model_version": f"chronos-2:{model_id.split('/')[-1]}",
        "training_fold": "pretrained",
        "inference_latency_ms": round(latency_ms, 1),
    })

    return result
