"""FRED API service for macro economic data."""

import logging
from datetime import datetime, timedelta

import pandas as pd
from fredapi import Fred

logger = logging.getLogger(__name__)


class FredService:
    """Federal Reserve Economic Data (FRED) API wrapper."""

    # Key series we use
    SERIES = {
        "DGS10": "10-Year Treasury Yield",
        "DGS2": "2-Year Treasury Yield",
        "DGS3MO": "3-Month Treasury Yield",
        "FEDFUNDS": "Federal Funds Rate",
    }

    def __init__(self, api_key: str):
        self.fred = Fred(api_key=api_key)

    def get_yields(self, lookback_days: int = 365) -> dict[str, pd.Series]:
        """Fetch recent Treasury yields and Fed Funds rate."""
        start = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        results = {}
        for series_id in self.SERIES:
            try:
                data = self.fred.get_series(series_id, observation_start=start)
                results[series_id] = data.dropna()
                logger.info(f"FRED {series_id}: {len(results[series_id])} observations")
            except Exception as e:
                logger.error(f"FRED {series_id} failed: {e}")
                raise
        return results

    def get_yield_curve_snapshot(self) -> dict:
        """Current yield curve snapshot for macro agent."""
        yields = self.get_yields(lookback_days=30)
        latest = {}
        for series_id, data in yields.items():
            if len(data) > 0:
                latest[series_id] = float(data.iloc[-1])

        spread_2s10s = None
        if "DGS10" in latest and "DGS2" in latest:
            spread_2s10s = latest["DGS10"] - latest["DGS2"]

        return {
            "yields": latest,
            "spread_2s10s": spread_2s10s,
            "description": self.SERIES,
        }

    def get_series(self, series_id: str, start: str) -> pd.Series:
        """Fetch a single FRED series."""
        return self.fred.get_series(series_id, observation_start=start).dropna()
