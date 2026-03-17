"""Backtest API endpoint — run and retrieve backtest results."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app.backtest.engine import BacktestConfig, BacktestResult

logger = logging.getLogger(__name__)

router = APIRouter()


class BacktestRequest(BaseModel):
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"
    rebalance_day: int = Field(default=0, ge=0, le=4, description="0=Monday, 4=Friday")
    decile_pct: float = Field(default=0.10, gt=0, le=0.5)
    min_adv_dollars: float = Field(default=5_000_000, gt=0)


class BacktestSummaryResponse(BaseModel):
    gross_sharpe: float
    net_sharpe: float
    annual_return: float
    annual_vol: float
    max_drawdown: float
    avg_daily_turnover: float
    n_days: int


@router.post("/backtest", response_model=BacktestSummaryResponse)
async def run_backtest(req: BacktestRequest):
    """Run a backtest with the given configuration.

    Requires pre-computed predictions and price data in data/processed/.
    """
    try:
        import pandas as pd
        from pathlib import Path

        data_dir = Path(__file__).resolve().parent.parent.parent.parent.parent / "alpha_model" / "data"

        # Load predictions (from model evaluation output)
        pred_path = data_dir / "processed" / "predictions.parquet"
        if not pred_path.exists():
            raise HTTPException(
                status_code=422,
                detail="Predictions file not found. Run model evaluation first.",
            )
        predictions = pd.read_parquet(pred_path)

        # Load prices
        price_path = data_dir / "raw" / "prices.parquet"
        if not price_path.exists():
            raise HTTPException(status_code=422, detail="Price data not found.")
        prices_raw = pd.read_parquet(price_path)

        # Reshape prices for backtest engine
        prices = prices_raw.rename(columns={"adj_close": "adj_close"})[
            ["date", "ticker", "adj_close", "volume"]
        ] if "adj_close" in prices_raw.columns else prices_raw[["date", "ticker", "close", "volume"]].rename(
            columns={"close": "adj_close"}
        )

        # Market caps (approximate from price × shares outstanding, or use a separate file)
        mcap_path = data_dir / "processed" / "ticker_meta.parquet"
        if mcap_path.exists():
            meta = pd.read_parquet(mcap_path)
            # If market_cap column exists, use it
            if "market_cap" in meta.columns:
                market_caps = meta.pivot(index="date", columns="ticker", values="market_cap")
            else:
                # Default to mid-cap for all
                market_caps = pd.DataFrame()
        else:
            market_caps = pd.DataFrame()

        from backend.app.backtest.engine import BacktestEngine

        config = BacktestConfig(
            start_date=req.start_date,
            end_date=req.end_date,
            rebalance_day=req.rebalance_day,
            decile_pct=req.decile_pct,
            min_adv_dollars=req.min_adv_dollars,
        )
        engine = BacktestEngine(config)
        result = engine.run(predictions, prices, market_caps)
        return BacktestSummaryResponse(**result.summary())

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Backtest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
