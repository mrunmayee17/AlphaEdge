"""Test backtest engine with synthetic data."""

import numpy as np
import pandas as pd

from app.backtest.engine import BacktestConfig, BacktestEngine, BacktestResult


def test_backtest_result_properties():
    """Test BacktestResult computes metrics correctly."""
    np.random.seed(42)
    n = 100
    idx = pd.bdate_range("2024-01-01", periods=n)
    gross = pd.Series(np.random.randn(n) * 0.002, index=idx)
    net = gross - 0.0001  # small cost drag

    result = BacktestResult(
        gross_returns=gross,
        net_returns=net,
        turnover=pd.Series(np.random.rand(n) * 0.1, index=idx),
        cost_series=pd.Series(np.full(n, 0.0001), index=idx),
        n_longs=pd.Series(np.full(n, 10), index=idx),
        n_shorts=pd.Series(np.full(n, 10), index=idx),
    )

    assert isinstance(result.gross_sharpe, float)
    assert isinstance(result.net_sharpe, float)
    assert result.max_drawdown <= 0
    assert result.annual_vol > 0


def test_backtest_summary_keys():
    """Test summary dict contains all expected keys."""
    result = BacktestResult(
        gross_returns=pd.Series([0.01, -0.005, 0.003]),
        net_returns=pd.Series([0.009, -0.006, 0.002]),
        turnover=pd.Series([0.1, 0.0, 0.05]),
    )
    summary = result.summary()
    expected = [
        "gross_sharpe", "net_sharpe", "annual_return",
        "annual_vol", "max_drawdown", "avg_daily_turnover", "n_days",
    ]
    for key in expected:
        assert key in summary


def test_backtest_engine_with_data():
    """Test full backtest with properly structured synthetic data."""
    np.random.seed(42)
    tickers = [f"T{i:02d}" for i in range(30)]
    dates = pd.bdate_range("2024-01-01", periods=60)

    # Build long-form prices
    price_rows = []
    for t in tickers:
        base = 100 + np.random.randn() * 20
        prices = base * np.cumprod(1 + np.random.randn(len(dates)) * 0.02)
        for d, p in zip(dates, prices):
            price_rows.append({"date": d, "ticker": t, "adj_close": p, "volume": 1_000_000})

    prices_df = pd.DataFrame(price_rows)

    # Predictions (weekly, for Fridays before each Monday)
    pred_rows = []
    for d in dates:
        for t in tickers:
            pred_rows.append({"date": d, "ticker": t, "alpha_21d": np.random.randn() * 0.01})
    predictions_df = pd.DataFrame(pred_rows)

    # Market caps
    mcap_pivot = pd.DataFrame(
        {t: np.full(len(dates), 50e9) for t in tickers},
        index=dates,
    )

    engine = BacktestEngine(BacktestConfig(
        start_date="2024-01-15",
        end_date="2024-03-15",
        min_adv_dollars=100,  # low threshold for synthetic data
    ))
    result = engine.run(predictions_df, prices_df, mcap_pivot)
    summary = result.summary()
    assert summary["n_days"] > 0
    assert "net_sharpe" in summary
