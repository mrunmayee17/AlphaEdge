"""Test Chronos-2 inference service."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


CHRONOS_MODEL_ID = "amazon/chronos-bolt-base"


@pytest.fixture
def mock_pipeline():
    """Create a mock Chronos pipeline that returns shaped forecast."""
    pipeline = MagicMock()
    # Simulate forecast: (1, num_samples, prediction_length)
    forecast = torch.randn(1, 20, 63) * 0.01
    pipeline.predict.return_value = forecast
    return pipeline


@pytest.fixture
def mock_returns():
    """Synthetic daily returns."""
    return np.random.randn(512) * 0.01


def test_run_inference_shape(mock_pipeline, mock_returns):
    """Inference result should contain all expected keys."""
    from app.services.prediction.inference import _pipeline_cache

    # Inject mock pipeline into cache
    _pipeline_cache[CHRONOS_MODEL_ID] = mock_pipeline

    import asyncio
    from app.services.prediction.inference import run_inference

    with patch("app.services.prediction.inference._fetch_returns", return_value=mock_returns):
        result = asyncio.get_event_loop().run_until_complete(
            run_inference("AAPL", "Technology", "XLK", CHRONOS_MODEL_ID)
        )

    assert result["ticker"] == "AAPL"
    for label in ["1d", "5d", "21d", "63d"]:
        assert f"alpha_{label}" in result
        assert f"q10_{label}" in result
        assert f"q90_{label}" in result
    assert "chronos-2" in result["model_version"]

    # Clean up
    _pipeline_cache.pop(CHRONOS_MODEL_ID, None)


def test_quantiles_ordered(mock_pipeline, mock_returns):
    """q10 should be <= q50 <= q90 for each horizon."""
    from app.services.prediction.inference import _pipeline_cache

    # Use a pipeline that returns sorted-friendly samples
    forecast = torch.randn(1, 200, 63) * 0.01  # more samples for stability
    mock_pipeline.predict.return_value = forecast
    _pipeline_cache[CHRONOS_MODEL_ID] = mock_pipeline

    import asyncio
    from app.services.prediction.inference import run_inference

    with patch("app.services.prediction.inference._fetch_returns", return_value=mock_returns):
        result = asyncio.get_event_loop().run_until_complete(
            run_inference("AAPL", "Technology", "XLK", CHRONOS_MODEL_ID)
        )

    for label in ["1d", "5d", "21d", "63d"]:
        assert result[f"q10_{label}"] <= result[f"alpha_{label}"] + 0.01
        assert result[f"alpha_{label}"] <= result[f"q90_{label}"] + 0.01

    _pipeline_cache.pop(CHRONOS_MODEL_ID, None)
