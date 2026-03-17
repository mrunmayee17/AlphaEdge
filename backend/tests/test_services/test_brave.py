"""Test Brave Search client with real API calls."""

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_search(brave_client):
    results = await brave_client.search("AAPL stock news", count=3)
    assert "results" in results
    assert len(results["results"]) > 0
    assert results["results"][0]["title"]
    assert results["results"][0]["url"]
