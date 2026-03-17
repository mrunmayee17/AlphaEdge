"""Test fixtures — use REAL services (require running Redis + APIs reachable)."""

import pytest
import redis.asyncio as aioredis
from openai import AsyncOpenAI

from backend.app.config import get_settings
from backend.app.services.data.fred_service import FredService
from backend.app.services.data.yahoo_finance import YahooFinanceService
from backend.app.services.llm.nemotron_client import NemotronClient
from backend.app.services.search.brave_search import BraveSearchClient
from backend.app.services.search.brightdata_reddit import BrightDataClient


@pytest.fixture
def settings():
    return get_settings()


@pytest.fixture
async def redis_client(settings):
    client = aioredis.from_url(settings.redis_url, decode_responses=True)
    yield client
    await client.flushdb()
    await client.close()


@pytest.fixture
async def nemotron_client(settings):
    client = AsyncOpenAI(base_url=settings.nvidia_base_url, api_key=settings.nvidia_api_key)
    yield NemotronClient(client, settings.nvidia_model)


@pytest.fixture
def yahoo_finance():
    return YahooFinanceService()


@pytest.fixture
async def brave_client(settings):
    client = BraveSearchClient(settings.brave_api_key)
    yield client
    await client.close()


@pytest.fixture
async def brightdata_client(settings):
    client = BrightDataClient(settings.brightdata_api_key, settings.brightdata_dataset_id)
    yield client
    await client.close()


@pytest.fixture
def fred_service(settings):
    return FredService(settings.fred_api_key)
