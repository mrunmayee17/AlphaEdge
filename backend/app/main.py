"""FastAPI application — lifespan verifies ALL services at startup."""

import logging
import os
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI

from backend.app.api import api_router
from backend.app.config import get_settings
from backend.app.core.observability import init_tracing
from backend.app.services.data.fred_service import FredService
from backend.app.services.data.yahoo_finance import YahooFinanceService
from backend.app.services.llm.nemotron_client import NemotronClient
from backend.app.services.search.brave_search import BraveSearchClient
from backend.app.services.search.brightdata_reddit import BrightDataClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_DEFAULT_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://mrunmayee17.github.io",
]


def _resolve_cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "").strip()
    if not raw:
        return _DEFAULT_CORS_ORIGINS
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    return origins or _DEFAULT_CORS_ORIGINS


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: verify ALL services or fail loudly. No silent degradation."""
    settings = get_settings()
    app.state.settings = settings

    # Initialize OpenTelemetry tracing
    init_tracing("bam-backend", endpoint=settings.otel_endpoint if settings.otel_endpoint != "http://localhost:4317" else None)

    # 1. Redis
    logger.info("Connecting to Redis...")
    app.state.redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    await app.state.redis.ping()
    logger.info("Redis: OK")

    # 2. NVIDIA Nemotron API
    logger.info("Checking Nemotron API...")
    llm_client = AsyncOpenAI(
        base_url=settings.nvidia_base_url,
        api_key=settings.nvidia_api_key,
    )
    app.state.nemotron_client = NemotronClient(llm_client, settings.nvidia_model)
    healthy = await app.state.nemotron_client.health_check()
    if not healthy:
        raise RuntimeError("Nemotron API health check failed — check API key and endpoint")
    logger.info("Nemotron API: OK")

    # 3. Yahoo Finance (non-fatal — rate limits on cloud IPs are common)
    logger.info("Checking Yahoo Finance...")
    app.state.yahoo_finance = YahooFinanceService()
    try:
        test_info = app.state.yahoo_finance.get_ticker_info("AAPL")
        if not test_info.get("sector"):
            logger.warning("Yahoo Finance returned no data for AAPL — may be rate limited")
        else:
            logger.info("Yahoo Finance: OK")
    except Exception as e:
        logger.warning(f"Yahoo Finance startup check failed (non-fatal): {e}")

    # 4. Brave Search
    logger.info("Checking Brave Search...")
    app.state.brave_client = BraveSearchClient(settings.brave_api_key)
    brave_test = await app.state.brave_client.search("test", count=1)
    if not brave_test.get("results"):
        logger.warning("Brave Search returned no results for 'test' — may be rate limited")
    logger.info("Brave Search: OK")

    # 5. Bright Data (Reddit)
    logger.info("Setting up Bright Data...")
    app.state.brightdata_client = BrightDataClient(
        settings.brightdata_api_key,
        settings.brightdata_dataset_id,
    )
    logger.info("Bright Data: OK (async — no startup check)")

    # 6. FRED
    logger.info("Checking FRED API...")
    app.state.fred_service = FredService(settings.fred_api_key)
    logger.info("FRED: OK")

    logger.info("All services initialized — server ready")

    yield

    # Shutdown
    logger.info("Shutting down...")
    await app.state.redis.close()
    await app.state.brave_client.close()
    await app.state.brightdata_client.close()


app = FastAPI(
    title="Alpha Edge — Investment Committee API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_resolve_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {"status": "ok"}
