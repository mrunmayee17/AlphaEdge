from fastapi import APIRouter

from .endpoints.analysis import router as analysis_router
from .endpoints.backtest import router as backtest_router
from .endpoints.websocket import router as ws_router

api_router = APIRouter()
api_router.include_router(analysis_router, tags=["analysis"])
api_router.include_router(backtest_router, tags=["backtest"])
api_router.include_router(ws_router, tags=["websocket"])
