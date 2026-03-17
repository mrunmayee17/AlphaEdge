"""Analysis API endpoints — POST to start, GET to check status/results."""

import asyncio
import json
import logging
import uuid
from datetime import date

from fastapi import APIRouter, HTTPException, Request

from backend.app.agents.orchestrator import (
    CommitteeState,
    extract_claims,
    run_agent_debate,
    run_agent_round1,
    synthesize_memo,
)
from backend.app.agents.prompts import AGENT_NAMES
from backend.app.agents.tools import set_committee_state, set_tool_services
from backend.app.core.hallucination_guard import check_agent_claims, build_correction_prompt
from backend.app.core.observability import trace_span
from backend.app.models.schemas import AnalysisRequest, AnalysisResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/analysis", response_model=AnalysisResponse)
async def start_analysis(req: AnalysisRequest, request: Request):
    """Start a full committee analysis for a ticker.

    Creates a session, launches the 3-round pipeline in the background.
    """
    analysis_id = str(uuid.uuid4())
    ticker = req.ticker.upper()

    # Store initial state in Redis
    redis = request.app.state.redis
    state: CommitteeState = {
        "ticker": ticker,
        "alpha_prediction": None,
        "status": "pending",
        "trace": [],
        "error": None,
    }
    for name in AGENT_NAMES:
        state[f"{name}_view"] = None
        state[f"{name}_debate"] = None
    state["extracted_claims"] = None
    state["memo"] = None

    await redis.set(
        f"analysis:{analysis_id}",
        json.dumps(state, default=str),
        ex=request.app.state.settings.redis_session_ttl_seconds,
    )

    # Launch pipeline in background
    asyncio.create_task(
        _run_pipeline(analysis_id, state, request.app)
    )

    return AnalysisResponse(analysis_id=analysis_id, ticker=ticker, status="pending")


@router.get("/analysis/{analysis_id}")
async def get_analysis_status(analysis_id: str, request: Request):
    """Get current analysis status and results."""
    redis = request.app.state.redis
    data = await redis.get(f"analysis:{analysis_id}")
    if data is None:
        raise HTTPException(status_code=404, detail="Analysis not found or expired")
    return json.loads(data)


@router.get("/analysis/{analysis_id}/memo")
async def get_memo(analysis_id: str, request: Request):
    """Get the final Investment Memo (only available when status=complete)."""
    redis = request.app.state.redis
    data = await redis.get(f"analysis:{analysis_id}")
    if data is None:
        raise HTTPException(status_code=404, detail="Analysis not found or expired")
    state = json.loads(data)
    if state["status"] != "complete":
        raise HTTPException(status_code=409, detail=f"Analysis status: {state['status']}")
    if state.get("memo") is None:
        raise HTTPException(status_code=500, detail="Memo not generated")
    return state["memo"]


async def _update_state(redis, analysis_id: str, state: dict, ttl: int):
    """Persist state to Redis."""
    await redis.set(
        f"analysis:{analysis_id}",
        json.dumps(state, default=str),
        ex=ttl,
    )


async def _run_pipeline(analysis_id: str, state: CommitteeState, app):
    """Execute the full 3-round committee pipeline with OTel tracing."""
    redis = app.state.redis
    nemotron = app.state.nemotron_client
    ttl = app.state.settings.redis_session_ttl_seconds

    try:
        with trace_span("analysis_session", {"analysis_id": analysis_id, "ticker": state["ticker"]}):
            # ── Round 0: Alpha Prediction ──
            with trace_span("predict_alpha", {"ticker": state["ticker"]}):
                state["status"] = "predicting"
                await _update_state(redis, analysis_id, state, ttl)

                from backend.app.services.data.yahoo_finance import YahooFinanceService
                yf_svc = YahooFinanceService()

                # Resolve ticker (e.g. CL → CL=F "WTI Crude Oil Futures")
                raw_ticker = state["ticker"]
                ticker, asset_name = yf_svc.resolve_ticker(raw_ticker)
                state["ticker"] = ticker
                state["asset_name"] = asset_name

                sector_etf = yf_svc.get_sector_etf(ticker)
                info = yf_svc.get_ticker_info(ticker)
                sector = info.get("sector", "Unknown")

                # Try real model inference, fall back to placeholder
                alpha_pred = await _try_model_inference(ticker, sector, sector_etf, app)
                state["alpha_prediction"] = alpha_pred
                await _update_state(redis, analysis_id, state, ttl)

            # ── Round 1: Independent Agent Analysis (parallel) ──
            with trace_span("round_1", {"n_agents": len(AGENT_NAMES)}):
                state["status"] = "round_1"
                await _update_state(redis, analysis_id, state, ttl)

                set_tool_services(
                    yahoo=yf_svc,
                    brave=app.state.brave_client,
                    brightdata=app.state.brightdata_client,
                    fred=app.state.fred_service,
                )
                set_committee_state(state)

                tasks = [
                    run_agent_round1(state, name, nemotron)
                    for name in AGENT_NAMES
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                all_traces = []
                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"Agent failed: {result}")
                        continue
                    if isinstance(result, dict):
                        for key, value in result.items():
                            if key == "trace":
                                all_traces.extend(value)
                            else:
                                state[key] = value

                state["trace"] = all_traces

                # Hallucination check on agent views
                for name in AGENT_NAMES:
                    view = state.get(f"{name}_view")
                    if view and isinstance(view, dict):
                        mismatches = check_agent_claims(view, all_traces, name)
                        if mismatches:
                            correction = build_correction_prompt(mismatches, name)
                            if correction:
                                logger.warning(f"Agent {name} hallucination detected, requesting correction")
                                corrected = await _correct_agent_view(
                                    state, name, nemotron, correction
                                )
                                if corrected:
                                    state[f"{name}_view"] = corrected

                await _update_state(redis, analysis_id, state, ttl)

            # ── Claims extraction (pure function, no LLM) ──
            with trace_span("extract_claims"):
                claims_result = extract_claims(state)
                state.update(claims_result)

            # ── Round 2: Debate (parallel) ──
            with trace_span("round_2", {"n_agents": len(AGENT_NAMES)}):
                state["status"] = "round_2"
                await _update_state(redis, analysis_id, state, ttl)

                debate_tasks = [
                    run_agent_debate(state, name, nemotron)
                    for name in AGENT_NAMES
                ]
                debate_results = await asyncio.gather(*debate_tasks, return_exceptions=True)

                for result in debate_results:
                    if isinstance(result, Exception):
                        logger.error(f"Debate failed: {result}")
                        continue
                    if isinstance(result, dict):
                        state.update(result)

                await _update_state(redis, analysis_id, state, ttl)

            # ── Round 3: Memo Synthesis ──
            with trace_span("round_3"):
                state["status"] = "round_3"
                await _update_state(redis, analysis_id, state, ttl)

                memo_result = await synthesize_memo(state, nemotron)
                state.update(memo_result)

                await _update_state(redis, analysis_id, state, ttl)
                logger.info(f"Analysis {analysis_id} complete for {ticker}")

    except Exception as e:
        logger.exception(f"Pipeline failed for {analysis_id}: {e}")
        state["status"] = "error"
        state["error"] = str(e)
        await _update_state(redis, analysis_id, state, ttl)


async def _try_model_inference(ticker: str, sector: str, sector_etf: str, app) -> dict:
    """Try Chronos-2 model inference, fall back to placeholder values."""
    try:
        from backend.app.services.prediction.inference import run_inference
        return await run_inference(ticker, sector, sector_etf, app.state.settings.chronos_model_id)
    except Exception as e:
        logger.warning(f"Model inference failed, using placeholder: {e}")
        return {
            "ticker": ticker,
            "prediction_date": str(date.today()),
            "sector": sector,
            "sector_etf": sector_etf,
            "alpha_1d": 0.002, "alpha_5d": 0.008, "alpha_21d": 0.025, "alpha_63d": 0.045,
            "q10_1d": -0.015, "q90_1d": 0.020,
            "q10_5d": -0.030, "q90_5d": 0.050,
            "q10_21d": -0.040, "q90_21d": 0.090,
            "q10_63d": -0.060, "q90_63d": 0.150,
            "patch_attention": [], "top_features": [],
            "model_version": "placeholder", "training_fold": "none",
            "inference_latency_ms": 0.0,
        }


async def _correct_agent_view(state: dict, agent_name: str, nemotron, correction_prompt: str) -> dict | None:
    """Re-prompt agent with hallucination correction."""
    original_view = state.get(f"{agent_name}_view", {}) or {}
    try:
        result = await nemotron.chat_completion_json(
            [
                {"role": "system", "content": (
                    f"You are the {agent_name} agent. Fix your analysis. "
                    f"You MUST include these fields in your JSON response: "
                    f"direction (BULLISH/BEARISH/NEUTRAL), conviction (0.0-1.0), "
                    f"summary, key_claims, risks, agrees_with_alpha, time_horizon."
                )},
                {"role": "user", "content": (
                    f"Your original analysis:\n{json.dumps(original_view, default=str)}\n\n"
                    f"{correction_prompt}"
                )},
            ],
            max_tokens=2000,
        )
        # Preserve critical fields from original if correction omits them
        for field in ("direction", "conviction", "summary", "key_claims", "risks",
                      "agrees_with_alpha", "time_horizon"):
            if field not in result and field in original_view:
                result[field] = original_view[field]
        result.setdefault("agent_name", agent_name)
        result.setdefault("ticker", state["ticker"])
        return result
    except Exception as e:
        logger.error(f"Correction for {agent_name} failed: {e}")
        return None
