"""LangGraph StateGraph orchestrator — 3-round Investment Committee protocol.

Round 1: 5 agents analyze independently (parallel)
Round 2: Agents debate — each sees others' claims (parallel)
Round 3: Synthesize into Investment Memo
"""

import json
import logging
from datetime import date, datetime
from typing import Any, Optional

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from backend.app.models.schemas import (
    AgentDebateResponse,
    AgentView,
    AlphaPrediction,
    InvestmentMemo,
    TraceEvent,
)

from .prompts import AGENT_NAMES, AGENT_PROMPTS, DEBATE_SYSTEM, MEMO_SYSTEM
from .tools import AGENT_TOOLS, set_committee_state

logger = logging.getLogger(__name__)

FORECAST_MODEL_LABELS = {
    "chronos": "Chronos-2",
    "fincast_lora": "Fine-tuned FinCast LoRA",
}


class CommitteeState(TypedDict, total=False):
    """Full state for the 3-round committee protocol."""
    ticker: str
    forecast_model: str
    alpha_prediction: Optional[dict]
    # Round 1: agent views
    quant_view: Optional[dict]
    fundamentals_view: Optional[dict]
    sentiment_view: Optional[dict]
    risk_view: Optional[dict]
    macro_view: Optional[dict]
    # Claims extraction
    extracted_claims: Optional[dict]
    # Round 2: debate responses
    quant_debate: Optional[dict]
    fundamentals_debate: Optional[dict]
    sentiment_debate: Optional[dict]
    risk_debate: Optional[dict]
    macro_debate: Optional[dict]
    # Round 3: memo
    memo: Optional[dict]
    # Metadata
    status: str
    trace: list
    error: Optional[str]


def _prefetch_agent_data(agent_name: str, ticker: str, asset_name: str) -> tuple[dict[str, str], list[dict]]:
    """Eagerly fetch ALL tool data for an agent before the LLM loop.

    Returns (tool_results_map, trace_events) where tool_results_map is
    {tool_name: json_string} for all successfully fetched tools.
    """
    from .tools import AGENT_TOOLS

    tools = AGENT_TOOLS.get(agent_name, [])
    tool_map = {t.name: t for t in tools}
    results = {}
    trace_events = []

    # Define how to invoke each tool
    invocations = {
        "get_alpha_prediction": lambda fn: fn.invoke({}),
        "think": None,  # skip — LLM calls this interactively
        "get_fundamentals": lambda fn: fn.invoke({"ticker": ticker}),
        "get_financial_statements": lambda fn: fn.invoke({"ticker": ticker}),
        "get_analyst_estimates": lambda fn: fn.invoke({"ticker": ticker}),
        "get_options_data": lambda fn: fn.invoke({"ticker": ticker}),
        "get_short_interest": lambda fn: fn.invoke({"ticker": ticker}),
        "get_relative_valuation": lambda fn: fn.invoke({"ticker": ticker}),
        "get_risk_metrics": lambda fn: fn.invoke({"ticker": ticker}),
        "get_price_history": lambda fn: fn.invoke({"ticker": ticker, "period": "6mo"}),
        "search_web": lambda fn: fn.invoke({"query": f"{asset_name} ({ticker}) news", "count": 5}),
        "search_news_with_extraction": lambda fn: fn.invoke({"ticker": ticker, "company_name": asset_name}),
        "search_reddit": lambda fn: fn.invoke({"keyword": asset_name, "num_posts": 5}),
        "get_yield_curve": lambda fn: fn.invoke({}),
        "get_macro_data": lambda fn: fn.invoke({}),
        "get_regime": lambda fn: fn.invoke({"ticker": ticker}),
    }

    for tool_name, tool_fn in tool_map.items():
        invoker = invocations.get(tool_name)
        if invoker is None:
            continue
        try:
            result = invoker(tool_fn)
            result_str = json.dumps(result, default=str)[:3000]
            results[tool_name] = result_str
            trace_events.append(TraceEvent(
                agent=agent_name,
                tool=tool_name,
                args={"ticker": ticker},
                result=result_str,
            ).model_dump())
            logger.info(f"  {agent_name} prefetch {tool_name}: OK")
        except Exception as e:
            logger.warning(f"  {agent_name} prefetch {tool_name} failed: {e}")
            results[tool_name] = json.dumps({"error": str(e)})

    return results, trace_events


async def run_agent_round1(
    state: CommitteeState,
    agent_name: str,
    nemotron_client,
) -> dict:
    """Run a single agent's Round 1 analysis.

    1. Eagerly pre-fetches ALL tool data so the LLM always has real data.
    2. Sends data + alpha prediction in the prompt.
    3. LLM produces the AgentView JSON in 1-2 turns (no fragile tool-call loop).
    """
    ticker = state["ticker"]
    asset_name = state.get("asset_name", ticker)
    forecast_model = state.get("forecast_model", "chronos")
    forecast_model_label = FORECAST_MODEL_LABELS.get(forecast_model, forecast_model)
    system_prompt = AGENT_PROMPTS[agent_name].format(
        ticker=ticker,
        agent_name=agent_name,
        asset_name=asset_name,
        forecast_model=forecast_model,
        forecast_model_label=forecast_model_label,
    )

    # Set state so tools can access alpha prediction
    set_committee_state(state)

    # Inject alpha prediction
    alpha_pred = state.get("alpha_prediction")
    alpha_str = json.dumps(alpha_pred, default=str) if alpha_pred else "No alpha prediction available"

    # ── Pre-fetch ALL tool data eagerly ──
    logger.info(f"Agent {agent_name}: pre-fetching tool data for {ticker}...")
    tool_data, trace_events = _prefetch_agent_data(agent_name, ticker, asset_name)

    # Build a formatted data block from all prefetched results
    data_sections = []
    for tool_name, result_str in tool_data.items():
        data_sections.append(f"=== {tool_name} ===\n{result_str}")
    all_data_str = "\n\n".join(data_sections) if data_sections else "No tool data available."

    # Build messages — all data is already included
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": (
            f"Analyze {asset_name} ({ticker}).\n\n"
            f"SELECTED FORECAST ENGINE: {forecast_model} ({forecast_model_label})\n\n"
            f"ALPHA PREDICTION (from the selected forecast model — use these EXACT numbers):\n{alpha_str}\n\n"
            f"TOOL DATA (pre-fetched — use these EXACT values in your analysis):\n{all_data_str}\n\n"
            f"Guidelines for direction based on alpha_21d:\n"
            f"- If |alpha_21d| < 0.005 (0.5%), the signal is WEAK — lean NEUTRAL unless tool data strongly supports a direction.\n"
            f"- If alpha_21d < -0.005, lean BEARISH unless other evidence is overwhelmingly positive.\n"
            f"- If alpha_21d > 0.005, lean BULLISH unless risks are severe.\n"
            f"- Conviction should reflect signal strength: weak signals = lower conviction.\n\n"
            f"IMPORTANT: Base your key_claims and supporting_evidence ONLY on the exact values from the tool data above.\n"
            f"Return your analysis as a valid AgentView JSON object."
        )},
    ]

    # ── LLM call — usually produces valid JSON in 1-2 turns ──
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            response = await nemotron_client.chat_completion(
                messages, stream=False, max_tokens=3000, temperature=0.4
            )
        except Exception as e:
            logger.error(f"Agent {agent_name} LLM call failed: {e}")
            return {f"{agent_name}_view": None, "trace": trace_events}

        if response is None:
            logger.error(f"Agent {agent_name} returned None")
            break

        # Try to parse JSON from response
        try:
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            view_dict = json.loads(cleaned.strip())

            if "direction" in view_dict and "conviction" in view_dict:
                view_dict.setdefault("agent_name", agent_name)
                view_dict.setdefault("ticker", ticker)
                view_dict.setdefault("key_claims", [])
                view_dict.setdefault("risks", [])
                view_dict.setdefault("summary", "")
                view_dict.setdefault("agrees_with_alpha", False)
                view_dict.setdefault("time_horizon", "1M")
                logger.info(f"Agent {agent_name}: {view_dict.get('direction')} "
                          f"conviction={view_dict.get('conviction')}")
                return {f"{agent_name}_view": view_dict, "trace": trace_events}
        except (json.JSONDecodeError, KeyError):
            pass

        # Not valid JSON — ask again
        messages.append({"role": "assistant", "content": response})
        messages.append({
            "role": "user",
            "content": "Please return ONLY a valid AgentView JSON object now. No markdown, no explanation."
        })

    # Final fallback — force JSON mode
    try:
        final_response = await nemotron_client.chat_completion_json(
            messages + [{"role": "user", "content": "Return ONLY the AgentView JSON now."}],
            max_tokens=2000,
        )
        final_response.setdefault("agent_name", agent_name)
        final_response.setdefault("ticker", ticker)
        final_response.setdefault("direction", "NEUTRAL")
        final_response.setdefault("conviction", 0.5)
        final_response.setdefault("key_claims", [])
        final_response.setdefault("risks", [])
        final_response.setdefault("summary", "")
        final_response.setdefault("agrees_with_alpha", False)
        final_response.setdefault("time_horizon", "1M")
        return {f"{agent_name}_view": final_response, "trace": trace_events}
    except Exception as e:
        logger.error(f"Agent {agent_name} failed to produce valid output: {e}")
        return {f"{agent_name}_view": None, "trace": trace_events}


def extract_claims(state: CommitteeState) -> dict:
    """Pure function — serialize key_claims from each AgentView. NOT an LLM call."""
    claims = {}
    for agent_name in AGENT_NAMES:
        view = state.get(f"{agent_name}_view")
        if view and isinstance(view, dict):
            claims[agent_name] = view.get("key_claims", [])[:5]
    return {"extracted_claims": claims}


async def run_agent_debate(
    state: CommitteeState,
    agent_name: str,
    nemotron_client,
) -> dict:
    """Run Round 2 debate for a single agent."""
    ticker = state["ticker"]
    all_claims = state.get("extracted_claims", {})
    own_view = state.get(f"{agent_name}_view", {})

    others_claims = {k: v for k, v in all_claims.items() if k != agent_name}

    prompt = DEBATE_SYSTEM.format(
        agent_name=agent_name,
        ticker=ticker,
        others_claims=json.dumps(others_claims, default=str, indent=2),
        own_direction=own_view.get("direction", "NEUTRAL") if own_view else "NEUTRAL",
        own_conviction=own_view.get("conviction", 0.5) if own_view else 0.5,
        own_claims=json.dumps(own_view.get("key_claims", []), default=str) if own_view else "[]",
    )

    try:
        result = await nemotron_client.chat_completion_json(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Review the other agents' claims and respond with your debate JSON."},
            ],
            max_tokens=2000,
        )
        result.setdefault("agent_name", agent_name)
        return {f"{agent_name}_debate": result}
    except Exception as e:
        logger.error(f"Debate for {agent_name} failed: {e}")
        return {f"{agent_name}_debate": None}


async def synthesize_memo(
    state: CommitteeState,
    nemotron_client,
) -> dict:
    """Round 3: Synthesize all views + debate into final Investment Memo."""
    ticker = state["ticker"]
    forecast_model = state.get("forecast_model", "chronos")
    forecast_model_label = FORECAST_MODEL_LABELS.get(forecast_model, forecast_model)
    alpha = state.get("alpha_prediction", {})

    # Collect agent views
    agent_views = {}
    debate_responses = {}
    for name in AGENT_NAMES:
        view = state.get(f"{name}_view")
        if view:
            agent_views[name] = {
                "direction": view.get("direction"),
                "conviction": view.get("conviction"),
                "summary": view.get("summary", ""),
                "key_claims": view.get("key_claims", []),
            }
        debate = state.get(f"{name}_debate")
        if debate:
            debate_responses[name] = debate

    alpha_summary = ""
    if alpha:
        if hasattr(alpha, "summary"):
            alpha_summary = alpha.summary()
        elif isinstance(alpha, dict):
            alpha_summary = (
                f"{ticker}: model={forecast_model}, "
                f"model_version={alpha.get('model_version', 'N/A')}, "
                f"alpha_21d={alpha.get('alpha_21d', 'N/A')}, "
                f"[{alpha.get('q10_21d', 'N/A')}, {alpha.get('q90_21d', 'N/A')}]"
            )

    prompt = MEMO_SYSTEM.format(
        ticker=ticker,
        date=str(date.today()),
        forecast_model=forecast_model,
        forecast_model_label=forecast_model_label,
        agent_views=json.dumps(agent_views, default=str, indent=2),
        debate_responses=json.dumps(debate_responses, default=str, indent=2),
        alpha_summary=alpha_summary,
    )

    try:
        memo_dict = await nemotron_client.chat_completion_json(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Synthesize the Investment Memo now."},
            ],
            max_tokens=4000,
        )
        memo_dict.setdefault("ticker", ticker)
        memo_dict.setdefault("date", str(date.today()))

        # Add alpha prediction to memo
        if alpha:
            memo_dict["alpha_prediction"] = alpha if isinstance(alpha, dict) else alpha.model_dump()

        return {"memo": memo_dict, "status": "complete"}
    except Exception as e:
        logger.error(f"Memo synthesis failed: {e}")
        return {"memo": None, "status": "error", "error": str(e)}


def build_committee_graph():
    """Build the LangGraph StateGraph for the 3-round committee protocol.

    Returns the compiled graph. The caller must provide nemotron_client
    via the config when invoking.
    """
    graph = StateGraph(CommitteeState)

    # Nodes are defined but actual execution is handled by the runner
    # since we need async + nemotron_client injection
    graph.add_node("extract_claims", extract_claims)

    graph.set_entry_point("extract_claims")
    graph.add_edge("extract_claims", END)

    return graph.compile()
