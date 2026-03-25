"""Agent tools — real data fetching tools used by all 5 agents.

Each tool returns a dict. If a data source fails, it returns an error dict
so the agent LLM can reason with partial data (graceful degradation).
"""

import asyncio
import logging
import math
from datetime import date, datetime
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# These will be set at app startup from app.state
_yahoo_finance_service = None
_brave_search_client = None
_brightdata_client = None
_fred_service = None
_committee_state = None


def set_tool_services(yahoo=None, brave=None, brightdata=None, fred=None):
    """Called at startup to inject real service instances."""
    global _yahoo_finance_service, _brave_search_client, _brightdata_client, _fred_service
    if yahoo:
        _yahoo_finance_service = yahoo
    if brave:
        _brave_search_client = brave
    if brightdata:
        _brightdata_client = brightdata
    if fred:
        _fred_service = fred


def set_committee_state(state: dict):
    """Set the current committee state for tool access."""
    global _committee_state
    _committee_state = state


# ── Shared tools (all agents) ──────────────────────────────────────────────


@tool
def get_alpha_prediction() -> dict:
    """Read the model-generated alpha prediction from committee state.
    Returns the full prediction with quantile bands at 1d/5d/21d/63d horizons."""
    if _committee_state is None or _committee_state.get("alpha_prediction") is None:
        return {"error": "No alpha prediction available"}
    pred = _committee_state["alpha_prediction"]
    forecast_model = _committee_state.get("forecast_model", "chronos")
    if hasattr(pred, "model_dump"):
        payload = pred.model_dump()
    else:
        payload = dict(pred)
    payload.setdefault("forecast_model", forecast_model)
    return payload


@tool
def think(reasoning: str) -> str:
    """Internal scratchpad for reasoning. Use this to organize your thoughts
    before submitting your final view."""
    return f"Noted: {reasoning}"


# ── Yahoo Finance tools ────────────────────────────────────────────────────


@tool
def get_price_history(ticker: str, period: str = "6mo") -> dict:
    """Fetch OHLCV price history. Period: 1mo, 3mo, 6mo, 1y, 2y, 5y."""
    try:
        import yfinance as yf
        df = yf.download(ticker, period=period, progress=False)
        if df.empty:
            return {"error": f"No price data for {ticker}"}
        return {
            "ticker": ticker,
            "rows": len(df),
            "latest_close": float(df["Close"].iloc[-1].iloc[0]) if hasattr(df["Close"].iloc[-1], "iloc") else float(df["Close"].iloc[-1]),
            "period_return": float((df["Close"].iloc[-1] / df["Close"].iloc[0] - 1).iloc[0]) if hasattr(df["Close"].iloc[-1], "iloc") else float(df["Close"].iloc[-1] / df["Close"].iloc[0] - 1),
            "period_high": float(df["High"].max().iloc[0]) if hasattr(df["High"].max(), "iloc") else float(df["High"].max()),
            "period_low": float(df["Low"].min().iloc[0]) if hasattr(df["Low"].min(), "iloc") else float(df["Low"].min()),
        }
    except Exception as e:
        return {"error": str(e), "source": "yahoo_finance"}


@tool
def get_fundamentals(ticker: str) -> dict:
    """Fetch company fundamentals: P/E, EV/EBITDA, margins, growth, etc."""
    try:
        if _yahoo_finance_service is None:
            from backend.app.services.data.yahoo_finance import YahooFinanceService
            svc = YahooFinanceService()
        else:
            svc = _yahoo_finance_service
        return svc.get_fundamentals(ticker)
    except Exception as e:
        return {"error": str(e), "source": "yahoo_finance"}


@tool
def get_financial_statements(ticker: str) -> dict:
    """Fetch income statement, balance sheet, cash flow from Yahoo Finance."""
    try:
        if _yahoo_finance_service is None:
            from backend.app.services.data.yahoo_finance import YahooFinanceService
            svc = YahooFinanceService()
        else:
            svc = _yahoo_finance_service
        return svc.get_financial_statements(ticker)
    except Exception as e:
        return {"error": str(e), "source": "yahoo_finance"}


@tool
def get_analyst_estimates(ticker: str) -> dict:
    """Fetch analyst recommendations, target prices, earnings estimates."""
    try:
        if _yahoo_finance_service is None:
            from backend.app.services.data.yahoo_finance import YahooFinanceService
            svc = YahooFinanceService()
        else:
            svc = _yahoo_finance_service
        return svc.get_analyst_estimates(ticker)
    except Exception as e:
        return {"error": str(e), "source": "yahoo_finance"}


@tool
def get_options_data(ticker: str) -> dict:
    """Fetch options data: implied volatility, put/call ratio."""
    try:
        if _yahoo_finance_service is None:
            from backend.app.services.data.yahoo_finance import YahooFinanceService
            svc = YahooFinanceService()
        else:
            svc = _yahoo_finance_service
        return svc.get_options_data(ticker)
    except Exception as e:
        return {"error": str(e), "source": "yahoo_finance"}


@tool
def get_short_interest(ticker: str) -> dict:
    """Fetch short interest: short ratio, % of float, shares short."""
    try:
        if _yahoo_finance_service is None:
            from backend.app.services.data.yahoo_finance import YahooFinanceService
            svc = YahooFinanceService()
        else:
            svc = _yahoo_finance_service
        return svc.get_short_interest(ticker)
    except Exception as e:
        return {"error": str(e), "source": "yahoo_finance"}


# ── Search tools ────────────────────────────────────────────────────────────


@tool
def search_web(query: str, count: int = 5) -> dict:
    """Search the web for financial news using Brave Search."""
    try:
        if _brave_search_client is None:
            return {"error": "Brave Search client not configured"}
        return asyncio.get_event_loop().run_until_complete(
            _brave_search_client.search(query, count=count)
        )
    except Exception as e:
        return {"error": str(e), "source": "brave_search", "results": []}


@tool
def search_reddit(keyword: str, num_posts: int = 5) -> dict:
    """Search Reddit for stock discussions via Bright Data."""
    try:
        if _brightdata_client is None:
            return {"error": "Bright Data client not configured"}
        snapshot_id = asyncio.get_event_loop().run_until_complete(
            _brightdata_client.search_reddit(keyword, num_posts=num_posts)
        )
        return {"snapshot_id": snapshot_id, "status": "triggered", "keyword": keyword}
    except Exception as e:
        return {"error": str(e), "source": "bright_data", "results": []}


# ── Macro tools ─────────────────────────────────────────────────────────────


@tool
def get_yield_curve() -> dict:
    """Fetch current yield curve data from FRED: 3M, 2Y, 10Y, Fed Funds."""
    try:
        if _fred_service is None:
            return {"error": "FRED service not configured"}
        return _fred_service.get_yield_curve_snapshot()
    except Exception as e:
        return {"error": str(e), "source": "fred"}


@tool
def get_macro_data() -> dict:
    """Fetch macro market data: VIX, SPY, TLT, GLD, DXY, credit spreads."""
    try:
        import yfinance as yf
        tickers = ["SPY", "^VIX", "TLT", "GLD", "HYG", "LQD"]
        data = yf.download(tickers, period="5d", group_by="ticker", progress=False)
        result = {}
        for t in tickers:
            try:
                if t in data.columns.get_level_values(0):
                    close = data[t]["Close"].dropna()
                    if len(close) > 0:
                        result[t] = {
                            "latest": float(close.iloc[-1]),
                            "change_1d": float(close.iloc[-1] / close.iloc[-2] - 1) if len(close) > 1 else 0,
                        }
            except Exception:
                continue
        # Credit spread
        if "HYG" in result and "LQD" in result:
            result["hyg_lqd_spread"] = result["HYG"]["latest"] - result["LQD"]["latest"]
        return result
    except Exception as e:
        return {"error": str(e), "source": "yahoo_finance"}


# ── Enhanced Sentiment tools ───────────────────────────────────────────────


def _decay_weight(article_date: date, now: date, half_life_days: float = 3.0) -> float:
    """Temporal decay: recent articles weighted higher. Half-life = 3 days."""
    age = (now - article_date).days
    return math.exp(-math.log(2) * age / half_life_days)


@tool
def search_news_with_extraction(ticker: str, company_name: str = "") -> dict:
    """Search web for financial news, extract article text with trafilatura,
    and apply temporal decay weighting. Returns top articles with content."""
    try:
        if _brave_search_client is None:
            return {"error": "Brave Search client not configured"}

        queries = [
            f"{ticker} stock news today",
            f"{ticker} earnings analyst rating",
        ]
        if company_name:
            queries.append(f"{ticker} {company_name} market outlook")

        all_results = []
        for q in queries:
            try:
                resp = asyncio.get_event_loop().run_until_complete(
                    _brave_search_client.search(q, count=5, freshness="pw")
                )
                all_results.extend(resp.get("results", []))
            except Exception as e:
                logger.warning(f"Search query '{q}' failed: {e}")

        # Deduplicate by URL
        seen_urls = set()
        unique_results = []
        for r in all_results:
            url = r.get("url", "")
            if url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(r)

        # Extract article text with trafilatura
        articles = []
        for r in unique_results[:10]:  # max 10 articles
            article = {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "description": r.get("description", ""),
                "age": r.get("age", ""),
            }

            # Try to extract full article text
            try:
                import trafilatura
                import httpx

                resp = httpx.get(r["url"], timeout=5.0, follow_redirects=True)
                content = trafilatura.extract(
                    resp.text,
                    include_comments=False,
                    include_tables=False,
                )
                if content:
                    article["content"] = content[:2000]  # ~500 tokens
            except Exception:
                pass  # Use description as fallback

            # Parse age for decay weighting
            age_str = r.get("age", "")
            weight = 1.0
            try:
                if "hour" in age_str:
                    weight = 1.0  # Same day
                elif "day" in age_str:
                    days = int("".join(c for c in age_str if c.isdigit()) or "1")
                    weight = _decay_weight(
                        date.today() - __import__("datetime").timedelta(days=days),
                        date.today(),
                    )
            except Exception:
                pass
            article["decay_weight"] = round(weight, 3)

            articles.append(article)

        # Sort by decay weight (most recent first)
        articles.sort(key=lambda a: a.get("decay_weight", 0), reverse=True)

        return {
            "ticker": ticker,
            "n_articles": len(articles),
            "articles": articles,
        }
    except Exception as e:
        return {"error": str(e), "source": "brave_search", "results": []}


# ── Relative Valuation tool ────────────────────────────────────────────────


@tool
def get_relative_valuation(ticker: str) -> dict:
    """Compute EV/EBITDA, P/E, P/B percentile ranks vs sector median using Yahoo Finance."""
    try:
        if _yahoo_finance_service is None:
            from backend.app.services.data.yahoo_finance import YahooFinanceService
            svc = YahooFinanceService()
        else:
            svc = _yahoo_finance_service

        target = svc.get_fundamentals(ticker)
        sector = target.get("sector", "")

        from backend.app.services.analysis.valuation import (
            compute_percentile_ranks,
            get_sector_constituents,
        )
        peer_tickers = get_sector_constituents(sector, svc)
        # Remove target from peers
        peer_tickers = [t for t in peer_tickers if t != ticker][:20]

        peer_data = {}
        for t in peer_tickers:
            try:
                peer_data[t] = svc.get_fundamentals(t)
            except Exception:
                continue

        return compute_percentile_ranks(target, peer_data)
    except Exception as e:
        return {"error": str(e), "source": "yahoo_finance"}


# ── Risk analysis tools ────────────────────────────────────────────────────


@tool
def get_risk_metrics(ticker: str) -> dict:
    """Compute VaR, CVaR, and stress test results for a stock."""
    try:
        import yfinance as yf
        import pandas as pd
        from backend.app.services.analysis.risk import compute_var_cvar, stress_test

        # Get price history for VaR
        df = yf.download(ticker, period="2y", progress=False)
        if df.empty:
            return {"error": f"No price data for {ticker}"}

        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        returns = close.pct_change().dropna()

        var_cvar = compute_var_cvar(returns)

        # Stress test using SPY
        spy = yf.download("SPY", start="2008-01-01", end=str(date.today()), progress=False)
        spy_close = spy["Close"]
        if isinstance(spy_close, pd.DataFrame):
            spy_close = spy_close.iloc[:, 0]
        stress = stress_test(spy_close)

        return {"var_cvar": var_cvar, "stress_test": stress}
    except Exception as e:
        return {"error": str(e), "source": "risk_analysis"}


@tool
def get_regime(ticker: str = "") -> dict:
    """Detect current macro regime (risk_on/transition/risk_off) using HMM."""
    try:
        import yfinance as yf
        import numpy as np

        # Fetch VIX, HYG, LQD, and yield data
        vix_df = yf.download("^VIX", period="2y", progress=False)
        hyg_df = yf.download("HYG", period="2y", progress=False)
        lqd_df = yf.download("LQD", period="2y", progress=False)

        for frame_name, frame in [("VIX", vix_df), ("HYG", hyg_df), ("LQD", lqd_df)]:
            if frame.empty:
                return {"error": f"No data for {frame_name}"}

        import pandas as pd
        vix_close = vix_df["Close"]
        hyg_close = hyg_df["Close"]
        lqd_close = lqd_df["Close"]
        if isinstance(vix_close, pd.DataFrame):
            vix_close = vix_close.iloc[:, 0]
        if isinstance(hyg_close, pd.DataFrame):
            hyg_close = hyg_close.iloc[:, 0]
        if isinstance(lqd_close, pd.DataFrame):
            lqd_close = lqd_close.iloc[:, 0]

        spread = hyg_close - lqd_close
        # Use a simple yield curve proxy (VIX as regime indicator)
        yield_proxy = np.zeros(len(vix_close))  # simplified

        from backend.app.services.analysis.macro import fit_regime_model, predict_regime

        vix_arr = vix_close.values.flatten()
        spread_arr = spread.reindex(vix_close.index).fillna(method="ffill").values.flatten()
        yield_arr = yield_proxy

        hmm, state_order = fit_regime_model(vix_arr, spread_arr, yield_arr)

        # Predict current regime
        result = predict_regime(
            hmm, state_order,
            vix=float(vix_arr[-1]),
            hyg_lqd_spread=float(spread_arr[-1]),
            yield_2s10s=0.0,
            recent_history=np.column_stack([vix_arr[-20:], spread_arr[-20:], yield_arr[-20:]]),
        )
        return result
    except Exception as e:
        return {"error": str(e), "source": "hmm_regime"}


# ── Tool registry per agent ─────────────────────────────────────────────────

SHARED_TOOLS = [get_alpha_prediction, think]

AGENT_TOOLS = {
    "quant": SHARED_TOOLS + [get_price_history, get_fundamentals],
    "fundamentals": SHARED_TOOLS + [get_fundamentals, get_financial_statements, get_analyst_estimates, get_relative_valuation],
    "sentiment": SHARED_TOOLS + [search_web, search_news_with_extraction, search_reddit, get_price_history],
    "risk": SHARED_TOOLS + [get_options_data, get_short_interest, get_price_history, get_fundamentals, get_risk_metrics],
    "macro": SHARED_TOOLS + [get_yield_curve, get_macro_data, get_price_history, get_regime],
}
