# BAM: Bloomberg AI Multi-Agent Investment Committee

## Context
Two-layer architecture:
1. **Prediction Layer**: PatchTST + cross-channel mixer → sector-neutralized alpha signal at 1d/5d/21d/63d horizons with calibrated uncertainty.
2. **Analysis Layer**: LangGraph committee of 5 ReAct agents → qualitative context, risk framing, narrative → Investment Memo.

**Design principle**: Transformer generates the quantitative signal. Agents provide context a PM needs to act on it. ALL agents see the alpha prediction. Agents DO NOT improve the signal.

**NO MOCKS, NO PLACEHOLDERS, NO FALLBACKS.** Every component is real implementation from Phase 1.

## Hardware + Services
- **M1 MacBook** — runs FastAPI backend, React frontend, Redis, PatchTST inference (CPU/MPS)
- **Google Colab (T4 GPU)** — PatchTST training only (9 walk-forward folds, ~30min/fold)
- **NVIDIA Nemotron API** — LLM for all 5 agents (120B model, OpenAI-compatible, ~8s/response, 0.33s TTFT)
- **No Bloomberg** — Yahoo Finance for all price/fundamental data
- **No local GPU needed** — LLM is cloud API, training is Colab

## Prerequisites
- Python 3.13 (available on machine)
- Node.js ≥20 (available on machine)
- Redis server
- Google Colab account (free tier T4 GPU sufficient)
- API keys (all verified working):
  - NVIDIA Nemotron: `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`
  - Brave Search: `BSAcJOx5z1SLdKAOOWbyVA7oONbpTOl`
  - Bright Data (Reddit): `e0025823-51a5-4ec0-b6b0-a005c9367b2c`
  - FRED: `d69d7cac0a2640886f05e0d29c6ac6ec`
  - W&B: (to be set)

## Tech Stack (pinned versions)
```
# Backend
python = ">=3.12"                      # 3.13 preferred, 3.12 fallback if deps lack wheels
fastapi = ">=0.115.0,<0.116"
uvicorn = {extras = ["standard"], version = ">=0.32.0"}
pydantic = ">=2.9.0,<3.0"
pydantic-settings = ">=2.5.0"
langgraph = ">=0.3.0,<0.4"          # PINNED — API changed between 0.2→0.3
langchain-core = ">=0.3.0,<0.4"
langchain-community = ">=0.3.0"
openai = ">=1.0.0"                  # NVIDIA Nemotron API (OpenAI-compatible)
httpx = ">=0.27.0"
redis = {extras = ["hiredis"], version = ">=5.0.0"}
opentelemetry-api = ">=1.27.0"
opentelemetry-sdk = ">=1.27.0"

# Alpha model (training on Colab, inference on M1)
torch = ">=2.4.0"
numpy = ">=1.26.0"
pandas = ">=2.2.0"
wandb = ">=0.18.0"
hmmlearn = ">=0.3.2"                # HMM regime detection
fredapi = ">=0.5.0"                 # FRED economic data
yfinance = ">=0.2.36"               # Yahoo Finance — ALL price + fundamental data
waybackpy = ">=3.0.0"               # Wayback Machine for historical S&P 500 constituents
pandera = ">=0.20.0"                # DataFrame schema validation at pipeline boundaries
cachetools = ">=5.3.0"              # TTL cache for yfinance API calls

# Sentiment
trafilatura = ">=1.12.0"            # article text extraction
# Reddit via Bright Data API (httpx) — no praw needed

# Testing
pytest = ">=8.3.0"
pytest-asyncio = ">=0.24.0"
pytest-cov = ">=5.0.0"

# Frontend
node = ">=20.0.0"
react = "18.3.x"
typescript = "5.5.x"
vite = "5.x"
tailwindcss = "3.4.x"
@radix-ui/react-* (component primitives)
shadcn/ui (component library on top of Radix)
zustand = "5.x"
@tanstack/react-query = "5.x"
lightweight-charts = "4.x"
vitest = "2.x"                      # frontend testing
@testing-library/react = "16.x"
```

## Key Schemas (contracts — defined before any implementation)

```python
class AlphaPrediction(BaseModel):
    ticker: str
    prediction_date: date
    sector: str
    sector_etf: str
    # Point estimates (50th percentile)
    alpha_1d: float; alpha_5d: float; alpha_21d: float; alpha_63d: float
    # Quantile bands
    q10_1d: float; q90_1d: float
    q10_5d: float; q90_5d: float
    q10_21d: float; q90_21d: float
    q10_63d: float; q90_63d: float
    # Interpretability
    patch_attention: list[PatchAttention]
    top_features: list[FeatureContribution]
    # Metadata
    model_version: str; training_fold: str; inference_latency_ms: float

class AgentView(BaseModel):
    agent_name: str; ticker: str
    alpha_seen: AlphaPredictionSummary    # proves agent read the signal
    direction: Literal["BULLISH","BEARISH","NEUTRAL"]
    conviction: float                      # 0-1
    time_horizon: Literal["1D","1W","1M","1Q"]
    agrees_with_alpha: bool
    key_claims: list[Claim]                # max 5 claims
    supporting_evidence: list[Evidence]    # max 8 evidence items
    risks: list[str]                       # max 5
    summary: str                           # max 200 tokens

class AgentDebateResponse(BaseModel):
    agent_name: str
    agreements: list[ClaimReference]
    disagreements: list[Disagreement]
    revised_conviction: float
    revised_direction: Optional[Literal["BULLISH","BEARISH","NEUTRAL"]]

class InvestmentMemo(BaseModel):
    ticker: str; date: date
    alpha_prediction: AlphaPrediction
    alpha_decay_halflife_days: Optional[float]
    factor_r2: float
    recommendation: Literal["STRONG_BUY","BUY","HOLD","SELL","STRONG_SELL"]
    confidence: float
    recommended_horizon: str
    position_size_pct: float              # capped at 10%
    quant_summary: str; fundamentals_summary: str; sentiment_summary: str
    risk_summary: str; macro_summary: str
    consensus_claims: list[str]
    dissenting_opinions: list[DissentingOpinion]
    upcoming_events: list[str]
    stress_test_worst_case: float
    current_regime: str
```

## Sector ETF Mapping
```
Energy=XLE, Materials=XLB, Industrials=XLI, ConsDisc=XLY, ConsStaples=XLP,
HealthCare=XLV, Financials=XLF, IT=XLK, CommServices=XLC*, Utilities=XLU, RealEstate=XLRE

*XLC launched 2018-06. Pre-2018: use constituent-weighted avg return of comm stocks.
```

---

## Phase 1: Infrastructure + Project Scaffold + Schemas

### Purpose
Set up the entire project structure, install all real dependencies, verify every external service connection (NVIDIA Nemotron API, Redis, Brave Search, Bright Data Reddit, FRED, Yahoo Finance), and define all Pydantic contracts. If a service isn't available, the app fails loudly at startup.

### Exact steps:
```bash
# 1. Environment
cd /Users/mrunmayeerane/Desktop/Mrun/BAM
git init
python3.13 -m venv .venv && source .venv/bin/activate

# 2. Backend scaffold
mkdir -p backend/app/{api/endpoints,models,services/{data,llm,search,prediction,analysis},agents,memo,backtest,core}
mkdir -p backend/tests/{test_agents,test_services,test_api}
mkdir -p alpha_model/{model,data/{raw,processed},training,evaluation}
mkdir -p data/{raw,processed}
mkdir -p models

# 3. Frontend scaffold
npm create vite@latest frontend -- --template react-ts
cd frontend && npm install && npx shadcn-ui@latest init
npx tailwindcss init -p
npm install zustand @tanstack/react-query lightweight-charts react-router-dom lucide-react axios
npm install -D vitest @testing-library/react @testing-library/jest-dom

# 4. Start Redis
redis-server --appendonly yes --daemonize yes
```

### Files created in Phase 1:

**`backend/pyproject.toml`** — all deps pinned as above. `[tool.pytest.ini_options]` configured. `[tool.ruff]` configured.

**`.env`** — all keys required, no defaults:
```bash
# NVIDIA Nemotron API (OpenAI-compatible)
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_API_KEY=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
NVIDIA_MODEL=nvidia/nemotron-3-super-120b-a12b
# Redis
REDIS_URL=redis://localhost:6379/0
# Brave Search
BRAVE_API_KEY=BSAcJOx5z1SLdKAOOWbyVA7oONbpTOl
# Bright Data (Reddit scraping)
BRIGHTDATA_API_KEY=e0025823-51a5-4ec0-b6b0-a005c9367b2c
BRIGHTDATA_DATASET_ID=gd_lvz8ah06191smkebj4
# FRED
FRED_API_KEY=d69d7cac0a2640886f05e0d29c6ac6ec
# W&B
WANDB_API_KEY=<real_key>
WANDB_PROJECT=bam-alpha
# PatchTST
MODEL_PATH=./models/patch_tst_fold9.pt
# Observability
OTEL_ENDPOINT=http://localhost:4317
```

**`backend/app/config.py`**:
```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    # NVIDIA Nemotron API — required
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_api_key: str
    nvidia_model: str = "nvidia/nemotron-3-super-120b-a12b"
    # Redis — required
    redis_url: str
    redis_session_ttl_seconds: int = 86400  # 24 hours
    # Brave Search — required
    brave_api_key: str
    # Bright Data (Reddit) — required
    brightdata_api_key: str
    brightdata_dataset_id: str = "gd_lvz8ah06191smkebj4"
    # FRED — required
    fred_api_key: str
    # PatchTST — required, must point to trained model
    model_path: str
    # Observability
    otel_endpoint: str = "http://localhost:4317"

    @field_validator("nvidia_api_key", "redis_url", "brave_api_key", "fred_api_key", "model_path")
    @classmethod
    def must_not_be_empty(cls, v: str, info) -> str:
        if not v.strip():
            raise ValueError(f"{info.field_name} must be set — no defaults, no empty values")
        return v
```

**`backend/app/main.py`**:
```python
from contextlib import asynccontextmanager
from openai import AsyncOpenAI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup — verify ALL services or fail
    # 1. Redis
    app.state.redis = await create_redis_pool(settings.redis_url)
    await app.state.redis.ping()  # fail loud if Redis is down

    # 2. NVIDIA Nemotron API health check
    app.state.llm_client = AsyncOpenAI(
        base_url=settings.nvidia_base_url,
        api_key=settings.nvidia_api_key,
    )
    # Quick test call to verify API key works
    test = await app.state.llm_client.chat.completions.create(
        model=settings.nvidia_model,
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=5,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    assert test.choices[0].message.content, "Nemotron API not responding"

    # 3. Load trained PatchTST model (CPU inference on M1)
    app.state.alpha_model = AlphaModel.load(settings.model_path, device="cpu")

    # 4. Brave Search client
    app.state.brave_client = BraveSearchClient(settings.brave_api_key)

    # 5. Bright Data client (Reddit)
    app.state.brightdata_client = BrightDataClient(settings.brightdata_api_key, settings.brightdata_dataset_id)

    yield
    # Shutdown
    await app.state.redis.close()

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], ...)
app.include_router(api_router, prefix="/api/v1")
```

**`backend/app/models/`** — ALL Pydantic schemas from Key Schemas section above. Every field typed. Every schema importable. These are the contracts.

**`backend/app/services/data/yahoo_finance.py`** — Yahoo Finance service (replaces Bloomberg):
```python
import yfinance as yf
from functools import lru_cache

class YahooFinanceService:
    """All financial data via Yahoo Finance. No Bloomberg needed."""

    def get_price_history(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """OHLCV daily data."""
        return yf.download(ticker, start=start, end=end, progress=False)

    def get_fundamentals(self, ticker: str) -> dict:
        """Company info, P/E, EV/EBITDA, margins, etc."""
        t = yf.Ticker(ticker)
        info = t.info
        return {
            "sector": info.get("sector"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "ev_ebitda": info.get("enterpriseToEbitda"),
            "price_to_book": info.get("priceToBook"),
            "profit_margin": info.get("profitMargins"),
            "roe": info.get("returnOnEquity"),
            "debt_to_equity": info.get("debtToEquity"),
            "free_cash_flow": info.get("freeCashflow"),
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
        }

    def get_financial_statements(self, ticker: str) -> dict:
        """Income statement, balance sheet, cash flow."""
        t = yf.Ticker(ticker)
        return {
            "income_stmt": t.income_stmt.to_dict() if not t.income_stmt.empty else {},
            "balance_sheet": t.balance_sheet.to_dict() if not t.balance_sheet.empty else {},
            "cashflow": t.cashflow.to_dict() if not t.cashflow.empty else {},
        }

    def get_analyst_estimates(self, ticker: str) -> dict:
        """Analyst recommendations and earnings estimates."""
        t = yf.Ticker(ticker)
        return {
            "recommendations": t.recommendations.to_dict() if t.recommendations is not None else {},
            "earnings_estimate": t.earnings_estimate.to_dict() if t.earnings_estimate is not None else {},
            "target_price": t.info.get("targetMeanPrice"),
            "recommendation_key": t.info.get("recommendationKey"),
            "num_analysts": t.info.get("numberOfAnalystOpinions"),
        }

    def get_options_data(self, ticker: str) -> dict:
        """Options chain for implied vol and put/call ratio."""
        t = yf.Ticker(ticker)
        dates = t.options  # expiration dates
        if not dates:
            return {"implied_vol": None, "put_call_ratio": None}
        # Use nearest expiration
        chain = t.option_chain(dates[0])
        total_call_oi = chain.calls["openInterest"].sum()
        total_put_oi = chain.puts["openInterest"].sum()
        avg_call_iv = chain.calls["impliedVolatility"].mean()
        return {
            "put_call_ratio": total_put_oi / max(total_call_oi, 1),
            "implied_vol": avg_call_iv,
            "expiration": dates[0],
        }

    def get_short_interest(self, ticker: str) -> dict:
        """Short interest data from yfinance info."""
        info = yf.Ticker(ticker).info
        return {
            "short_ratio": info.get("shortRatio"),
            "short_pct_float": info.get("shortPercentOfFloat"),
            "shares_short": info.get("sharesShort"),
        }
```

**`backend/app/services/llm/nemotron_client.py`** — NVIDIA Nemotron client:
```python
from openai import AsyncOpenAI

class NemotronClient:
    """Wraps OpenAI-compatible calls to NVIDIA Nemotron 120B API."""

    def __init__(self, client: AsyncOpenAI, model: str):
        self.client = client
        self.model = model

    async def chat_completion(self, messages: list[dict],
                               stream: bool = True, max_tokens: int = 3000,
                               temperature: float = 0.7) -> Any:
        """Non-streaming completion for structured JSON output."""
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            stream=stream,
        )
        if stream:
            return resp  # caller iterates
        return resp.choices[0].message.content

    async def chat_completion_json(self, messages: list[dict],
                                    max_tokens: int = 3000) -> dict:
        """Get JSON response, validate, retry on parse failure (max 2x)."""
        messages = [m.copy() for m in messages]  # never mutate caller's list
        for attempt in range(3):
            content = await self.chat_completion(messages, stream=False, max_tokens=max_tokens)
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                if attempt < 2:
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": f"Invalid JSON: {e}. Return valid JSON only."})
                else:
                    raise

    async def health_check(self) -> bool:
        resp = await self.chat_completion(
            [{"role": "user", "content": "ping"}], stream=False, max_tokens=5
        )
        return resp is not None
```

**`backend/app/services/search/brightdata_reddit.py`** — Bright Data Reddit scraper:
```python
import httpx

class BrightDataClient:
    """Reddit scraping via Bright Data Datasets API."""

    BASE_URL = "https://api.brightdata.com/datasets/v3"

    def __init__(self, api_key: str, dataset_id: str):
        self.api_key = api_key
        self.dataset_id = dataset_id
        self.client = httpx.AsyncClient(timeout=30.0)

    async def search_reddit(self, keyword: str, num_posts: int = 10,
                             sort_by: str = "Hot", date_range: str = "Past month") -> str:
        """Trigger Reddit scrape by keyword. Returns snapshot_id."""
        resp = await self.client.post(
            f"{self.BASE_URL}/trigger",
            params={
                "dataset_id": self.dataset_id,
                "notify": "false",
                "include_errors": "true",
                "type": "discover_new",
                "discover_by": "keyword",
            },
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"input": [{"keyword": keyword, "date": date_range,
                              "num_of_posts": num_posts, "sort_by": sort_by}]},
        )
        resp.raise_for_status()
        return resp.json()["snapshot_id"]

    async def get_snapshot(self, snapshot_id: str) -> list[dict]:
        """Poll for completed snapshot data."""
        resp = await self.client.get(
            f"{self.BASE_URL}/snapshot/{snapshot_id}",
            params={"format": "json"},
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        if resp.status_code == 202:
            return []  # still processing
        resp.raise_for_status()
        return resp.json()

    async def search_subreddit(self, subreddit_url: str, keyword: str = "",
                                sort_by: str = "New", num_posts: int = 10) -> str:
        """Trigger Reddit scrape by subreddit URL."""
        resp = await self.client.post(
            f"{self.BASE_URL}/trigger",
            params={
                "dataset_id": self.dataset_id,
                "notify": "false",
                "include_errors": "true",
                "type": "discover_new",
                "discover_by": "subreddit_url",
            },
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"input": [{"url": subreddit_url, "sort_by": sort_by,
                              "keyword": keyword, "start_date": ""}]},
        )
        resp.raise_for_status()
        return resp.json()["snapshot_id"]
```

**`backend/app/agents/base.py`** — BaseAgent with shared tools:
```python
# Shared tools ALL agents get:
@tool
def get_alpha_prediction() -> dict:
    """Read the PatchTST alpha prediction from committee state."""
    state = get_current_state()
    pred = state["alpha_prediction"]
    return pred.model_dump()

@tool
def think(reasoning: str) -> str:
    """Internal scratchpad for reasoning."""
    return f"Noted: {reasoning}"

@tool
def submit_view(view_json: str) -> str:
    """Submit final structured view. Must be valid AgentView JSON."""
    view = AgentView.model_validate_json(view_json)
    return f"View submitted: {view.direction} with conviction {view.conviction}"
```

**`backend/app/agents/orchestrator.py`** — Full LangGraph StateGraph (all 3 rounds):
```python
from langgraph.graph import StateGraph, Send, END

class CommitteeState(TypedDict):
    ticker: str
    alpha_prediction: AlphaPrediction | None
    quant_view: AgentView | None
    fundamentals_view: AgentView | None
    sentiment_view: AgentView | None
    risk_view: AgentView | None
    macro_view: AgentView | None
    extracted_claims: dict | None
    quant_debate: AgentDebateResponse | None
    fundamentals_debate: AgentDebateResponse | None
    sentiment_debate: AgentDebateResponse | None
    risk_debate: AgentDebateResponse | None
    macro_debate: AgentDebateResponse | None
    memo: InvestmentMemo | None
    status: str
    trace: list[TraceEvent]

graph = StateGraph(CommitteeState)
graph.add_node("predict_alpha", predict_alpha_node)
for name in AGENT_NAMES:
    graph.add_node(f"agent_{name}", make_agent_node(name))
graph.add_node("extract_claims", extract_claims_node)
graph.add_node("debate_router", debate_router_node)
for name in AGENT_NAMES:
    graph.add_node(f"debate_{name}", make_debate_node(name))
graph.add_node("synthesize_memo", synthesize_memo_node)

graph.set_entry_point("predict_alpha")
for name in AGENT_NAMES:
    graph.add_edge("predict_alpha", f"agent_{name}")
for name in AGENT_NAMES:
    graph.add_edge(f"agent_{name}", "extract_claims")
graph.add_conditional_edges("extract_claims", debate_router_fn)
for name in AGENT_NAMES:
    graph.add_edge(f"debate_{name}", "synthesize_memo")
graph.add_edge("synthesize_memo", END)
```

**`backend/app/api/endpoints/analysis.py`**:
- POST /analysis: creates UUID, stores session in Redis (TTL=1hr), launches full StateGraph
- GET /analysis/{id}: reads from Redis, returns current state with `status` field
- GET /analysis/{id}/memo: returns InvestmentMemo when complete
- GET /analysis/{id}/trace: returns full tool call trace

**`backend/tests/conftest.py`**:
```python
# Tests use REAL services — require running Redis + NVIDIA API reachable
@pytest.fixture
def settings():
    return Settings()  # loads from .env

@pytest.fixture
async def redis_client(settings):
    client = await create_redis_pool(settings.redis_url)
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
async def app_client(settings):
    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        yield client
```

**`frontend/src/App.tsx`** — React Router:
```
/ → DashboardPage
/analysis/:id → DashboardPage (with analysis loaded)
/memo/:id → MemoPage
/backtest/:id → BacktestPage
```

**`frontend/src/store/analysisStore.ts`**:
```typescript
interface AnalysisStore {
  ticker: string | null
  analysisId: string | null
  status: "idle" | "predicting" | "round_1" | "round_2" | "round_3" | "complete" | "error"
  alphaPrediction: AlphaPrediction | null
  agents: Record<AgentName, { status: string; streamedText: string; view: AgentView | null }>
  debateResponses: AgentDebateResponse[]
  memo: InvestmentMemo | null
  startAnalysis: (ticker: string) => Promise<void>
  reset: () => void
}
```

**Milestone test** (requires Redis + internet):
```bash
# Verify all services
redis-cli ping                                      # PONG
python -c "from openai import OpenAI; print('OK')"  # openai installed
python -c "import yfinance; print('OK')"            # yfinance installed

# Backend
pytest backend/tests/test_services/test_nemotron.py -v
# → Sends real prompt to Nemotron API, gets structured JSON back
pytest backend/tests/test_services/test_yahoo.py -v
# → Fetches real AAPL data from Yahoo Finance
pytest backend/tests/test_services/test_brave.py -v
# → Searches Brave, returns news results
```

---

## Phase 2: Data Pipeline — Real Historical Data

### Data pipeline micro details:

**`alpha_model/data/download_universe.py`** — Survivorship-bias-free S&P 500 constituents:
```python
import waybackpy
from bs4 import BeautifulSoup
import pandas as pd

def get_sp500_constituents_at_date(target_date: str) -> list[str]:
    """Fetch S&P 500 list as it existed at target_date via Wayback Machine."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    user_agent = "BAM/1.0"
    wayback = waybackpy.Url(url, user_agent)
    # Get the closest archived snapshot to target_date
    archive = wayback.near(year=int(target_date[:4]),
                           month=int(target_date[5:7]),
                           day=int(target_date[8:10]))
    html = archive.page  # full HTML of archived page
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", {"id": "constituents"})
    tickers = []
    for row in table.find_all("tr")[1:]:
        cells = row.find_all("td")
        if cells:
            ticker = cells[0].text.strip().replace(".", "-")  # BRK.B → BRK-B (yfinance format)
            tickers.append(ticker)
    return tickers

# Build yearly snapshots: Jan 1 of each year from 2010 to 2024
yearly_constituents = {}
for year in range(2010, 2025):
    target = f"{year}-01-15"  # mid-January to ensure Wikipedia is updated
    tickers = get_sp500_constituents_at_date(target)
    yearly_constituents[year] = tickers
    print(f"{year}: {len(tickers)} constituents")

# Validate: each year should have ~500 tickers
# Cross-validate: compare consecutive years — additions/removals should be <50/year
# Save: data/raw/sp500_constituents.parquet (year, ticker, added_date, removed_date)
```

- Download OHLCV via `yfinance.download(tickers, start="2009-01-01", end="2024-12-31")` — start 2009 to allow 250d warm-up before 2010 training data begins.
- For any ticker where yfinance has gaps: forward-fill up to **2 days** (not 5), then mark as NaN. Drop ticker-dates with >5% NaN rate.
- Validate: tickers with <100 rows total are excluded (delisted/insufficient data)
- Save: `data/raw/prices.parquet` (ticker, date, open, high, low, close, adj_close, volume)

**`download_factors.py`**:
- Download from `https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip`
- Parse: skip header rows, handle date format (YYYYMMDD), divide by 100 (data is in percentage)
- Also download Momentum factor: `F-F_Momentum_Factor_daily_CSV.zip`
- Save: `data/raw/ff_factors.parquet` (date, Mkt-RF, SMB, HML, RMW, CMA, Mom, RF)

**`download_macro.py`**:
- yfinance: SPY, ^VIX, TLT, GLD, DX-Y.NYB, HYG, LQD + all sector ETFs (XLK, XLF, etc.)
- FRED API (via `fredapi`): DGS10 (10Y yield), DGS2 (2Y yield), DGS3MO (3M yield), FEDFUNDS
- Save: `data/raw/macro.parquet`

**`build_features.py`** — strict point-in-time:
```python
# For each ticker, for each date:
# 1. Raw OHLC as % change from prior close (4 channels)
pct_open  = (open - prev_close) / prev_close
pct_high  = (high - prev_close) / prev_close
pct_low   = (low  - prev_close) / prev_close
pct_close = (close - prev_close) / prev_close  # = daily return

# 2. Volume (2 channels)
log_volume = log(volume + 1)
volume_ratio = volume / volume.rolling(20).mean()

# 3. Volatility (3 channels)
realized_vol_21d = returns.rolling(21).std() * sqrt(252)
garman_klass = sqrt(252 * rolling_mean(0.5*(log(H/L))^2 - (2*log(2)-1)*(log(C/O))^2, 21))
vol_of_vol = realized_vol_21d.rolling(21).std()

# 4. Factor betas (6 channels) — rolling 63d OLS
for factor in [MktRF, SMB, HML, RMW, CMA, Mom]:
    beta = rolling_ols(stock_excess_return, factor, window=63).params[factor]

# 5. Cross-asset (6 channels)
spy_return, vix_level, vix_change, yield_10y_change, dxy_change, hyg_lqd_spread_change

# 7. Sector-neutralized returns (2 channels) — CRITICAL to prevent sector beta leakage
sector_neutral_return_1d = stock_return - sector_etf_return
sector_neutral_return_5d = (stock_return_5d - sector_etf_return_5d)

# TOTAL: 23 channels × 250 days per sample (was 21, added 2 sector-neutral channels)
# First valid sample: day 313 (250 context + 63 factor warm-up)
# Save: data/processed/features.parquet (ticker, date, feature_name, value)
```

**`build_targets.py`**:
```python
# Sector-neutralized forward return
for horizon in [1, 5, 21, 63]:
    stock_fwd_ret = (adj_close.shift(-horizon) / adj_close) - 1
    sector_etf_fwd_ret = ...  # same for sector ETF
    alpha = stock_fwd_ret - sector_etf_fwd_ret
    # XLC pre-2018: compute from constituent-weighted return of Comm stocks
```

### Model implementation:

**Decision: Write PatchTST from scratch** (~400 lines). Libraries (neuralforecast, tsai) don't support our custom cross-mixer + static encoder + quantile heads cleanly. Custom implementation gives full control.

**`alpha_model/model/patch_tst.py`**:
```python
class PatchTST(nn.Module):
    def __init__(self, n_channels=23, context_len=250, patch_len=5,
                 d_model=128, n_heads=8, n_layers=3, d_ff=256, dropout=0.2):
        # Patching: unfold context_len into patches
        self.n_patches = context_len // patch_len  # 50
        self.patch_proj = nn.Linear(patch_len, d_model)  # 5→128
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True),
            num_layers=n_layers
        )
        # Output: mean-pool over patches → 128-dim per channel

    def forward(self, x):  # x: (batch, n_channels, context_len)
        B, C, L = x.shape
        # Patch each channel independently
        x = x.unfold(2, self.patch_len, self.patch_len)  # (B, C, n_patches, patch_len)
        x = x.reshape(B * C, self.n_patches, self.patch_len)
        x = self.patch_proj(x) + self.pos_embed  # (B*C, 50, 128)
        x = self.encoder(x)  # (B*C, 50, 128)
        x = x.mean(dim=1)  # (B*C, 128) — mean pool
        x = x.reshape(B, C, -1)  # (B, 21, 128)
        return x  # per-channel representations
```

**`alpha_model/model/static_encoder.py`**:
```python
class StaticEncoder(nn.Module):
    def __init__(self, n_sector=11, n_cap=5, d_out=64):
        self.linear = nn.Linear(n_sector + n_cap, d_out)  # 16→64
    def forward(self, sector_onehot, cap_onehot):
        return F.gelu(self.linear(torch.cat([sector_onehot, cap_onehot], dim=-1)))
```

**`alpha_model/model/cross_mixer.py`**:
```python
class CrossChannelMixer(nn.Module):
    def __init__(self, n_channels=21, d_channel=128, d_static=64, d_hidden=512, d_out=256, dropout=0.3):
        d_in = n_channels * d_channel + d_static  # 21*128+64 = 2752
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out), nn.GELU(), nn.Dropout(dropout),
        )
    def forward(self, channel_reps, static_rep):
        # channel_reps: (B, 21, 128) → flatten → (B, 2688)
        x = torch.cat([channel_reps.flatten(1), static_rep], dim=-1)  # (B, 2752)
        return self.mlp(x)  # (B, 256)
```

**`alpha_model/model/prediction_head.py`**:
```python
class QuantileHead(nn.Module):
    def __init__(self, d_in=256, n_horizons=4, n_quantiles=3):
        # Separate head per horizon (prevents interference)
        self.heads = nn.ModuleList([nn.Linear(d_in, n_quantiles) for _ in range(n_horizons)])
    def forward(self, x):  # x: (B, 256)
        return torch.stack([head(x) for head in self.heads], dim=1)  # (B, 4, 3)
        # dim 1 = horizons [1d,5d,21d,63d], dim 2 = quantiles [q10,q50,q90]
```

### Training:

**Walk-forward folds**:
```
Fold 1: train 2010-2014, val 2015, test 2016
Fold 2: train 2011-2015, val 2016, test 2017
...
Fold 8: train 2017-2021, val 2022, test 2023
Fold 9: train 2018-2022, val 2023, test 2024
→ 9 folds, 9 models
```

**At inference: use the MOST RECENT fold's model** (Fold 9 as of 2024). Store model version + fold in AlphaPrediction metadata.

**Compute estimate**: ~21 features × 50 patches × batch 256 × 3 layers. Roughly ~2M params. Training on **Google Colab T4 GPU**: ~30 min/fold, ~4.5 hours total for 9 folds. Download final `patch_tst_fold9.pt` to M1 Mac for CPU inference.

**Feature attribution method**: Input × gradient (cheapest, good enough for feature ranking). Compute via `torch.autograd.grad`. Store top 5 features in `FeatureContribution`.

---

## Phase 3: Backtesting with Transaction Costs

### Portfolio construction (NOW SPECIFIED):
```
Strategy: Long-short decile portfolio
- Universe: S&P 500 constituents (at that date)
- Liquidity filter: exclude stocks with ADV < $5M
- Sort stocks by alpha_21d prediction (50th percentile)
- Long: top decile (~50 stocks), equal weight
- Short: bottom decile (~50 stocks), equal weight
- Rebalance: WEEKLY (Monday close) — informed by alpha decay half-life
  (If half-life < 5d: daily rebalance. If 5-15d: weekly. If >15d: bi-weekly)
- Execution: assume all trades at Monday close price
- Position sizing: equal weight within each leg (NOT Kelly — Kelly used only for single-stock analysis)
```

### Transaction cost specifics:
```python
# Spread estimation — fixed assumptions per market cap tier:
SPREAD_BPS = {"mega": 1, "large": 3, "mid": 8, "small": 20}
spread_cost = 0.5 * SPREAD_BPS[cap_tier] / 10000  # half-spread

# Market impact (Almgren-Chriss square root model):
# trade_size and ADV both in DOLLARS
K_IMPACT = {"mega": 0.05, "large": 0.15, "mid": 0.4, "small": 0.8}
impact_cost = K_IMPACT[cap_tier] * sqrt(abs(trade_dollars) / adv_dollars)

# Commission:
commission_per_dollar = 0.0001  # ~1bp, institutional rate

total_cost = spread_cost + impact_cost + commission_per_dollar
```

### Sharpe computation:
```
daily_strategy_return = 0.5 * (long_return - short_return)  # dollar-neutral
excess_return = daily_strategy_return - risk_free_rate / 252
sharpe = mean(excess_return) / std(excess_return) * sqrt(252)
Report: Gross Sharpe and Net Sharpe (after all costs)
```

---

## Phase 4: Observability + Error Recovery

### OTel span structure:
```
analysis_session (root span)
├── predict_alpha (PatchTST inference)
│   └── attrs: ticker, latency_ms, model_version
├── round_1 (parallel agents)
│   ├── agent:{name} (per agent)
│   │   ├── tool_call:{tool_name} (per tool invocation)
│   │   │   └── attrs: args, result_summary, latency_ms
│   │   ├── llm_call (per LLM generation)
│   │   │   └── attrs: prompt_tokens, completion_tokens, latency_ms, adapter
│   │   └── output_validation
│   │       └── attrs: schema_valid, hallucination_check, retry_count
├── extract_claims
├── round_2 (debate)
│   └── agent:{name}_debate
└── round_3 (synthesis)
```

### Circuit breaker config:
```python
# NVIDIA Nemotron API: cloud, ~8s latency
nemotron_cb = CircuitBreaker(failure_threshold=3, recovery_timeout=30, expected_exception=(ConnectionError, TimeoutError))

# Yahoo Finance: free API, can rate-limit
yfinance_cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60, expected_exception=Exception)

# Brave Search: rate-limited
brave_cb = CircuitBreaker(failure_threshold=10, recovery_timeout=120, expected_exception=HTTPStatusError)

# Bright Data (Reddit): async scraping, can be slow
brightdata_cb = CircuitBreaker(failure_threshold=3, recovery_timeout=120, expected_exception=HTTPStatusError)
```

### Redis session lifecycle:
- TTL = 1 hour from creation
- On GET /analysis/{id}: if missing → `{"status": "expired", "message": "Session expired"}`
- On WS reconnect: client sends last_event_index → server replays events from Redis list

### Hallucination guardrail implementation:
- Every tool call result stored in `CommitteeState.trace: list[TraceEvent]`
- TraceEvent = `{agent, tool, args, result, timestamp}`
- Guardrail function: iterate Evidence items → find matching TraceEvent by `source_tool` → compare `value` → flag mismatches
- On mismatch: re-prompt agent with "Your evidence for {metric} shows {claimed_value} but the tool returned {actual_value}. Correct your analysis."

---

## Phase 5: Frontend

### Component library: shadcn/ui on Radix primitives
- Provides: Card, Dialog, Tabs, Select, Badge, Tooltip, Skeleton (loading)
- Consistent Bloomberg-dark theme via Tailwind config

### Routing:
```typescript
<Routes>
  <Route path="/" element={<DashboardPage />} />
  <Route path="/analysis/:id" element={<DashboardPage />} />
  <Route path="/memo/:id" element={<MemoPage />} />
  <Route path="/backtest/:id" element={<BacktestPage />} />
</Routes>
```

### AlphaCurve → rename to **AlphaForecast**:
- Discrete bar chart (4 bars: 1D, 1W, 1M, 1Q)
- Each bar: height = q50 (point estimate), error bars = q10 to q90
- Green bars for positive alpha, red for negative
- Library: lightweight-charts histogram series OR recharts BarChart

### WebSocket strategy:
```typescript
// Use native WebSocket with manual reconnect
class AnalysisWebSocket {
  private ws: WebSocket | null = null
  private lastEventIndex = 0

  connect(analysisId: string) {
    this.ws = new WebSocket(`ws://localhost:8000/api/v1/ws/analysis/${analysisId}`)
    this.ws.onmessage = (e) => { this.lastEventIndex++; this.handleEvent(JSON.parse(e.data)) }
    this.ws.onclose = () => setTimeout(() => this.reconnect(analysisId), 2000)
  }

  reconnect(analysisId: string) {
    // Fetch current state via REST, then resume stream
    fetch(`/api/v1/analysis/${analysisId}`).then(...)
    this.connect(analysisId)
  }
}
```

### Frontend tests (Vitest):
- `TickerSearch.test.tsx`: renders, accepts input, calls API
- `AgentPanel.test.tsx`: renders streaming text, shows tool badges
- `AlphaForecast.test.tsx`: renders 4 bars with correct values from test fixture data

---

## Phase 6: Real Computations

### HMM regime detection (`analysis/macro.py`):
```python
from hmmlearn.hmm import GaussianHMM

# Features: [VIX_level, HYG_LQD_spread, yield_2s10s]
# Covariance: full (3 features, thousands of samples → stable estimate)
# n_iter: 100 EM iterations
# Initialization: k-means on features
hmm = GaussianHMM(n_components=3, covariance_type="full", n_iter=100, init_params="mc", random_state=42)
hmm.means_init = kmeans(features, 3).cluster_centers_

# After fitting, label states by sorting emission means of VIX:
# Highest VIX mean → "risk_off", lowest → "risk_on", middle → "transition"
state_order = np.argsort(hmm.means_[:, 0])  # sort by VIX mean
LABELS = {state_order[0]: "risk_on", state_order[1]: "transition", state_order[2]: "risk_off"}

# Retrain: monthly with expanding window from 2005
```

### VaR/CVaR (`analysis/risk.py`):
```python
# Historical simulation: 500-day lookback
returns_500d = price_returns[-500:]

# VaR (95%, 99%):
var_95 = np.percentile(returns_500d, 5)   # left tail
var_99 = np.percentile(returns_500d, 1)

# CVaR (Expected Shortfall):
cvar_95 = returns_500d[returns_500d <= var_95].mean()
cvar_99 = returns_500d[returns_500d <= var_99].mean()

# No parametric distribution assumed — purely historical empirical
```

### Stress testing (`analysis/risk.py`):
```python
SCENARIOS = {
    "2008_GFC": {"start": "2008-09-15", "end": "2009-03-09", "description": "Lehman to market bottom"},
    "2020_COVID": {"start": "2020-02-19", "end": "2020-03-23", "description": "Peak to trough"},
    "2022_RATE": {"start": "2022-01-03", "end": "2022-10-12", "description": "Rate hike selloff"},
}
# Method: apply the DAILY RETURN PATH of SPY during scenario to current position
# → path-dependent, captures drawdown duration
# Output: cumulative P&L over scenario, max drawdown, recovery days (if applicable)
```

### Volatility-scaled position sizing (`analysis/risk.py`):
```python
def position_size(alpha_q50: float, realized_vol_21d: float,
                  conviction: float, target_vol: float = 0.15) -> float:
    """Vol-scaled sizing. More robust than Kelly with only 3 quantile points.

    - Scales inversely with realized vol (risk parity intuition)
    - Scales with alpha magnitude and agent conviction
    - Capped at 10% of portfolio
    """
    raw_size = (target_vol / max(realized_vol_21d, 0.05)) * abs(alpha_q50) * conviction
    return max(0.0, min(raw_size, 0.10))  # cap at 10%
```

### DROP DCF — use relative valuation only:
- EV/EBITDA vs sector median (percentile rank)
- P/E vs own 5-year history (percentile rank)
- Price/Book vs sector
- No DCF — it's trivially attackable and adds no value for a quant system

---

## Phase 7: Debate Protocol

### `extract_claims` is NOT an LLM call:
```python
def extract_claims_node(state: CommitteeState) -> dict:
    """Pure function — just serialize key_claims from each AgentView."""
    claims = {}
    for agent_name in ["quant", "fundamentals", "sentiment", "risk", "macro"]:
        view = state[f"{agent_name}_view"]
        if view:
            claims[agent_name] = [c.model_dump() for c in view.key_claims[:5]]  # max 5
    return {"extracted_claims": claims}
```

### Round 2: Use LangGraph `Send()` for parallel debate with different inputs:
```python
def debate_router(state: CommitteeState) -> list[Send]:
    """Fan out: each agent gets others' claims (not their own)."""
    all_claims = state["extracted_claims"]
    sends = []
    for agent_name in AGENT_NAMES:
        others_claims = {k: v for k, v in all_claims.items() if k != agent_name}
        sends.append(Send(f"debate_{agent_name}", {
            "alpha_summary": state["alpha_prediction"].summary(),
            "own_view": state[f"{agent_name}_view"],
            "others_claims": others_claims,
        }))
    return sends
```

### MemoGenerator token budget (Nemotron 120B has 32K context):
```
System prompt: 800 tokens
Alpha summary: 150 tokens
5 agent FULL views: 5 × 500 = 2500 tokens (with Nemotron's 32K context, we can use full views)
5 debate responses: 5 × 300 = 1500 tokens
Generation budget: 4000 tokens
TOTAL: ~9000 tokens — well within 32K context ✓

With 32K context (vs Llama 3 8B's 8K), we can pass full agent views to memo synthesis
instead of truncated summaries. This produces higher quality memos.
```

### Conviction weighting — use EQUAL weights for agents, alpha model IC for signal:
```python
# Agent weights: equal (1.0 each) — no historical agent accuracy exists
agent_conviction = mean([view.conviction for view in all_views])

# Alpha model weight: based on out-of-sample IC from most recent backtest fold
alpha_weight = min(1.0, max(0.0, backtest_ic * 10))  # IC=0.02 → weight=0.2

# Final confidence = blend
confidence = 0.6 * alpha_weight + 0.4 * agent_conviction
```

---

## Phase 8: Sentiment Agent — Micro Details

### Search queries (3 per ticker):
```python
queries = [
    f"{ticker} stock news today",
    f"{ticker} earnings analyst rating",
    f"{ticker} {company_name} market outlook",
]
```

### Article extraction: `trafilatura`
```python
import trafilatura
content = trafilatura.extract(html, include_comments=False, include_tables=False)
# Truncate to ~500 tokens (~375 words)
content = content[:2000]  # rough char limit
```

### Reddit via Bright Data API:
```python
# Trigger async scrape, then poll for results
brightdata = get_brightdata_client()

# Search by keyword across Reddit
snapshot_id = await brightdata.search_reddit(
    keyword=f"{ticker} stock",
    num_posts=10,
    sort_by="Hot",
    date_range="Past month"
)

# Also search specific subreddits
snapshot_id2 = await brightdata.search_subreddit(
    subreddit_url="https://www.reddit.com/r/wallstreetbets/",
    keyword=ticker,
    sort_by="New",
    num_posts=10
)

# Poll for results (Bright Data is async — may take 10-30s)
import asyncio
for _ in range(10):  # max 10 polls × 5s = 50s timeout
    posts = await brightdata.get_snapshot(snapshot_id)
    if posts:
        break
    await asyncio.sleep(5)
# Use title + body text, truncate per token budget
```

### Temporal decay formula:
```python
def decay_weight(article_date: date, now: date, half_life_days: float = 3.0) -> float:
    age = (now - article_date).days
    return math.exp(-math.log(2) * age / half_life_days)
# 0 days → 1.0, 3 days → 0.5, 7 days → 0.2, 14 days → 0.04
```

### Token budget for sentiment prompt (Nemotron 32K context):
```
System prompt: 500 tokens
Top 10 articles: 10 × 500 = 5000 tokens (truncated by trafilatura)
Top 10 Reddit posts: 10 × 200 = 2000 tokens (title + body excerpt)
Generation: 3000 tokens
TOTAL: ~10500 tokens — well within 32K ✓

With 32K context we can include more sources than the 8K Llama 3 budget allowed.
```

---

## Phase 9: Nemotron API Configuration

Nemotron is set up in Phase 1 via `NemotronClient`. This phase covers the agent prompting strategy and latency optimization.

### Agent prompting — thinking mode OFF for structured output:
```python
# IMPORTANT: Nemotron has a reasoning mode that returns content in reasoning_content field
# For agent structured JSON output, DISABLE thinking to get content directly:
resp = await client.chat.completions.create(
    model="nvidia/nemotron-3-super-120b-a12b",
    messages=messages,
    temperature=0.7,
    max_tokens=3000,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    stream=False,  # non-streaming for JSON parsing reliability
)
content = resp.choices[0].message.content  # valid JSON string
# Post-validate with Pydantic, retry with correction prompt on failure (max 2x)
```

### Streaming for UI (agent reasoning visible to user):
```python
# For the frontend streaming panels, USE streaming:
stream = await client.chat.completions.create(
    model="nvidia/nemotron-3-super-120b-a12b",
    messages=messages,
    max_tokens=3000,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    stream=True,
)
async for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        yield chunk.choices[0].delta.content  # stream to WebSocket
```

### Realistic latency budget (measured from API tests):
```
PatchTST (CPU/M1):     0.5s
Round 1 (parallel):
  Quant/Fund/Risk/Macro: ~2 tools × 0.5s + ~8s Nemotron = ~9s each (parallel)
  Sentiment: 3 Brave searches × 1s + Bright Data Reddit ~30s + ~8s Nemotron = ~41s (bottleneck)
  → Round 1 total: ~41s (limited by Bright Data Reddit polling)
Round 2 (parallel):    ~8s Nemotron per agent = ~8s
Round 3:               ~8s Nemotron
────────────────────
Total:                ~58s optimistic, ~80s with Reddit polling + retries
Target: <90s
```

### Token budget — Nemotron 120B has 32K context (much more than Llama 3 8B's 8K):
With 32K context, token budgets from Phase 7/8 have generous headroom. No truncation concerns for memo synthesis.

---

## Phase 10: Yahoo Finance Agent Tools

Yahoo Finance is integrated in Phase 1 (`YahooFinanceService`). This phase adds the specific agent tools that wrap yfinance data.

### Fundamentals Agent tools:
```python
@tool
def get_financial_statements(ticker: str) -> dict:
    """Fetch income statement, balance sheet, cash flow via Yahoo Finance."""
    yf_svc = get_yahoo_finance_service()
    return yf_svc.get_financial_statements(ticker)

@tool
def get_analyst_estimates(ticker: str) -> dict:
    """Fetch consensus analyst estimates via Yahoo Finance."""
    yf_svc = get_yahoo_finance_service()
    return yf_svc.get_analyst_estimates(ticker)

@tool
def get_relative_valuation(ticker: str) -> dict:
    """EV/EBITDA, P/E, P/B vs sector median via Yahoo Finance."""
    yf_svc = get_yahoo_finance_service()
    data = yf_svc.get_fundamentals(ticker)
    sector = data["sector"]
    # Get sector constituents from S&P 500 list
    sector_tickers = get_sector_constituents(sector)
    sector_data = {t: yf_svc.get_fundamentals(t) for t in sector_tickers[:20]}  # top 20 by mcap
    return compute_percentile_ranks(data, sector_data)
```

### Risk Agent tools:
```python
@tool
def get_risk_data(ticker: str) -> dict:
    """Fetch short interest, options implied vol, put/call ratio via Yahoo Finance."""
    yf_svc = get_yahoo_finance_service()
    short = yf_svc.get_short_interest(ticker)
    options = yf_svc.get_options_data(ticker)
    return {**short, **options}
```

### Point-in-time limitation with Yahoo Finance:
Yahoo Finance provides CURRENT fundamentals only — it does not support historical as-of-date queries like Bloomberg's `FILING_STATUS=MR`. For the PatchTST training pipeline (Phase 2), this is not an issue because PatchTST uses price-only features. For agent analysis at inference time, Yahoo Finance returns the most recent available data, which is what a PM would see today. Historical backtesting of agent views is not supported — only the quant backtest (Phase 3) uses historical data.

---

## Phase 11: Docker + CI/CD

### docker-compose.yml:
```yaml
services:
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes: [redis_data:/data]
    ports: ["6379:6379"]

  backend:
    build: ./backend
    depends_on: [redis]
    env_file: .env
    ports: ["8000:8000"]

  frontend:
    build: ./frontend
    ports: ["3000:3000"]

volumes:
  redis_data:
```

No GPU container needed — LLM is NVIDIA Nemotron cloud API, PatchTST inference runs on CPU.

### CI (.github/workflows/ci.yml):
```yaml
jobs:
  backend:
    steps:
      - uses: actions/setup-python@v5
        with: { python-version: "3.13" }
      - run: pip install -e ".[test]"
      - run: ruff check backend/
      - run: mypy backend/app/ --strict
      - run: pytest backend/tests/ -v --cov=app --cov-report=xml
    env:
      NVIDIA_API_KEY: ${{ secrets.NVIDIA_API_KEY }}
      BRAVE_API_KEY: ${{ secrets.BRAVE_API_KEY }}
      FRED_API_KEY: ${{ secrets.FRED_API_KEY }}

  frontend:
    steps:
      - uses: actions/setup-node@v4
        with: { node-version: "20" }
      - run: cd frontend && npm ci && npm run lint && npx vitest run
```

---

## Verification Checklist (with exact commands)
```bash
# 1. API connectivity (already verified in test_apis.py)
python test_apis.py
# → All 8 APIs: PASS

# 2. Alpha model (trained on Colab, evaluate locally)
cd alpha_model && python evaluate.py --model ../models/patch_tst_fold9.pt
# → Prints: IC_1d, IC_5d, IC_21d, IC_63d
# → Prints: ICIR_21d, HitRate_21d, FactorR2
# → W&B dashboard link

# 3. Backtest
python -m app.backtest.engine --start 2020-01-01 --end 2024-12-31
# → Prints: Gross Sharpe, Net Sharpe, MaxDD

# 4. Backend tests
cd backend && pytest tests/ -v --cov
# → All pass, coverage >80%

# 5. Frontend tests
cd frontend && npx vitest run
# → All pass

# 6. Integration (local — Redis + Nemotron API)
redis-server --appendonly yes --daemonize yes
cd backend && uvicorn app.main:app --reload &
curl -X POST localhost:8000/api/v1/analysis -d '{"ticker":"AAPL"}' | jq .analysis_id
# → Returns UUID
curl localhost:8000/api/v1/analysis/{id} | jq .status
# → "complete"
curl localhost:8000/api/v1/analysis/{id}/memo | jq .recommendation
# → "BUY" (or similar)

# 7. End-to-end in browser
cd frontend && npm run dev &
open http://localhost:5173
# → Type AAPL → see AlphaForecast bars → 5 agent panels streaming → debate → memo
```

---

## Engineering Review — Fixes Incorporated Into Plan

The following fixes address 25 issues found during distinguished ML/AI engineering review. Each fix is tagged with its severity and the phase it modifies.

### CRITICAL fixes (applied to phases above)

**C1. API keys out of source** (Phase 1)
- `test_apis.py` must load from `.env` via `python-dotenv`, not hardcoded strings
- `.gitignore` includes `.env` before `git init`
- Keys in Prerequisites section above are for reference only — never committed

**C2. Sector-neutralized features** (Phase 2)
- Add channel 22: `sector_neutral_return = stock_return - sector_etf_return` (daily)
- Add channel 23: `sector_neutral_return_5d` (rolling 5d cumulative)
- TOTAL channels: 23 (was 21). Update PatchTST `n_channels=23`
- This prevents model from learning sector beta instead of alpha

**C3. Forward-fill policy tightened** (Phase 2)
- Forward-fill max **2 days** (was 5). After 2d, mark as NaN
- Rolling OLS requires **min 50/63 valid observations** (`min_periods=50`)
- Add validation: `assert features.isna().sum().max() < 0.05 * len(features)` per ticker

**C4. Replace Kelly with vol-scaled sizing** (Phase 6)
```python
def position_size(alpha_q50: float, realized_vol_21d: float,
                  conviction: float, target_vol: float = 0.15) -> float:
    """Volatility-scaled position sizing. Simpler and more robust than Kelly."""
    raw_size = (target_vol / max(realized_vol_21d, 0.05)) * abs(alpha_q50) * conviction
    return max(0, min(raw_size, 0.10))  # cap at 10%
```

**C5. NemotronClient immutable messages** (Phase 1/9)
```python
async def chat_completion_json(self, messages: list[dict], max_tokens: int = 3000) -> dict:
    messages = [m.copy() for m in messages]  # never mutate caller's list
    for attempt in range(3):
        ...
```

**C6. Pandera schema validation at pipeline boundaries** (Phase 2)
```python
import pandera as pa

prices_schema = pa.DataFrameSchema({
    "ticker": pa.Column(str), "date": pa.Column("datetime64[ns]"),
    "open": pa.Column(float, pa.Check.gt(0)),
    "close": pa.Column(float, pa.Check.gt(0)),
    "volume": pa.Column(float, pa.Check.ge(0)),
})
# Validate after each download/transform step
prices_schema.validate(prices_df)
```
Add `pandera>=0.20.0` to tech stack.

### HIGH fixes

**H1. Delisted ticker handling** (Phase 2)
```python
hist = yf.download(ticker, start=start, end=end, progress=False)
if hist.empty or len(hist) < 100:  # less than ~5 months of data
    log.warning(f"Skipping {ticker}: insufficient data ({len(hist)} rows)")
    continue  # exclude from universe for this period
```

**H2. Quantile calibration check** (Phase 2/3)
```python
# In evaluation.py, after generating predictions:
for q, target_pct in [(10, 0.10), (90, 0.90)]:
    actual_below = (actuals < predictions[f"q{q}"]).mean()
    print(f"  q{q}: expected {target_pct:.0%} below, actual {actual_below:.1%}")
    assert abs(actual_below - target_pct) < 0.05, f"q{q} miscalibrated: {actual_below:.1%}"
```

**H3. Colab checkpoint-per-fold** (Phase 2)
```python
for fold in range(1, 10):
    ckpt_path = f"checkpoints/fold_{fold}.pt"
    if os.path.exists(ckpt_path):
        print(f"Fold {fold} already trained, skipping")
        continue
    model = train_fold(fold, ...)
    torch.save(model.state_dict(), ckpt_path)
    # Also upload to W&B Artifacts for transfer to M1
    wandb.log_artifact(wandb.Artifact(f"model-fold{fold}", type="model"))
```

**H4. Reddit pre-fetch for latency** (Phase 8)
- Add `backend/app/services/search/reddit_prefetch.py`:
  - On analysis start, immediately trigger Bright Data scrape (fire-and-forget)
  - Sentiment agent polls for result when it runs (~30s later in pipeline)
  - If result ready: use it. If not: poll up to 20s more, then proceed with Brave-only data
  - This overlaps Reddit latency with Round 1 agent LLM calls

**H5. Prediction-execution timing** (Phase 3)
```python
# Backtest: predict using FRIDAY CLOSE data, execute at MONDAY CLOSE
# This matches real-world workflow: run analysis Friday evening, trade Monday
for date in trading_dates:
    if date.weekday() != 0:  # Monday
        continue
    friday = date - pd.offsets.BDay(1)  # previous business day
    features = get_features_as_of(friday)
    predictions = model.predict(features)
    execute_trades(date, predictions)  # Monday close prices
```

**H6. Redis TTL extended** (Phase 1)
- `redis_session_ttl_seconds: int = 86400` (24 hours, was 3600)

**H7. Reproducibility seeds** (Phase 2)
```python
import torch, numpy as np, random
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)
```

### MEDIUM fixes

**M1. Attention rollout for interpretability** (Phase 2)
- Replace input × gradient with attention rollout (average attention weights across 3 layers)
- Free to compute: just hook `nn.TransformerEncoderLayer` attention weights
- Returns per-patch importance → maps to temporal windows

**M2. XLC pre-2018 proxy** (Phase 2)
```python
XLC_PROXY_TICKERS = ["GOOGL", "META", "NFLX", "DIS", "T", "VZ", "CMCSA"]
# Equal-weighted daily return before 2018-06-18 (XLC launch date)
```

**M3. yfinance caching** (Phase 10)
```python
from cachetools import TTLCache
_fundamentals_cache = TTLCache(maxsize=500, ttl=3600)

def get_fundamentals(self, ticker: str) -> dict:
    if ticker in _fundamentals_cache:
        return _fundamentals_cache[ticker]
    result = self._fetch_fundamentals(ticker)
    _fundamentals_cache[ticker] = result
    return result
```
Add `cachetools>=5.3.0` to tech stack.

**M4. Agent tool graceful degradation** (Phase 8/10)
```python
@tool
def search_brave(query: str) -> dict:
    try:
        return brave_client.search(query)
    except Exception as e:
        return {"error": str(e), "results": [], "source": "brave", "available": False}
# Agent LLM sees the error and reasons with partial data
```

**M5. Python version flexibility** (Phase 1)
- Change `python = ">=3.12"` (was `>=3.13`). Test with 3.13 first, fall back to 3.12 if `hmmlearn` or `waybackpy` lack wheels.

**M6. M1 memory management** (Phase 2)
- Process universe in chunks of 50 tickers during feature engineering
- PatchTST batch size: 64 for inference (single ticker, memory-safe)
- `del` intermediate DataFrames and call `gc.collect()` after each chunk

### LOW fixes (defer to post-MVP)

- W&B Artifacts for model versioning (H3 partially covers this)
- WebSocket exponential backoff with jitter
- Docker deferred to post-MVP (local dev sufficient)
- CI uses `@pytest.mark.integration` to separate unit vs integration tests
- GarmanKlass overnight return correction (minor impact)
- HMM automated monthly retraining via cron
