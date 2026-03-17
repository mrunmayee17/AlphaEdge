"""All Pydantic schemas — contracts between prediction layer, agents, and API."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ── Alpha Prediction (from Chronos-2) ───────────────────────────────────────


class PatchAttention(BaseModel):
    """Attention weight for a temporal patch (5-day window)."""
    patch_index: int
    start_day: int
    end_day: int
    weight: float = Field(ge=0.0, le=1.0)


class FeatureContribution(BaseModel):
    """Top feature attribution from attention rollout."""
    feature_name: str
    channel_index: int
    importance: float = Field(ge=0.0)


class AlphaPrediction(BaseModel):
    ticker: str
    prediction_date: date
    sector: str
    sector_etf: str

    # Point estimates (50th percentile) — sector-neutralized forward returns
    alpha_1d: float
    alpha_5d: float
    alpha_21d: float
    alpha_63d: float

    # Quantile bands
    q10_1d: float
    q90_1d: float
    q10_5d: float
    q90_5d: float
    q10_21d: float
    q90_21d: float
    q10_63d: float
    q90_63d: float

    # Interpretability
    patch_attention: list[PatchAttention] = Field(default_factory=list)
    top_features: list[FeatureContribution] = Field(default_factory=list)

    # Metadata
    model_version: str
    training_fold: str
    inference_latency_ms: float

    def summary(self) -> str:
        """Short text summary for agent context."""
        direction = "BULLISH" if self.alpha_21d > 0 else "BEARISH"
        return (
            f"{self.ticker} alpha prediction ({self.prediction_date}): "
            f"{direction} — 1d={self.alpha_1d:+.3f}, 5d={self.alpha_5d:+.3f}, "
            f"21d={self.alpha_21d:+.3f} [{self.q10_21d:+.3f}, {self.q90_21d:+.3f}], "
            f"63d={self.alpha_63d:+.3f}. Sector: {self.sector} (vs {self.sector_etf})."
        )


class AlphaPredictionSummary(BaseModel):
    """Compact view proving an agent read the alpha signal."""
    ticker: str
    alpha_21d: float
    q10_21d: float
    q90_21d: float
    direction: Literal["BULLISH", "BEARISH", "NEUTRAL"]


# ── Agent Views (Round 1 output) ────────────────────────────────────────────


class Claim(BaseModel):
    """A factual claim made by an agent, with evidence source."""
    claim: str
    metric: Optional[str] = None
    value: Optional[str] = None
    source_tool: Optional[str] = None


class Evidence(BaseModel):
    """Supporting evidence for an agent's view."""
    source: str  # e.g. "yahoo_finance", "brave_search", "fred"
    source_tool: str
    metric: str
    value: str
    retrieved_at: datetime = Field(default_factory=datetime.utcnow)


class AgentView(BaseModel):
    agent_name: str
    ticker: str
    alpha_seen: AlphaPredictionSummary
    direction: Literal["BULLISH", "BEARISH", "NEUTRAL"]
    conviction: float = Field(ge=0.0, le=1.0)
    time_horizon: Literal["1D", "1W", "1M", "1Q"]
    agrees_with_alpha: bool
    key_claims: list[Claim] = Field(max_length=5)
    supporting_evidence: list[Evidence] = Field(max_length=8)
    risks: list[str] = Field(max_length=5)
    summary: str = Field(max_length=1000)


# ── Debate (Round 2 output) ─────────────────────────────────────────────────


class ClaimReference(BaseModel):
    """Reference to another agent's claim the current agent agrees with."""
    agent_name: str
    claim_index: int
    reason: str


class Disagreement(BaseModel):
    """Structured disagreement with another agent's claim."""
    agent_name: str
    claim_index: int
    counter_argument: str
    supporting_evidence: Optional[str] = None


class AgentDebateResponse(BaseModel):
    agent_name: str
    agreements: list[ClaimReference] = Field(default_factory=list)
    disagreements: list[Disagreement] = Field(default_factory=list)
    revised_conviction: float = Field(ge=0.0, le=1.0)
    revised_direction: Optional[Literal["BULLISH", "BEARISH", "NEUTRAL"]] = None


# ── Investment Memo (Round 3 output) ────────────────────────────────────────


class DissentingOpinion(BaseModel):
    agent_name: str
    direction: Literal["BULLISH", "BEARISH", "NEUTRAL"]
    reason: str


class InvestmentMemo(BaseModel):
    ticker: str
    date: date
    alpha_prediction: AlphaPrediction
    alpha_decay_halflife_days: Optional[float] = None
    factor_r2: float

    recommendation: Literal["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"]
    confidence: float = Field(ge=0.0, le=1.0)
    recommended_horizon: str
    position_size_pct: float = Field(ge=0.0, le=10.0)

    quant_summary: str
    fundamentals_summary: str
    sentiment_summary: str
    risk_summary: str
    macro_summary: str

    consensus_claims: list[str] = Field(default_factory=list)
    dissenting_opinions: list[DissentingOpinion] = Field(default_factory=list)
    upcoming_events: list[str] = Field(default_factory=list)
    stress_test_worst_case: float
    current_regime: str


# ── Orchestrator State ──────────────────────────────────────────────────────


class TraceEvent(BaseModel):
    """Single tool call trace for hallucination guardrail."""
    agent: str
    tool: str
    args: dict
    result: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ── API Request/Response ────────────────────────────────────────────────────


class AnalysisRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=10, pattern=r"^[A-Za-z0-9.\-=^]+$")


class AnalysisResponse(BaseModel):
    analysis_id: str
    ticker: str
    status: str


class AnalysisStatus(BaseModel):
    analysis_id: str
    ticker: str
    status: Literal[
        "pending", "predicting", "round_1", "round_2", "round_3", "complete", "error"
    ]
    alpha_prediction: Optional[AlphaPrediction] = None
    agent_views: dict[str, Optional[AgentView]] = Field(default_factory=dict)
    debate_responses: dict[str, Optional[AgentDebateResponse]] = Field(default_factory=dict)
    memo: Optional[InvestmentMemo] = None
    error: Optional[str] = None
