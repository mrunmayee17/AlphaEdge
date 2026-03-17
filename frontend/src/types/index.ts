export type AgentName = 'quant' | 'fundamentals' | 'sentiment' | 'risk' | 'macro';
export type Direction = 'BULLISH' | 'BEARISH' | 'NEUTRAL';
export type Recommendation = 'STRONG_BUY' | 'BUY' | 'HOLD' | 'SELL' | 'STRONG_SELL';
export type AnalysisStatus = 'idle' | 'pending' | 'predicting' | 'round_1' | 'round_2' | 'round_3' | 'complete' | 'error';

export interface AlphaPrediction {
  ticker: string;
  prediction_date: string;
  sector: string;
  sector_etf: string;
  alpha_1d: number;
  alpha_5d: number;
  alpha_21d: number;
  alpha_63d: number;
  q10_1d: number; q90_1d: number;
  q10_5d: number; q90_5d: number;
  q10_21d: number; q90_21d: number;
  q10_63d: number; q90_63d: number;
  model_version: string;
  training_fold: string;
  inference_latency_ms: number;
}

export interface Claim {
  metric: string;
  value: string;
  source_tool: string;
  direction: Direction;
}

export interface Evidence {
  source_tool: string;
  data_point: string;
  relevance: string;
}

export interface AgentView {
  agent_name: AgentName;
  ticker: string;
  direction: Direction;
  conviction: number;
  time_horizon: '1D' | '1W' | '1M' | '1Q';
  agrees_with_alpha: boolean;
  key_claims: Claim[];
  supporting_evidence: Evidence[];
  risks: string[];
  summary: string;
}

export interface AgentDebateResponse {
  agent_name: AgentName;
  revised_conviction: number;
  revised_direction?: Direction;
}

export interface InvestmentMemo {
  ticker: string;
  date: string;
  recommendation: Recommendation;
  confidence: number;
  recommended_horizon: string;
  position_size_pct: number;
  quant_summary: string;
  fundamentals_summary: string;
  sentiment_summary: string;
  risk_summary: string;
  macro_summary: string;
  consensus_claims: string[];
  dissenting_opinions: { agent_name: string; disagreement: string }[];
  upcoming_events: string[];
  stress_test_worst_case: number;
  current_regime: string;
}

export interface AgentState {
  status: 'idle' | 'thinking' | 'done' | 'error';
  streamedText: string;
  view: AgentView | null;
  debate: AgentDebateResponse | null;
}

export interface AnalysisResponse {
  analysis_id: string;
  ticker: string;
  status: AnalysisStatus;
}
