import { create } from 'zustand';
import type {
  AgentName, AgentState, AlphaPrediction, AnalysisStatus, InvestmentMemo,
} from '../types';
import { startAnalysis, getAnalysisStatus } from '../api/client';

const AGENT_NAMES: AgentName[] = ['quant', 'fundamentals', 'sentiment', 'risk', 'macro'];

function initialAgentState(): AgentState {
  return { status: 'idle', streamedText: '', view: null, debate: null };
}

interface AnalysisStore {
  ticker: string | null;
  analysisId: string | null;
  status: AnalysisStatus;
  error: string | null;
  alphaPrediction: AlphaPrediction | null;
  agents: Record<AgentName, AgentState>;
  memo: InvestmentMemo | null;

  setTicker: (ticker: string) => void;
  runAnalysis: (ticker: string) => Promise<void>;
  pollStatus: () => Promise<void>;
  reset: () => void;
  updateAgentStream: (agent: AgentName, text: string) => void;
  setAgentView: (agent: AgentName, view: AgentState['view']) => void;
}

export const useAnalysisStore = create<AnalysisStore>((set, get) => ({
  ticker: null,
  analysisId: null,
  status: 'idle',
  error: null,
  alphaPrediction: null,
  agents: Object.fromEntries(AGENT_NAMES.map(a => [a, initialAgentState()])) as Record<AgentName, AgentState>,
  memo: null,

  setTicker: (ticker) => set({ ticker }),

  runAnalysis: async (ticker) => {
    set({
      ticker,
      status: 'predicting',
      error: null,
      alphaPrediction: null,
      memo: null,
      agents: Object.fromEntries(AGENT_NAMES.map(a => [a, initialAgentState()])) as Record<AgentName, AgentState>,
    });

    try {
      const resp = await startAnalysis(ticker);
      set({ analysisId: resp.analysis_id, status: resp.status as AnalysisStatus });

      // Start polling
      const poll = setInterval(async () => {
        try {
          await get().pollStatus();
          const s = get().status;
          if (s === 'complete' || s === 'error') {
            clearInterval(poll);
          }
        } catch {
          clearInterval(poll);
        }
      }, 2000);
    } catch (err) {
      set({ status: 'error', error: err instanceof Error ? err.message : 'Unknown error' });
    }
  },

  pollStatus: async () => {
    const { analysisId } = get();
    if (!analysisId) return;

    const data = await getAnalysisStatus(analysisId);
    const update: Partial<AnalysisStore> = { status: data.status as AnalysisStatus };

    if (data.alpha_prediction) update.alphaPrediction = data.alpha_prediction;
    if (data.memo) update.memo = data.memo;
    if (data.error) update.error = data.error;

    // Update agent states from backend response
    // Backend stores views as {agent_name}_view and {agent_name}_debate
    const agents = { ...get().agents };
    for (const name of AGENT_NAMES) {
      const viewKey = `${name}_view`;
      const debateKey = `${name}_debate`;

      if (data[viewKey] && data[viewKey] !== null) {
        agents[name] = {
          ...agents[name],
          view: data[viewKey],
          status: 'done',
        };
      } else if (data.status === 'round_1' && !data[viewKey]) {
        agents[name] = {
          ...agents[name],
          status: 'thinking',
        };
      }

      if (data[debateKey] && data[debateKey] !== null) {
        agents[name] = {
          ...agents[name],
          debate: data[debateKey],
        };
      }
    }
    update.agents = agents;

    set(update as any);
  },

  reset: () => set({
    ticker: null,
    analysisId: null,
    status: 'idle',
    error: null,
    alphaPrediction: null,
    memo: null,
    agents: Object.fromEntries(AGENT_NAMES.map(a => [a, initialAgentState()])) as Record<AgentName, AgentState>,
  }),

  updateAgentStream: (agent, text) => set(state => ({
    agents: {
      ...state.agents,
      [agent]: { ...state.agents[agent], streamedText: state.agents[agent].streamedText + text, status: 'thinking' },
    },
  })),

  setAgentView: (agent, view) => set(state => ({
    agents: {
      ...state.agents,
      [agent]: { ...state.agents[agent], view, status: 'done' },
    },
  })),
}));
