import { TickerSearch } from '../components/TickerSearch';
import { StatusBar } from '../components/StatusBar';
import { AlphaForecast } from '../components/AlphaForecast';
import { AgentPanel } from '../components/AgentPanel';
import { MemoView } from '../components/MemoView';
import { useAnalysisStore } from '../store/analysisStore';
import type { AgentName } from '../types';

const AGENT_NAMES: AgentName[] = ['quant', 'fundamentals', 'sentiment', 'risk', 'macro'];

export function DashboardPage() {
  const { alphaPrediction, agents, memo, status } = useAnalysisStore();

  return (
    <div className="min-h-screen bg-[var(--bg-primary)]">
      {/* Header */}
      <header className="border-b border-[var(--border)] bg-[var(--bg-secondary)]">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-[var(--accent-blue)] rounded flex items-center justify-center text-white font-bold text-sm">
              αE
            </div>
            <div>
              <h1 className="text-sm font-bold text-[var(--text-primary)] tracking-wider">Alpha Edge</h1>
              <p className="text-[10px] text-[var(--text-muted)]">AI Investment Committee</p>
            </div>
          </div>
          <TickerSearch />
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-6 py-6 space-y-6">
        <StatusBar />

        {/* Alpha Prediction */}
        {alphaPrediction && <AlphaForecast prediction={alphaPrediction} />}

        {/* Agent Panels */}
        {status !== 'idle' && status !== 'predicting' && (
          <div>
            <h2 className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider mb-3">
              Agent Analysis
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
              {AGENT_NAMES.map(name => (
                <AgentPanel key={name} name={name} state={agents[name]} />
              ))}
            </div>
          </div>
        )}

        {/* Investment Memo */}
        {memo && <MemoView memo={memo} />}
      </main>
    </div>
  );
}
