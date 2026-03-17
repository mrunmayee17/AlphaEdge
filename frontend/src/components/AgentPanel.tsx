import { Loader2, CheckCircle, AlertCircle, TrendingUp, TrendingDown, Minus } from 'lucide-react';
import type { AgentName, AgentState } from '../types';

interface Props {
  name: AgentName;
  state: AgentState;
}

const AGENT_CONFIG: Record<AgentName, { label: string; color: string; icon: string }> = {
  quant: { label: 'Quantitative', color: 'var(--accent-blue)', icon: '📊' },
  fundamentals: { label: 'Fundamentals', color: 'var(--accent-green)', icon: '📈' },
  sentiment: { label: 'Sentiment', color: 'var(--accent-orange)', icon: '💬' },
  risk: { label: 'Risk Guardian', color: 'var(--accent-red)', icon: '🛡️' },
  macro: { label: 'Macro Regime', color: 'var(--accent-purple)', icon: '🌍' },
};

function DirectionBadge({ direction }: { direction: string }) {
  if (direction === 'BULLISH') return (
    <span className="flex items-center gap-1 text-[var(--accent-green)] text-xs font-bold">
      <TrendingUp className="w-3 h-3" /> BULLISH
    </span>
  );
  if (direction === 'BEARISH') return (
    <span className="flex items-center gap-1 text-[var(--accent-red)] text-xs font-bold">
      <TrendingDown className="w-3 h-3" /> BEARISH
    </span>
  );
  return (
    <span className="flex items-center gap-1 text-[var(--text-muted)] text-xs font-bold">
      <Minus className="w-3 h-3" /> NEUTRAL
    </span>
  );
}

export function AgentPanel({ name, state }: Props) {
  const config = AGENT_CONFIG[name];

  return (
    <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-lg overflow-hidden animate-slide-in">
      {/* Header */}
      <div className="px-4 py-3 border-b border-[var(--border)] flex items-center justify-between"
           style={{ borderTopColor: config.color, borderTopWidth: '2px' }}>
        <div className="flex items-center gap-2">
          <span>{config.icon}</span>
          <span className="text-sm font-semibold text-[var(--text-primary)]">{config.label}</span>
        </div>
        <div className="flex items-center gap-2">
          {state.status === 'thinking' && <Loader2 className="w-3.5 h-3.5 animate-spin text-[var(--accent-blue)]" />}
          {state.status === 'done' && <CheckCircle className="w-3.5 h-3.5 text-[var(--accent-green)]" />}
          {state.status === 'error' && <AlertCircle className="w-3.5 h-3.5 text-[var(--accent-red)]" />}
        </div>
      </div>

      {/* Content */}
      <div className="p-4">
        {state.status === 'idle' && (
          <p className="text-xs text-[var(--text-muted)]">Waiting...</p>
        )}

        {state.status === 'thinking' && (
          <div className="space-y-2">
            <div className="text-xs text-[var(--text-secondary)] whitespace-pre-wrap font-mono leading-relaxed max-h-48 overflow-y-auto">
              {state.streamedText || 'Analyzing...'}
            </div>
          </div>
        )}

        {state.view && (
          <div className="space-y-3">
            {/* Direction + Conviction */}
            <div className="flex items-center justify-between">
              <DirectionBadge direction={state.view.direction} />
              <div className="flex items-center gap-2">
                <span className="text-xs text-[var(--text-muted)]">Conviction</span>
                <div className="w-16 h-1.5 bg-[var(--bg-primary)] rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-700"
                    style={{
                      width: `${(state.view.conviction ?? 0.5) * 100}%`,
                      backgroundColor: config.color,
                    }}
                  />
                </div>
                <span className="text-xs font-bold" style={{ color: config.color }}>
                  {((state.view.conviction ?? 0.5) * 100).toFixed(0)}%
                </span>
              </div>
            </div>

            {/* Summary */}
            <p className="text-xs text-[var(--text-secondary)] leading-relaxed">
              {state.view.summary}
            </p>

            {/* Key claims */}
            {state.view.key_claims?.length > 0 && (
              <div className="space-y-1">
                <span className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider">Key Claims</span>
                {state.view.key_claims.slice(0, 3).map((claim, i) => (
                  <div key={i} className="flex items-start gap-2 text-xs">
                    <span className={claim.direction === 'BULLISH' ? 'text-[var(--accent-green)]' : claim.direction === 'BEARISH' ? 'text-[var(--accent-red)]' : 'text-[var(--text-muted)]'}>
                      {claim.direction === 'BULLISH' ? '▲' : claim.direction === 'BEARISH' ? '▼' : '—'}
                    </span>
                    <span className="text-[var(--text-secondary)]">{claim.metric}: {claim.value}</span>
                  </div>
                ))}
              </div>
            )}

            {/* Risks */}
            {state.view.risks?.length > 0 && (
              <div className="space-y-1">
                <span className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider">Risks</span>
                {state.view.risks.slice(0, 2).map((risk, i) => (
                  <p key={i} className="text-xs text-[var(--text-muted)]">⚠ {risk}</p>
                ))}
              </div>
            )}

            {/* Alpha agreement */}
            <div className="pt-2 border-t border-[var(--border)]">
              <span className={`text-[10px] ${state.view.agrees_with_alpha ? 'text-[var(--accent-green)]' : 'text-[var(--accent-orange)]'}`}>
                {state.view.agrees_with_alpha ? '✓ Agrees with alpha signal' : '✗ Disagrees with alpha signal'}
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
