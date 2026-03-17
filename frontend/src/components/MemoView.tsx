import { TrendingUp, TrendingDown, AlertTriangle, Shield, Globe } from 'lucide-react';
import type { InvestmentMemo, Recommendation } from '../types';

interface Props {
  memo: InvestmentMemo;
}

const REC_CONFIG: Record<Recommendation, { label: string; color: string; bg: string }> = {
  STRONG_BUY: { label: 'STRONG BUY', color: '#00d26a', bg: 'rgba(0, 210, 106, 0.1)' },
  BUY: { label: 'BUY', color: '#00d26a', bg: 'rgba(0, 210, 106, 0.07)' },
  HOLD: { label: 'HOLD', color: '#f59e0b', bg: 'rgba(245, 158, 11, 0.07)' },
  SELL: { label: 'SELL', color: '#ff4757', bg: 'rgba(255, 71, 87, 0.07)' },
  STRONG_SELL: { label: 'STRONG SELL', color: '#ff4757', bg: 'rgba(255, 71, 87, 0.1)' },
};

function Section({ title, icon, children }: { title: string; icon: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        {icon}
        <h4 className="text-xs font-semibold text-[var(--text-primary)] uppercase tracking-wider">{title}</h4>
      </div>
      <div className="text-xs text-[var(--text-secondary)] leading-relaxed">{children}</div>
    </div>
  );
}

export function MemoView({ memo }: Props) {
  const rec = REC_CONFIG[memo.recommendation] ?? { label: memo.recommendation, color: '#94a3b8', bg: 'rgba(148, 163, 184, 0.07)' };

  return (
    <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-lg animate-slide-in">
      {/* Header */}
      <div className="p-5 border-b border-[var(--border)]">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-[var(--text-primary)] tracking-wide">
            INVESTMENT MEMO — {memo.ticker}
          </h3>
          <span className="text-xs text-[var(--text-muted)]">{memo.date}</span>
        </div>

        {/* Recommendation badge */}
        <div className="flex items-center gap-4">
          <div className="px-4 py-2 rounded-md text-sm font-bold tracking-wider"
               style={{ color: rec.color, backgroundColor: rec.bg, border: `1px solid ${rec.color}30` }}>
            {rec.label}
          </div>
          <div className="flex items-center gap-4 text-xs">
            <div>
              <span className="text-[var(--text-muted)]">Confidence </span>
              <span className="font-bold text-[var(--text-primary)]">{(memo.confidence * 100).toFixed(0)}%</span>
            </div>
            <div>
              <span className="text-[var(--text-muted)]">Horizon </span>
              <span className="font-bold text-[var(--text-primary)]">{memo.recommended_horizon}</span>
            </div>
            <div>
              <span className="text-[var(--text-muted)]">Position </span>
              <span className="font-bold text-[var(--text-primary)]">{memo.position_size_pct.toFixed(1)}%</span>
            </div>
            <div>
              <span className="text-[var(--text-muted)]">Regime </span>
              <span className="font-bold text-[var(--text-primary)]">{memo.current_regime}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Body */}
      <div className="p-5 grid grid-cols-2 gap-6">
        <Section title="Quantitative" icon={<TrendingUp className="w-3.5 h-3.5 text-[var(--accent-blue)]" />}>
          {memo.quant_summary}
        </Section>
        <Section title="Fundamentals" icon={<TrendingUp className="w-3.5 h-3.5 text-[var(--accent-green)]" />}>
          {memo.fundamentals_summary}
        </Section>
        <Section title="Sentiment" icon={<TrendingDown className="w-3.5 h-3.5 text-[var(--accent-orange)]" />}>
          {memo.sentiment_summary}
        </Section>
        <Section title="Macro" icon={<Globe className="w-3.5 h-3.5 text-[var(--accent-purple)]" />}>
          {memo.macro_summary}
        </Section>
      </div>

      {/* Risk section */}
      <div className="px-5 pb-5 space-y-4">
        <Section title="Risk Assessment" icon={<Shield className="w-3.5 h-3.5 text-[var(--accent-red)]" />}>
          <p>{memo.risk_summary}</p>
          <p className="mt-2 text-[var(--accent-red)]">
            Stress test worst case: {(memo.stress_test_worst_case * 100).toFixed(1)}%
          </p>
        </Section>

        {/* Consensus */}
        {memo.consensus_claims.length > 0 && (
          <div className="space-y-2">
            <h4 className="text-xs font-semibold text-[var(--text-primary)] uppercase tracking-wider">Consensus</h4>
            <ul className="space-y-1">
              {memo.consensus_claims.map((claim, i) => (
                <li key={i} className="text-xs text-[var(--text-secondary)] flex items-start gap-2">
                  <span className="text-[var(--accent-green)]">✓</span> {claim}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Dissent */}
        {memo.dissenting_opinions.length > 0 && (
          <div className="space-y-2">
            <h4 className="text-xs font-semibold text-[var(--text-primary)] uppercase tracking-wider flex items-center gap-2">
              <AlertTriangle className="w-3 h-3 text-[var(--accent-orange)]" /> Dissent
            </h4>
            {memo.dissenting_opinions.map((d, i) => (
              <div key={i} className="text-xs text-[var(--text-secondary)]">
                <span className="text-[var(--accent-orange)] font-semibold">{d.agent_name}:</span> {d.disagreement}
              </div>
            ))}
          </div>
        )}

        {/* Upcoming events */}
        {memo.upcoming_events.length > 0 && (
          <div className="space-y-1">
            <h4 className="text-xs font-semibold text-[var(--text-primary)] uppercase tracking-wider">Upcoming Events</h4>
            {memo.upcoming_events.map((evt, i) => (
              <p key={i} className="text-xs text-[var(--text-muted)]">📅 {evt}</p>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
