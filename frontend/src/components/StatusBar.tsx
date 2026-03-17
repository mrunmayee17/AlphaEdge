import { useAnalysisStore } from '../store/analysisStore';
import type { AnalysisStatus } from '../types';

const STATUS_LABELS: Record<AnalysisStatus, { label: string; color: string }> = {
  idle: { label: 'READY', color: 'var(--text-muted)' },
  pending: { label: 'PENDING...', color: 'var(--accent-blue)' },
  predicting: { label: 'RUNNING PATCHTST...', color: 'var(--accent-blue)' },
  round_1: { label: 'ROUND 1 — AGENT ANALYSIS', color: 'var(--accent-blue)' },
  round_2: { label: 'ROUND 2 — DEBATE', color: 'var(--accent-orange)' },
  round_3: { label: 'ROUND 3 — SYNTHESIS', color: 'var(--accent-purple)' },
  complete: { label: 'COMPLETE', color: 'var(--accent-green)' },
  error: { label: 'ERROR', color: 'var(--accent-red)' },
};

const STEPS: AnalysisStatus[] = ['predicting', 'round_1', 'round_2', 'round_3', 'complete'];

export function StatusBar() {
  const { status, ticker, error } = useAnalysisStore();
  const config = STATUS_LABELS[status] ?? { label: status.toUpperCase(), color: 'var(--accent-blue)' };
  const activeStep = STEPS.indexOf(status);

  if ((status as string) === 'idle') return null;

  return (
    <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-lg px-5 py-3">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-3">
          <span className="text-xs font-bold tracking-wider" style={{ color: config.color }}>
            {config.label}
          </span>
          {ticker && <span className="text-xs text-[var(--text-muted)]">{ticker}</span>}
        </div>
        {status !== 'idle' && status !== 'error' && (
          <div className="flex items-center gap-1">
            {status !== 'complete' && (
              <div className="w-1.5 h-1.5 rounded-full animate-pulse-green" style={{ backgroundColor: config.color }} />
            )}
          </div>
        )}
      </div>

      {/* Progress bar */}
      <div className="flex gap-1">
        {STEPS.map((step, i) => (
          <div
            key={step}
            className="h-1 flex-1 rounded-full transition-all duration-500"
            style={{
              backgroundColor: i <= activeStep ? config.color : 'var(--border)',
              opacity: i === activeStep ? 0.7 : i < activeStep ? 1 : 0.3,
            }}
          />
        ))}
      </div>

      {error && <p className="mt-2 text-xs text-[var(--accent-red)]">{error}</p>}
    </div>
  );
}
