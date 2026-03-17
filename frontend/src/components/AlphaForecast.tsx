import type { AlphaPrediction } from '../types';

interface Props {
  prediction: AlphaPrediction;
}

const HORIZONS = [
  { key: '1d', label: '1D' },
  { key: '5d', label: '1W' },
  { key: '21d', label: '1M' },
  { key: '63d', label: '1Q' },
] as const;

function formatPct(v: number): string {
  return (v * 100).toFixed(2) + '%';
}

export function AlphaForecast({ prediction }: Props) {
  const maxAbsAlpha = Math.max(
    ...HORIZONS.map(h => Math.abs(prediction[`alpha_${h.key}` as keyof AlphaPrediction] as number)),
    0.001
  );

  return (
    <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-lg p-5">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-[var(--text-primary)] tracking-wide">
          ALPHA FORECAST
        </h3>
        <span className="text-xs text-[var(--text-muted)]">
          {prediction.model_version} · {prediction.inference_latency_ms.toFixed(0)}ms
        </span>
      </div>

      <div className="grid grid-cols-4 gap-3">
        {HORIZONS.map(({ key, label }) => {
          const alpha = prediction[`alpha_${key}` as keyof AlphaPrediction] as number;
          const q10 = prediction[`q10_${key}` as keyof AlphaPrediction] as number;
          const q90 = prediction[`q90_${key}` as keyof AlphaPrediction] as number;
          const isPositive = alpha >= 0;
          const barHeight = Math.max(10, (Math.abs(alpha) / maxAbsAlpha) * 80);

          return (
            <div key={key} className="flex flex-col items-center">
              <span className="text-xs text-[var(--text-muted)] mb-2">{label}</span>

              {/* Bar chart with error bars */}
              <div className="relative w-full h-24 flex items-end justify-center">
                {/* Error bar line */}
                <div className="absolute w-px bg-[var(--text-muted)]"
                  style={{
                    height: `${Math.max(10, ((Math.abs(q90 - q10)) / (maxAbsAlpha * 2)) * 80)}px`,
                    bottom: isPositive ? '50%' : undefined,
                    top: isPositive ? undefined : '50%',
                    left: '50%',
                    transform: 'translateX(-50%)',
                  }}
                />
                {/* Main bar */}
                <div
                  className={`w-8 rounded-sm transition-all duration-500 ${
                    isPositive ? 'bg-[var(--accent-green)]' : 'bg-[var(--accent-red)]'
                  }`}
                  style={{
                    height: `${barHeight}%`,
                    position: 'absolute',
                    bottom: isPositive ? '50%' : undefined,
                    top: !isPositive ? '50%' : undefined,
                    left: '50%',
                    transform: 'translateX(-50%)',
                    opacity: 0.8,
                  }}
                />
                {/* Center line */}
                <div className="absolute w-full h-px bg-[var(--border)] top-1/2" />
              </div>

              {/* Values */}
              <div className="mt-2 text-center">
                <div className={`text-sm font-bold ${isPositive ? 'text-[var(--accent-green)]' : 'text-[var(--accent-red)]'}`}>
                  {isPositive ? '+' : ''}{formatPct(alpha)}
                </div>
                <div className="text-[10px] text-[var(--text-muted)]">
                  [{formatPct(q10)}, {formatPct(q90)}]
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <div className="mt-4 pt-3 border-t border-[var(--border)] flex justify-between text-xs text-[var(--text-muted)]">
        <span>Sector: {prediction.sector} ({prediction.sector_etf})</span>
        <span>Fold: {prediction.training_fold}</span>
      </div>
    </div>
  );
}
