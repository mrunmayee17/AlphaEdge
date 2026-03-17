import { useState } from 'react';
import { BarChart3, TrendingUp, TrendingDown, Activity } from 'lucide-react';
import axios from 'axios';

interface BacktestResult {
  gross_sharpe: number;
  net_sharpe: number;
  annual_return: number;
  annual_vol: number;
  max_drawdown: number;
  avg_daily_turnover: number;
  n_days: number;
}

export function BacktestPage() {
  const [startDate, setStartDate] = useState('2020-01-01');
  const [endDate, setEndDate] = useState('2024-12-31');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const runBacktest = async () => {
    setLoading(true);
    setError(null);
    try {
      const { data } = await axios.post('/api/v1/backtest', {
        start_date: startDate,
        end_date: endDate,
      });
      setResult(data);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[var(--bg-primary)]">
      <header className="border-b border-[var(--border)] bg-[var(--bg-secondary)]">
        <div className="max-w-5xl mx-auto px-6 py-4 flex items-center gap-3">
          <BarChart3 className="w-5 h-5 text-[var(--accent-blue)]" />
          <h1 className="text-sm font-bold text-[var(--text-primary)] tracking-wider">BACKTEST</h1>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-6 py-6 space-y-6">
        {/* Controls */}
        <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-lg p-5">
          <div className="flex items-end gap-4">
            <div>
              <label className="text-xs text-[var(--text-muted)] block mb-1">Start Date</label>
              <input type="date" value={startDate} onChange={e => setStartDate(e.target.value)}
                className="bg-[var(--bg-primary)] border border-[var(--border)] rounded px-3 py-2 text-sm text-[var(--text-primary)]" />
            </div>
            <div>
              <label className="text-xs text-[var(--text-muted)] block mb-1">End Date</label>
              <input type="date" value={endDate} onChange={e => setEndDate(e.target.value)}
                className="bg-[var(--bg-primary)] border border-[var(--border)] rounded px-3 py-2 text-sm text-[var(--text-primary)]" />
            </div>
            <button onClick={runBacktest} disabled={loading}
              className="px-6 py-2 bg-[var(--accent-blue)] text-white rounded text-sm font-medium
                         hover:bg-blue-600 disabled:opacity-40">
              {loading ? 'Running...' : 'Run Backtest'}
            </button>
          </div>
          {error && <p className="mt-3 text-xs text-[var(--accent-red)]">{error}</p>}
        </div>

        {/* Results */}
        {result && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <MetricCard label="Gross Sharpe" value={result.gross_sharpe.toFixed(3)}
              icon={<TrendingUp className="w-4 h-4" />} positive={result.gross_sharpe > 0} />
            <MetricCard label="Net Sharpe" value={result.net_sharpe.toFixed(3)}
              icon={<TrendingUp className="w-4 h-4" />} positive={result.net_sharpe > 0} />
            <MetricCard label="Annual Return" value={(result.annual_return * 100).toFixed(2) + '%'}
              icon={<TrendingUp className="w-4 h-4" />} positive={result.annual_return > 0} />
            <MetricCard label="Annual Vol" value={(result.annual_vol * 100).toFixed(2) + '%'}
              icon={<Activity className="w-4 h-4" />} positive={true} />
            <MetricCard label="Max Drawdown" value={(result.max_drawdown * 100).toFixed(2) + '%'}
              icon={<TrendingDown className="w-4 h-4" />} positive={false} />
            <MetricCard label="Avg Turnover" value={(result.avg_daily_turnover * 100).toFixed(2) + '%'}
              icon={<Activity className="w-4 h-4" />} positive={true} />
            <MetricCard label="Trading Days" value={result.n_days.toString()}
              icon={<BarChart3 className="w-4 h-4" />} positive={true} />
          </div>
        )}
      </main>
    </div>
  );
}

function MetricCard({ label, value, icon, positive }: {
  label: string; value: string; icon: React.ReactNode; positive: boolean;
}) {
  return (
    <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-lg p-4">
      <div className="flex items-center gap-2 mb-2">
        <span className={positive ? 'text-[var(--accent-green)]' : 'text-[var(--accent-red)]'}>{icon}</span>
        <span className="text-xs text-[var(--text-muted)] uppercase tracking-wider">{label}</span>
      </div>
      <div className={`text-lg font-bold ${positive ? 'text-[var(--text-primary)]' : 'text-[var(--accent-red)]'}`}>
        {value}
      </div>
    </div>
  );
}
