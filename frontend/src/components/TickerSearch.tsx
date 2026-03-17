import { useState, useCallback } from 'react';
import { Search, Loader2 } from 'lucide-react';
import { useAnalysisStore } from '../store/analysisStore';

export function TickerSearch() {
  const [input, setInput] = useState('');
  const { runAnalysis, status, reset } = useAnalysisStore();
  const isLoading = status !== 'idle' && status !== 'complete' && status !== 'error';

  const handleSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    const ticker = input.trim().toUpperCase();
    if (!ticker) return;
    reset();
    await runAnalysis(ticker);
  }, [input, runAnalysis, reset]);

  return (
    <form onSubmit={handleSubmit} className="flex items-center gap-3">
      <div className="relative flex-1 max-w-md">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[var(--text-muted)]" />
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value.toUpperCase())}
          placeholder="Enter ticker (e.g. AAPL)"
          maxLength={10}
          className="w-full pl-10 pr-4 py-2.5 bg-[var(--bg-card)] border border-[var(--border)] rounded-lg
                     text-[var(--text-primary)] placeholder:text-[var(--text-muted)]
                     focus:outline-none focus:border-[var(--accent-blue)] transition-colors
                     font-mono text-sm"
          disabled={isLoading}
        />
      </div>
      <button
        type="submit"
        disabled={isLoading || !input.trim()}
        className="px-6 py-2.5 bg-[var(--accent-blue)] text-white rounded-lg font-mono text-sm
                   hover:bg-blue-600 transition-colors disabled:opacity-40 disabled:cursor-not-allowed
                   flex items-center gap-2"
      >
        {isLoading ? (
          <>
            <Loader2 className="w-4 h-4 animate-spin" />
            Analyzing...
          </>
        ) : (
          'Analyze'
        )}
      </button>
    </form>
  );
}
