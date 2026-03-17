"""Risk analytics: VaR, CVaR, stress testing, position sizing."""

import numpy as np
import pandas as pd


def compute_var_cvar(returns: pd.Series, lookback: int = 500) -> dict:
    """Historical VaR and CVaR (Expected Shortfall).

    No parametric distribution assumed — purely empirical.
    """
    r = returns.dropna().tail(lookback).values
    if len(r) < 50:
        return {"error": "Insufficient data for VaR calculation"}

    var_95 = float(np.percentile(r, 5))
    var_99 = float(np.percentile(r, 1))
    cvar_95 = float(r[r <= var_95].mean()) if (r <= var_95).any() else var_95
    cvar_99 = float(r[r <= var_99].mean()) if (r <= var_99).any() else var_99

    return {
        "var_95": var_95,
        "var_99": var_99,
        "cvar_95": cvar_95,
        "cvar_99": cvar_99,
        "lookback_days": len(r),
    }


STRESS_SCENARIOS = {
    "2008_GFC": {
        "start": "2008-09-15",
        "end": "2009-03-09",
        "description": "Lehman to market bottom",
    },
    "2020_COVID": {
        "start": "2020-02-19",
        "end": "2020-03-23",
        "description": "Peak to trough",
    },
    "2022_RATE": {
        "start": "2022-01-03",
        "end": "2022-10-12",
        "description": "Rate hike selloff",
    },
}


def stress_test(spy_prices: pd.Series, position_value: float = 1_000_000) -> dict:
    """Apply historical SPY scenario paths to current position.

    Returns cumulative P&L, max drawdown, and recovery info per scenario.
    """
    results = {}
    for name, scenario in STRESS_SCENARIOS.items():
        mask = (spy_prices.index >= scenario["start"]) & (spy_prices.index <= scenario["end"])
        scenario_prices = spy_prices[mask]
        if len(scenario_prices) < 5:
            results[name] = {"error": "Insufficient data for this scenario period"}
            continue

        daily_returns = scenario_prices.pct_change().dropna()
        cumulative = (1 + daily_returns).cumprod()
        max_dd = float((cumulative / cumulative.cummax() - 1).min())
        total_return = float(cumulative.iloc[-1] - 1)

        results[name] = {
            "description": scenario["description"],
            "total_return": total_return,
            "max_drawdown": max_dd,
            "pnl": position_value * total_return,
            "duration_days": len(daily_returns),
        }
    return results


def position_size(
    alpha_q50: float,
    realized_vol_21d: float,
    conviction: float,
    target_vol: float = 0.15,
) -> float:
    """Volatility-scaled position sizing.

    More robust than Kelly with only 3 quantile points.
    Scales inversely with realized vol (risk parity intuition).
    Capped at 10% of portfolio.
    """
    raw_size = (target_vol / max(realized_vol_21d, 0.05)) * abs(alpha_q50) * conviction
    return max(0.0, min(raw_size, 0.10))
