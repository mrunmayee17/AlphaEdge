"""Backtest engine — long-short decile portfolio with transaction costs.

Strategy: Sort S&P 500 by alpha_21d prediction, long top decile, short bottom decile.
Rebalance weekly (Monday close). Includes spread, impact, and commission costs.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Transaction Cost Model ────────────────────────────────────────────────


def classify_cap_tier(market_cap: float) -> str:
    """Classify market cap into tiers for cost estimation."""
    if market_cap >= 200e9:
        return "mega"
    elif market_cap >= 10e9:
        return "large"
    elif market_cap >= 2e9:
        return "mid"
    return "small"


SPREAD_BPS = {"mega": 1, "large": 3, "mid": 8, "small": 20}
K_IMPACT = {"mega": 0.05, "large": 0.15, "mid": 0.4, "small": 0.8}
COMMISSION_PER_DOLLAR = 0.0001  # ~1bp institutional rate


def transaction_cost(
    trade_dollars: float,
    adv_dollars: float,
    cap_tier: str,
) -> float:
    """Total one-way transaction cost as a fraction of trade value.

    Components:
    - Half spread (bid-ask)
    - Market impact (Almgren-Chriss square root model)
    - Commission (~1bp)
    """
    spread_cost = 0.5 * SPREAD_BPS.get(cap_tier, 8) / 10_000
    impact_cost = K_IMPACT.get(cap_tier, 0.4) * math.sqrt(
        abs(trade_dollars) / max(adv_dollars, 1.0)
    )
    return spread_cost + impact_cost + COMMISSION_PER_DOLLAR


# ── Portfolio Construction ────────────────────────────────────────────────


@dataclass
class BacktestConfig:
    """Configuration for the backtest engine."""

    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"
    rebalance_day: int = 0  # Monday
    decile_pct: float = 0.10  # top/bottom 10%
    min_adv_dollars: float = 5_000_000  # $5M minimum ADV filter
    position_weight: str = "equal"  # equal weight within each leg


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    daily_returns: pd.Series = field(default_factory=pd.Series)
    gross_returns: pd.Series = field(default_factory=pd.Series)
    net_returns: pd.Series = field(default_factory=pd.Series)
    turnover: pd.Series = field(default_factory=pd.Series)
    cost_series: pd.Series = field(default_factory=pd.Series)
    n_longs: pd.Series = field(default_factory=pd.Series)
    n_shorts: pd.Series = field(default_factory=pd.Series)

    @property
    def gross_sharpe(self) -> float:
        if len(self.gross_returns) == 0:
            return 0.0
        return float(
            self.gross_returns.mean() / max(self.gross_returns.std(), 1e-8) * np.sqrt(252)
        )

    @property
    def net_sharpe(self) -> float:
        if len(self.net_returns) == 0:
            return 0.0
        return float(
            self.net_returns.mean() / max(self.net_returns.std(), 1e-8) * np.sqrt(252)
        )

    @property
    def max_drawdown(self) -> float:
        if len(self.net_returns) == 0:
            return 0.0
        cum = (1 + self.net_returns).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        return float(dd.min())

    @property
    def annual_return(self) -> float:
        if len(self.net_returns) == 0:
            return 0.0
        return float(self.net_returns.mean() * 252)

    @property
    def annual_vol(self) -> float:
        if len(self.net_returns) == 0:
            return 0.0
        return float(self.net_returns.std() * np.sqrt(252))

    @property
    def avg_turnover(self) -> float:
        if len(self.turnover) == 0:
            return 0.0
        return float(self.turnover.mean())

    def summary(self) -> dict:
        return {
            "gross_sharpe": round(self.gross_sharpe, 3),
            "net_sharpe": round(self.net_sharpe, 3),
            "annual_return": round(self.annual_return, 4),
            "annual_vol": round(self.annual_vol, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "avg_daily_turnover": round(self.avg_turnover, 4),
            "n_days": len(self.net_returns),
        }


class BacktestEngine:
    """Long-short decile backtest with transaction costs.

    Requires:
    - predictions: DataFrame with columns [date, ticker, alpha_21d]
    - prices: DataFrame with columns [date, ticker, adj_close, volume]
    - market_caps: DataFrame with columns [date, ticker, market_cap]
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    def run(
        self,
        predictions: pd.DataFrame,
        prices: pd.DataFrame,
        market_caps: pd.DataFrame,
        risk_free: Optional[pd.Series] = None,
    ) -> BacktestResult:
        """Execute full backtest.

        Args:
            predictions: Columns [date, ticker, alpha_21d]
            prices: Columns [date, ticker, adj_close, volume]
            market_caps: Columns [date, ticker, market_cap]
            risk_free: Optional daily risk-free rate series (annualized / 252)
        """
        cfg = self.config

        # Pivot prices for fast lookups
        price_pivot = prices.pivot(index="date", columns="ticker", values="adj_close")
        volume_pivot = prices.pivot(index="date", columns="ticker", values="volume")

        # Compute 20-day ADV in dollars
        adv_dollars = (price_pivot * volume_pivot).rolling(20).mean()

        # Get rebalance dates (Mondays within range)
        all_dates = price_pivot.index.sort_values()
        start = pd.Timestamp(cfg.start_date)
        end = pd.Timestamp(cfg.end_date)
        rebal_dates = [
            d for d in all_dates
            if start <= d <= end and d.weekday() == cfg.rebalance_day
        ]

        # Track portfolio positions
        current_longs: dict[str, float] = {}  # ticker → weight
        current_shorts: dict[str, float] = {}

        daily_gross = []
        daily_net = []
        daily_turnover = []
        daily_cost = []
        daily_n_longs = []
        daily_n_shorts = []
        result_dates = []

        # Build daily returns
        daily_returns = price_pivot.pct_change()

        for i, d in enumerate(all_dates):
            if d < start or d > end:
                continue

            # Check if rebalance day
            if d in rebal_dates:
                # Get predictions for this date (use Friday's predictions for Monday execution)
                friday = d - pd.offsets.BDay(1)
                day_preds = predictions[predictions["date"] <= friday]
                if len(day_preds) == 0:
                    continue
                # Use most recent prediction per ticker
                latest_preds = day_preds.sort_values("date").groupby("ticker").last()

                # Liquidity filter
                adv_today = adv_dollars.loc[:d].iloc[-1] if d in adv_dollars.index else None
                if adv_today is None:
                    continue

                valid_tickers = [
                    t for t in latest_preds.index
                    if t in adv_today.index and adv_today.get(t, 0) >= cfg.min_adv_dollars
                    and t in daily_returns.columns
                ]

                if len(valid_tickers) < 20:
                    continue

                # Sort by alpha_21d prediction
                signals = latest_preds.loc[valid_tickers, "alpha_21d"].dropna().sort_values()
                n_decile = max(1, int(len(signals) * cfg.decile_pct))

                short_tickers = list(signals.index[:n_decile])
                long_tickers = list(signals.index[-n_decile:])

                # Equal weight
                long_w = 1.0 / max(len(long_tickers), 1)
                short_w = 1.0 / max(len(short_tickers), 1)

                new_longs = {t: long_w for t in long_tickers}
                new_shorts = {t: short_w for t in short_tickers}

                # Compute turnover and costs
                all_tickers = set(
                    list(current_longs) + list(current_shorts)
                    + list(new_longs) + list(new_shorts)
                )

                turn = 0.0
                cost = 0.0
                for t in all_tickers:
                    old_pos = current_longs.get(t, 0) - current_shorts.get(t, 0)
                    new_pos = new_longs.get(t, 0) - new_shorts.get(t, 0)
                    trade_frac = abs(new_pos - old_pos)
                    turn += trade_frac

                    if trade_frac > 0 and t in market_caps.columns:
                        mcap = market_caps.loc[:d, t].dropna()
                        cap_tier = classify_cap_tier(float(mcap.iloc[-1])) if len(mcap) > 0 else "mid"
                        adv_val = float(adv_today.get(t, 1e6))
                        tc = transaction_cost(trade_frac * 1e6, adv_val, cap_tier)
                        cost += trade_frac * tc

                current_longs = new_longs
                current_shorts = new_shorts

            # Daily P&L
            if not current_longs and not current_shorts:
                continue

            day_ret = daily_returns.loc[d] if d in daily_returns.index else None
            if day_ret is None:
                continue

            long_ret = sum(
                current_longs.get(t, 0) * day_ret.get(t, 0)
                for t in current_longs
            )
            short_ret = sum(
                current_shorts.get(t, 0) * day_ret.get(t, 0)
                for t in current_shorts
            )

            # Dollar-neutral: long leg - short leg
            gross_ret = 0.5 * (long_ret - short_ret)

            # Apply costs on rebalance days
            net_ret = gross_ret
            day_cost = 0.0
            day_turn = 0.0
            if d in rebal_dates:
                day_cost = cost
                day_turn = turn
                net_ret = gross_ret - day_cost

            result_dates.append(d)
            daily_gross.append(gross_ret)
            daily_net.append(net_ret)
            daily_turnover.append(day_turn)
            daily_cost.append(day_cost)
            daily_n_longs.append(len(current_longs))
            daily_n_shorts.append(len(current_shorts))

        idx = pd.DatetimeIndex(result_dates)
        result = BacktestResult(
            gross_returns=pd.Series(daily_gross, index=idx),
            net_returns=pd.Series(daily_net, index=idx),
            turnover=pd.Series(daily_turnover, index=idx),
            cost_series=pd.Series(daily_cost, index=idx),
            n_longs=pd.Series(daily_n_longs, index=idx),
            n_shorts=pd.Series(daily_n_shorts, index=idx),
        )

        logger.info(f"Backtest complete: {result.summary()}")
        return result
