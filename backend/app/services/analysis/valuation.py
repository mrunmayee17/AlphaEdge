"""Relative valuation — EV/EBITDA, P/E, P/B percentile ranks vs sector peers.

No DCF — relative valuation only. Uses Yahoo Finance fundamentals
to rank the target stock against sector peers.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def compute_percentile_ranks(
    target: dict,
    sector_peers: dict[str, dict],
) -> dict:
    """Compute percentile ranks for valuation metrics vs sector peers.

    Args:
        target: Fundamentals dict for the target ticker
        sector_peers: Dict of {ticker: fundamentals} for sector peers

    Returns:
        Dict with percentile ranks and raw values
    """
    metrics = ["pe_ratio", "forward_pe", "ev_ebitda", "price_to_book"]
    result = {}

    for metric in metrics:
        target_val = target.get(metric)
        if target_val is None:
            result[metric] = {"value": None, "percentile": None, "sector_median": None}
            continue

        peer_vals = [
            v.get(metric) for v in sector_peers.values()
            if v.get(metric) is not None and np.isfinite(v.get(metric, float("nan")))
        ]

        if len(peer_vals) < 3:
            result[metric] = {
                "value": target_val,
                "percentile": None,
                "sector_median": None,
                "note": "Insufficient peer data",
            }
            continue

        peer_vals_arr = np.array(peer_vals)
        percentile = float(np.mean(peer_vals_arr <= target_val) * 100)
        median = float(np.median(peer_vals_arr))

        # For valuation multiples, lower percentile = cheaper (more attractive)
        result[metric] = {
            "value": round(target_val, 2),
            "percentile": round(percentile, 1),
            "sector_median": round(median, 2),
            "vs_median_pct": round((target_val / median - 1) * 100, 1) if median != 0 else None,
            "n_peers": len(peer_vals),
        }

    # Quality metrics (higher = better)
    quality_metrics = ["profit_margin", "roe", "revenue_growth", "earnings_growth"]
    for metric in quality_metrics:
        target_val = target.get(metric)
        if target_val is None:
            result[metric] = {"value": None, "percentile": None}
            continue

        peer_vals = [
            v.get(metric) for v in sector_peers.values()
            if v.get(metric) is not None and np.isfinite(v.get(metric, float("nan")))
        ]

        if len(peer_vals) < 3:
            result[metric] = {"value": target_val, "percentile": None}
            continue

        peer_vals_arr = np.array(peer_vals)
        percentile = float(np.mean(peer_vals_arr <= target_val) * 100)

        result[metric] = {
            "value": round(target_val, 4) if abs(target_val) < 1 else round(target_val, 2),
            "percentile": round(percentile, 1),
            "sector_median": round(float(np.median(peer_vals_arr)), 4),
            "n_peers": len(peer_vals),
        }

    # Overall valuation assessment
    val_percentiles = [
        result[m]["percentile"]
        for m in metrics
        if result[m].get("percentile") is not None
    ]
    if val_percentiles:
        avg_val_pctile = np.mean(val_percentiles)
        if avg_val_pctile < 30:
            assessment = "CHEAP"
        elif avg_val_pctile < 70:
            assessment = "FAIR"
        else:
            assessment = "EXPENSIVE"
        result["overall_assessment"] = assessment
        result["avg_valuation_percentile"] = round(float(avg_val_pctile), 1)

    return result


def get_sector_constituents(sector: str, yahoo_finance_svc) -> list[str]:
    """Get top S&P 500 tickers in the same sector for peer comparison."""
    # Common large-cap tickers by sector for quick lookup
    SECTOR_TICKERS = {
        "Technology": ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "AMD", "ADBE", "CSCO", "INTC",
                       "TXN", "QCOM", "IBM", "NOW", "AMAT", "MU", "LRCX", "ADI", "KLAC", "SNPS"],
        "Information Technology": ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "AMD", "ADBE", "CSCO", "INTC"],
        "Financial Services": ["BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "SPGI", "BLK",
                               "C", "AXP", "SCHW", "CB", "MMC", "PGR", "USB", "AON", "CME", "ICE"],
        "Financials": ["BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "SPGI", "BLK"],
        "Healthcare": ["LLY", "UNH", "JNJ", "MRK", "ABBV", "TMO", "ABT", "DHR", "PFE", "AMGN",
                       "BMY", "MDT", "ISRG", "GILD", "VRTX", "SYK", "BSX", "REGN", "ZTS", "ELV"],
        "Health Care": ["LLY", "UNH", "JNJ", "MRK", "ABBV", "TMO", "ABT", "DHR", "PFE", "AMGN"],
        "Energy": ["XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PXD", "PSX", "VLO", "OXY",
                   "HES", "WMB", "KMI", "HAL", "FANG", "DVN", "BKR", "TRGP", "OKE", "CTRA"],
        "Industrials": ["GE", "CAT", "UNP", "HON", "RTX", "BA", "DE", "LMT", "UPS", "ADP",
                        "MMM", "GD", "ITW", "EMR", "NSC", "CSX", "WM", "PH", "TDG", "CARR"],
        "Consumer Cyclical": ["AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "CMG",
                              "MAR", "ORLY", "AZO", "ROST", "DHI", "LEN", "GM", "F", "YUM", "DG"],
        "Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "CMG"],
        "Consumer Defensive": ["PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "MDLZ", "CL", "GIS",
                               "KHC", "KMB", "SYY", "HSY", "K", "CAG", "TSN", "SJM", "CPB", "HRL"],
        "Consumer Staples": ["PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "MDLZ", "CL", "GIS"],
        "Basic Materials": ["LIN", "APD", "SHW", "ECL", "DD", "NEM", "FCX", "NUE", "VMC", "MLM"],
        "Materials": ["LIN", "APD", "SHW", "ECL", "DD", "NEM", "FCX", "NUE", "VMC", "MLM"],
        "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "ED", "WEC"],
        "Real Estate": ["PLD", "AMT", "EQIX", "CCI", "PSA", "O", "SPG", "WELL", "DLR", "AVB"],
        "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "T", "VZ", "CMCSA", "TMUS", "CHTR", "EA"],
    }
    return SECTOR_TICKERS.get(sector, [])[:20]
