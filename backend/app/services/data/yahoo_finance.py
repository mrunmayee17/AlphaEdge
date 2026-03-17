"""Yahoo Finance service — replaces Bloomberg for all price + fundamental data."""

import logging
import time

import pandas as pd
import yfinance as yf
from cachetools import TTLCache

logger = logging.getLogger(__name__)

# TTL cache: 1 hour for fundamentals (they don't change intraday)
_fundamentals_cache: TTLCache = TTLCache(maxsize=500, ttl=3600)
_info_cache: TTLCache = TTLCache(maxsize=500, ttl=3600)

# Common commodity / index futures aliases → Yahoo Finance tickers
FUTURES_ALIASES = {
    "CL": ("CL=F", "WTI Crude Oil Futures"),
    "GC": ("GC=F", "Gold Futures"),
    "SI": ("SI=F", "Silver Futures"),
    "NG": ("NG=F", "Natural Gas Futures"),
    "HG": ("HG=F", "Copper Futures"),
    "PL": ("PL=F", "Platinum Futures"),
    "PA": ("PA=F", "Palladium Futures"),
    "ZC": ("ZC=F", "Corn Futures"),
    "ZW": ("ZW=F", "Wheat Futures"),
    "ZS": ("ZS=F", "Soybean Futures"),
    "KC": ("KC=F", "Coffee Futures"),
    "CT": ("CT=F", "Cotton Futures"),
    "SB": ("SB=F", "Sugar Futures"),
    "CC": ("CC=F", "Cocoa Futures"),
    "ES": ("ES=F", "S&P 500 E-mini Futures"),
    "NQ": ("NQ=F", "Nasdaq 100 E-mini Futures"),
    "YM": ("YM=F", "Dow Jones E-mini Futures"),
    "RTY": ("RTY=F", "Russell 2000 E-mini Futures"),
    "ZB": ("ZB=F", "30-Year T-Bond Futures"),
    "ZN": ("ZN=F", "10-Year T-Note Futures"),
    "ZT": ("ZT=F", "2-Year T-Note Futures"),
    "DX": ("DX-Y.NYB", "US Dollar Index"),
    "BTC": ("BTC-USD", "Bitcoin"),
    "ETH": ("ETH-USD", "Ethereum"),
    "BZ": ("BZ=F", "Brent Crude Oil Futures"),
    "RB": ("RB=F", "RBOB Gasoline Futures"),
    "HO": ("HO=F", "Heating Oil Futures"),
}

# GICS sector → ETF mapping
SECTOR_ETF_MAP = {
    "Technology": "XLK",
    "Information Technology": "XLK",
    "Financial Services": "XLF",
    "Financials": "XLF",
    "Healthcare": "XLV",
    "Health Care": "XLV",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Consumer Cyclical": "XLY",
    "Consumer Discretionary": "XLY",
    "Consumer Defensive": "XLP",
    "Consumer Staples": "XLP",
    "Basic Materials": "XLB",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}


class YahooFinanceService:
    """All financial data via Yahoo Finance. No Bloomberg needed."""

    def resolve_ticker(self, raw_ticker: str) -> tuple[str, str]:
        """Resolve a user-input ticker to a Yahoo Finance ticker + display name.

        Handles common futures/commodity aliases (CL → CL=F "WTI Crude Oil Futures")
        and falls back to Yahoo Finance's own name for equities.

        Returns (yf_ticker, display_name).
        """
        upper = raw_ticker.upper()

        # Check futures aliases first
        if upper in FUTURES_ALIASES:
            yf_ticker, display_name = FUTURES_ALIASES[upper]
            logger.info(f"Resolved futures alias: {upper} → {yf_ticker} ({display_name})")
            return yf_ticker, display_name

        # For equities / anything else, get the name from Yahoo Finance
        info = self.get_ticker_info(upper)
        display_name = info.get("shortName") or info.get("longName") or upper
        return upper, display_name

    def get_price_history(self, ticker: str, start: str, end: str, retries: int = 3) -> pd.DataFrame:
        """OHLCV daily data with retry for rate limits."""
        for attempt in range(retries):
            try:
                df = yf.download(ticker, start=start, end=end, progress=False)
                if df.empty:
                    raise ValueError(f"No price data for {ticker} from {start} to {end}")
                return df
            except Exception as e:
                if "Rate" in str(e) and attempt < retries - 1:
                    wait = 2 ** (attempt + 1)
                    logger.warning(f"Yahoo Finance rate limited, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise

    def get_ticker_info(self, ticker: str, retries: int = 3) -> dict:
        """Cached ticker.info with retry for rate limits."""
        if ticker in _info_cache:
            return _info_cache[ticker]
        for attempt in range(retries):
            try:
                t = yf.Ticker(ticker)
                info = t.info
                _info_cache[ticker] = info
                return info
            except Exception as e:
                if "Rate" in str(e) and attempt < retries - 1:
                    wait = 2 ** (attempt + 1)
                    logger.warning(f"Yahoo Finance rate limited, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise

    def get_sector_etf(self, ticker: str) -> str:
        """Look up sector ETF for a given ticker."""
        info = self.get_ticker_info(ticker)
        sector = info.get("sector", "")
        etf = SECTOR_ETF_MAP.get(sector)
        if not etf:
            logger.warning(f"No sector ETF mapping for {ticker} sector={sector}, defaulting to SPY")
            return "SPY"
        return etf

    def get_fundamentals(self, ticker: str) -> dict:
        """Company info, P/E, EV/EBITDA, margins, etc."""
        if ticker in _fundamentals_cache:
            return _fundamentals_cache[ticker]

        info = self.get_ticker_info(ticker)
        result = {
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "ev_ebitda": info.get("enterpriseToEbitda"),
            "price_to_book": info.get("priceToBook"),
            "profit_margin": info.get("profitMargins"),
            "roe": info.get("returnOnEquity"),
            "debt_to_equity": info.get("debtToEquity"),
            "free_cash_flow": info.get("freeCashflow"),
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "dividend_yield": info.get("dividendYield"),
            "beta": info.get("beta"),
        }
        _fundamentals_cache[ticker] = result
        return result

    def get_financial_statements(self, ticker: str) -> dict:
        """Income statement, balance sheet, cash flow."""
        t = yf.Ticker(ticker)
        return {
            "income_stmt": t.income_stmt.to_dict() if not t.income_stmt.empty else {},
            "balance_sheet": t.balance_sheet.to_dict() if not t.balance_sheet.empty else {},
            "cashflow": t.cashflow.to_dict() if not t.cashflow.empty else {},
        }

    def get_analyst_estimates(self, ticker: str) -> dict:
        """Analyst recommendations and earnings estimates."""
        t = yf.Ticker(ticker)
        info = self.get_ticker_info(ticker)
        recs = t.recommendations
        return {
            "recommendations": recs.to_dict() if recs is not None and not recs.empty else {},
            "target_price": info.get("targetMeanPrice"),
            "target_high": info.get("targetHighPrice"),
            "target_low": info.get("targetLowPrice"),
            "recommendation_key": info.get("recommendationKey"),
            "num_analysts": info.get("numberOfAnalystOpinions"),
        }

    def get_options_data(self, ticker: str) -> dict:
        """Options chain for implied vol and put/call ratio."""
        t = yf.Ticker(ticker)
        dates = t.options
        if not dates:
            return {"implied_vol": None, "put_call_ratio": None, "expiration": None}
        chain = t.option_chain(dates[0])
        total_call_oi = chain.calls["openInterest"].sum()
        total_put_oi = chain.puts["openInterest"].sum()
        avg_call_iv = chain.calls["impliedVolatility"].mean()
        return {
            "put_call_ratio": float(total_put_oi / max(total_call_oi, 1)),
            "implied_vol": float(avg_call_iv),
            "expiration": dates[0],
        }

    def get_short_interest(self, ticker: str) -> dict:
        """Short interest data from yfinance info."""
        info = self.get_ticker_info(ticker)
        return {
            "short_ratio": info.get("shortRatio"),
            "short_pct_float": info.get("shortPercentOfFloat"),
            "shares_short": info.get("sharesShort"),
        }

    def get_sector_etf_prices(self, etf: str, start: str, end: str) -> pd.DataFrame:
        """Fetch sector ETF prices for alpha neutralization."""
        return self.get_price_history(etf, start, end)
