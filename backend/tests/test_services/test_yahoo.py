"""Test Yahoo Finance service with real API calls."""

import pytest
from backend.app.services.data.yahoo_finance import YahooFinanceService

pytestmark = pytest.mark.integration


@pytest.fixture
def yf():
    return YahooFinanceService()


def test_price_history(yf):
    df = yf.get_price_history("AAPL", "2025-01-01", "2025-01-10")
    assert len(df) > 0
    assert "Close" in df.columns.get_level_values(0) or "Close" in df.columns


def test_fundamentals(yf):
    fund = yf.get_fundamentals("AAPL")
    assert fund["sector"] is not None
    assert fund["pe_ratio"] is not None
    assert fund["market_cap"] is not None
    assert fund["market_cap"] > 1e9  # Apple is worth more than $1B


def test_sector_etf(yf):
    etf = yf.get_sector_etf("AAPL")
    assert etf == "XLK"


def test_financial_statements(yf):
    stmts = yf.get_financial_statements("AAPL")
    assert "income_stmt" in stmts
    assert "balance_sheet" in stmts
    assert "cashflow" in stmts
    assert len(stmts["income_stmt"]) > 0


def test_analyst_estimates(yf):
    est = yf.get_analyst_estimates("AAPL")
    assert est["target_price"] is not None
    assert est["recommendation_key"] is not None


def test_short_interest(yf):
    short = yf.get_short_interest("AAPL")
    assert short["short_ratio"] is not None


def test_options_data(yf):
    opts = yf.get_options_data("AAPL")
    assert opts["implied_vol"] is not None
    assert opts["put_call_ratio"] is not None
