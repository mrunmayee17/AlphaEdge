"""Quick API connectivity test — verifies all data sources are reachable."""
import os
import sys

results = {}

# 1. Yahoo Finance — Prices
print("=" * 60)
print("1. YAHOO FINANCE — Price Data")
print("=" * 60)
try:
    import yfinance as yf
    ticker = yf.Ticker("AAPL")
    hist = ticker.history(period="5d")
    if hist.empty:
        raise ValueError("No data returned")
    print(f"   AAPL last 5 days: {len(hist)} rows")
    print(f"   Latest close: ${hist['Close'].iloc[-1]:.2f}")
    results["yfinance_prices"] = "PASS"
except Exception as e:
    print(f"   FAILED: {e}")
    results["yfinance_prices"] = f"FAIL: {e}"

# 2. Yahoo Finance — Fundamentals (replaces Bloomberg)
print("\n" + "=" * 60)
print("2. YAHOO FINANCE — Fundamentals (Bloomberg replacement)")
print("=" * 60)
try:
    import yfinance as yf
    ticker = yf.Ticker("AAPL")
    info = ticker.info
    print(f"   Company: {info.get('longName', 'N/A')}")
    print(f"   Sector: {info.get('sector', 'N/A')}")
    print(f"   P/E Ratio: {info.get('trailingPE', 'N/A')}")
    print(f"   EV/EBITDA: {info.get('enterpriseToEbitda', 'N/A')}")
    print(f"   Market Cap: ${info.get('marketCap', 0)/1e9:.1f}B")
    print(f"   Target Price: ${info.get('targetMeanPrice', 'N/A')}")
    print(f"   Recommendation: {info.get('recommendationKey', 'N/A')}")

    # Financial statements
    inc = ticker.income_stmt
    bs = ticker.balance_sheet
    cf = ticker.cashflow
    print(f"   Income stmt periods: {len(inc.columns)}")
    print(f"   Balance sheet periods: {len(bs.columns)}")
    print(f"   Cash flow periods: {len(cf.columns)}")

    if not inc.empty and not bs.empty:
        results["yfinance_fundamentals"] = "PASS"
    else:
        results["yfinance_fundamentals"] = "PARTIAL — missing some statements"
except Exception as e:
    print(f"   FAILED: {e}")
    results["yfinance_fundamentals"] = f"FAIL: {e}"

# 3. Yahoo Finance — Analyst Estimates
print("\n" + "=" * 60)
print("3. YAHOO FINANCE — Analyst Estimates")
print("=" * 60)
try:
    import yfinance as yf
    ticker = yf.Ticker("AAPL")
    # Analyst recommendations
    recs = ticker.recommendations
    if recs is not None and not recs.empty:
        print(f"   Recommendations: {len(recs)} entries")
        print(f"   Latest: {recs.iloc[0].to_dict()}")
    else:
        print("   No recommendations data")

    # Earnings estimates
    earnings = ticker.earnings_estimate
    if earnings is not None and not earnings.empty:
        print(f"   Earnings estimate periods: {list(earnings.columns)}")
    else:
        print("   No earnings estimate data (may need different accessor)")

    results["yfinance_estimates"] = "PASS"
except Exception as e:
    print(f"   FAILED: {e}")
    results["yfinance_estimates"] = f"FAIL: {e}"

# 4. FRED API
print("\n" + "=" * 60)
print("4. FRED API")
print("=" * 60)
try:
    from fredapi import Fred
    fred = Fred(api_key=os.getenv("FRED_API_KEY", ""))
    dgs10 = fred.get_series("DGS10", observation_start="2024-01-01")
    dgs2 = fred.get_series("DGS2", observation_start="2024-01-01")
    fedfunds = fred.get_series("FEDFUNDS", observation_start="2024-01-01")
    print(f"   DGS10 (10Y yield): {len(dgs10)} obs, latest={dgs10.dropna().iloc[-1]:.2f}%")
    print(f"   DGS2 (2Y yield): {len(dgs2)} obs, latest={dgs2.dropna().iloc[-1]:.2f}%")
    print(f"   FEDFUNDS: {len(fedfunds)} obs, latest={fedfunds.dropna().iloc[-1]:.2f}%")
    print(f"   2s10s spread: {(dgs10.dropna().iloc[-1] - dgs2.dropna().iloc[-1]):.2f}%")
    results["fredapi"] = "PASS"
except Exception as e:
    print(f"   FAILED: {e}")
    results["fredapi"] = f"FAIL: {e}"

# 5. Brave Search API
print("\n" + "=" * 60)
print("5. BRAVE SEARCH API")
print("=" * 60)
try:
    import httpx
    brave_key = "BSAcJOx5z1SLdKAOOWbyVA7oONbpTOl"
    resp = httpx.get(
        "https://api.search.brave.com/res/v1/web/search",
        params={"q": "AAPL stock news today", "count": 3},
        headers={"X-Subscription-Token": brave_key, "Accept": "application/json"},
        timeout=10.0,
    )
    if resp.status_code == 200:
        data = resp.json()
        web_results = data.get("web", {}).get("results", [])
        print(f"   Query: 'AAPL stock news today'")
        print(f"   Results: {len(web_results)} items")
        for i, r in enumerate(web_results[:3]):
            print(f"   [{i+1}] {r.get('title', 'N/A')[:60]}")
        results["brave_search"] = "PASS"
    else:
        print(f"   HTTP {resp.status_code}: {resp.text[:200]}")
        results["brave_search"] = f"FAIL: HTTP {resp.status_code}"
except Exception as e:
    print(f"   FAILED: {e}")
    results["brave_search"] = f"FAIL: {e}"

# 6. Bright Data — Reddit Scraping
print("\n" + "=" * 60)
print("6. BRIGHT DATA — Reddit Scraping")
print("=" * 60)
try:
    import httpx
    import json
    headers = {
        "Authorization": "Bearer 4561152efcf7312d5da5ff4669b15cb439866b4a8e0e24a429e93032403261d1",
        "Content-Type": "application/json",
    }
    # Test: search for AAPL on wallstreetbets + stocks
    data = json.dumps({
        "input": [
            {"keyword": "AAPL", "date": "Past month", "num_of_posts": 3, "sort_by": "Hot"},
        ],
    })
    resp = httpx.post(
        "https://api.brightdata.com/datasets/v3/trigger?dataset_id=gd_lvz8ah06191smkebj4&notify=false&include_errors=true&type=discover_new&discover_by=keyword",
        headers=headers,
        content=data,
        timeout=15.0,
    )
    print(f"   HTTP {resp.status_code}")
    resp_data = resp.json()
    print(f"   Response: {json.dumps(resp_data, indent=2)[:300]}")
    if resp.status_code in (200, 201, 202):
        results["bright_data_reddit"] = "PASS"
    else:
        results["bright_data_reddit"] = f"FAIL: HTTP {resp.status_code}"
except Exception as e:
    print(f"   FAILED: {e}")
    results["bright_data_reddit"] = f"FAIL: {e}"

# 7. Fama-French Factors
print("\n" + "=" * 60)
print("7. FAMA-FRENCH FACTORS")
print("=" * 60)
try:
    import httpx, io, zipfile
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
    resp = httpx.get(url, timeout=15.0, follow_redirects=True)
    z = zipfile.ZipFile(io.BytesIO(resp.content))
    fname = z.namelist()[0]
    content = z.read(fname).decode("utf-8")
    data_lines = [l for l in content.split("\n") if l.strip() and l.strip()[:4].isdigit()]
    print(f"   File: {fname}")
    print(f"   Rows: {len(data_lines)}")
    print(f"   Range: {data_lines[0].split(',')[0].strip()} to {data_lines[-1].split(',')[0].strip()}")
    results["fama_french"] = "PASS"
except Exception as e:
    print(f"   FAILED: {e}")
    results["fama_french"] = f"FAIL: {e}"

# 8. Yahoo Finance — Sector ETFs + Macro
print("\n" + "=" * 60)
print("8. YAHOO FINANCE — Sector ETFs + Macro Data")
print("=" * 60)
try:
    import yfinance as yf
    etfs = ["XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLB", "XLU", "XLRE", "XLC"]
    macro = ["SPY", "^VIX", "TLT", "GLD", "DX-Y.NYB", "HYG", "LQD"]
    all_tickers = etfs + macro
    data = yf.download(all_tickers, period="5d", group_by="ticker", progress=False)
    available = [t for t in all_tickers if t in data.columns.get_level_values(0)]
    print(f"   Requested: {len(all_tickers)} tickers")
    print(f"   Received: {len(available)} tickers")
    missing = set(all_tickers) - set(available)
    if missing:
        print(f"   Missing: {missing}")
    else:
        print(f"   All sector ETFs + macro data available")
    results["yfinance_etfs_macro"] = "PASS" if len(available) >= 15 else f"PARTIAL: {len(available)}/{len(all_tickers)}"
except Exception as e:
    print(f"   FAILED: {e}")
    results["yfinance_etfs_macro"] = f"FAIL: {e}"

# ═══════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for api, status in results.items():
    icon = "✓" if "PASS" in str(status) else ("⚠" if "PARTIAL" in str(status) else "✗")
    print(f"  {icon} {api:30s} → {status}")

failed = [k for k, v in results.items() if "FAIL" in str(v)]
if failed:
    print(f"\nBLOCKERS: {', '.join(failed)}")
else:
    print("\nAll APIs operational — ready to build!")
