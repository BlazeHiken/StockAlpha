"""
Centralized configuration module for StockAlpha.
"""

DEFAULT_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", 
    "INFY.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", 
    "HINDUNILVR.NS", "LARSEN.NS"
]

DEFAULT_RISK_FREE_RATE = 0.065  # Indian 10y G-Sec
MIN_STOCKS = 2
LOOKBACK_YEARS = 3
FRONTIER_POINTS = 50
TRADING_DAYS_PER_YEAR = 252

# Backtesting & Analytics
TRAIN_TEST_SPLIT_RATIO = 0.8  # 80% train, 20% test chronologically
MARKET_BENCHMARK = "^NSEI"    # NIFTY 50 index
