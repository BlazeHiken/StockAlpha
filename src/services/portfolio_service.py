import streamlit as st
import pandas as pd
from dataclasses import dataclass

from src.quant.ingest import fetch_stock_data
from src.quant.metrics import calculate_daily_returns, calculate_annualized_returns, calculate_covariance_matrix
from src.quant.optimizer import maximize_sharpe_ratio, generate_efficient_frontier
from src.quant.benchmark import compare_portfolios
from src.config import DEFAULT_RISK_FREE_RATE
from src.quant.models import PortfolioResult, BenchmarkResult

@dataclass
class PortfolioAnalysisData:
    """Aggregated results of the portfolio optimization process."""
    prices_df: pd.DataFrame
    opt_result: PortfolioResult
    benchmark_results: BenchmarkResult
    frontier_df: pd.DataFrame
    weight_sum: float

@st.cache_data
def fetch_data_cached(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches stock data and caches it to prevent redundant API calls.
    The cache invalidates when tickers, start_date, or end_date change.
    """
    # ensure tickers is sorted for consistent cache keys
    sorted_tickers = sorted(tickers)
    return fetch_stock_data(sorted_tickers, start_date, end_date)

def run_portfolio_analysis(tickers: list[str], start_date: str, end_date: str, risk_free_rate: float = DEFAULT_RISK_FREE_RATE) -> PortfolioAnalysisData | None:

    """
    Orchestrates the entire portfolio analysis flow:
    1. Fetches data
    2. Calculates metrics
    3. Runs optimization
    4. Benchmarks against naive portfolio
    5. Traces efficient frontier
    
    Args:
        tickers: List of stock symbols
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        risk_free_rate: The risk-free rate
        
    Returns:
        PortfolioAnalysisData or None if data fetching fails.
    """
    # 1. Fetch data
    prices_df = fetch_data_cached(tickers, start_date, end_date)
    
    if prices_df.empty or prices_df.shape[1] < 2:
        return None
        
    # 2. Calculate Base Metrics
    daily_returns = calculate_daily_returns(prices_df)
    ann_returns = calculate_annualized_returns(daily_returns)
    cov_matrix = calculate_covariance_matrix(daily_returns)

    # 3. Optimize Portfolio
    opt_result = maximize_sharpe_ratio(ann_returns, cov_matrix, daily_returns, risk_free_rate=risk_free_rate)
    
    # 4. Benchmark against Equal Weight
    benchmark_results = compare_portfolios(opt_result, ann_returns, cov_matrix, daily_returns, risk_free_rate=risk_free_rate)
    
    # 5. Generate Efficient Frontier
    frontier_df = generate_efficient_frontier(ann_returns, cov_matrix, risk_free_rate=risk_free_rate)
    
    # Calculate weight sum for validation
    weight_sum = sum(opt_result.weights.values())
    
    return PortfolioAnalysisData(
        prices_df=prices_df,
        opt_result=opt_result,
        benchmark_results=benchmark_results,
        frontier_df=frontier_df,
        weight_sum=weight_sum
    )
