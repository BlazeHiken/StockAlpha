import streamlit as st
import pandas as pd
from dataclasses import dataclass
import numpy as np

from src.quant.ingest import fetch_stock_data
from src.quant.metrics import (
    calculate_daily_returns, 
    calculate_annualized_returns, 
    calculate_covariance_matrix,
    split_data,
    calculate_correlation_matrix
)
from src.quant.optimizer import maximize_sharpe_ratio, generate_efficient_frontier
from src.quant.benchmark import compare_portfolios, evaluate_portfolio
from src.config import DEFAULT_RISK_FREE_RATE, TRAIN_TEST_SPLIT_RATIO, MARKET_BENCHMARK
from src.quant.models import PortfolioResult, BenchmarkResult

@dataclass
class PortfolioAnalysisData:
    """Aggregated results of the portfolio optimization process."""
    prices_df: pd.DataFrame
    opt_result: PortfolioResult
    benchmark_results: BenchmarkResult
    frontier_df: pd.DataFrame
    correlation_df: pd.DataFrame
    train_returns: pd.DataFrame
    test_returns: pd.DataFrame
    market_train: pd.Series | None
    market_test: pd.Series | None
    weight_sum: float

@st.cache_data
def fetch_data_cached(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches stock data and caches it to prevent redundant API calls.
    The cache invalidates when tickers, start_date, or end_date change.
    """
    sorted_tickers = sorted(tickers)
    return fetch_stock_data(sorted_tickers, start_date, end_date)

def run_portfolio_analysis(tickers: list[str], start_date: str, end_date: str, risk_free_rate: float = DEFAULT_RISK_FREE_RATE) -> PortfolioAnalysisData | None:
    """Orchestrates the entire portfolio analysis flow with proper out-of-sample backtesting."""
    
    # Append MARKET_BENCHMARK to tickers if not present
    fetch_tickers = list(set(tickers + [MARKET_BENCHMARK]))
    
    # 1. Fetch data
    prices_df = fetch_data_cached(fetch_tickers, start_date, end_date)
    
    if prices_df.empty:
        return None
        
    # Separate market benchmark
    market_prices = None
    if MARKET_BENCHMARK in prices_df.columns:
        market_prices = prices_df[MARKET_BENCHMARK]
        asset_prices = prices_df.drop(columns=[MARKET_BENCHMARK])
    else:
        asset_prices = prices_df
        
    if asset_prices.shape[1] < 2:
        return None
        
    # 2. Calculate Base Metrics
    daily_returns = calculate_daily_returns(asset_prices)
    
    market_returns = None
    market_train = None
    market_test = None
    
    if market_prices is not None:
        market_returns = calculate_daily_returns(market_prices.to_frame())[MARKET_BENCHMARK]
        # Align dates
        market_returns = market_returns.reindex(daily_returns.index).fillna(0)
    
    # 3. Chronological Train/Test Split
    train_returns, test_returns = split_data(daily_returns, ratio=TRAIN_TEST_SPLIT_RATIO)
    
    if market_returns is not None:
        market_train, market_test = split_data(market_returns.to_frame(), ratio=TRAIN_TEST_SPLIT_RATIO)
        market_train = market_train[MARKET_BENCHMARK]
        market_test = market_test[MARKET_BENCHMARK]
    
    # 4. Calculate metrics on Train set ONLY
    train_ann_returns = calculate_annualized_returns(train_returns)
    train_cov_matrix = calculate_covariance_matrix(train_returns)
    
    # Correlation Matrix (full period)
    correlation_df = calculate_correlation_matrix(daily_returns)
    
    # 5. Optimize Portfolio (Train Set)
    opt_result_in_sample = maximize_sharpe_ratio(train_ann_returns, train_cov_matrix, train_returns, risk_free_rate=risk_free_rate)
    
    # 6. Evaluate Portfolio on Test Set (Out-of-Sample)
    opt_result_out_of_sample = evaluate_portfolio(opt_result_in_sample.weights, test_returns, risk_free_rate)
    
    opt_result = PortfolioResult(
        weights=opt_result_in_sample.weights,
        in_sample_metrics=opt_result_in_sample.in_sample_metrics,
        out_of_sample_metrics=opt_result_out_of_sample,
        success=opt_result_in_sample.success,
        error_message=opt_result_in_sample.error_message
    )
    
    # 7. Benchmark against Naive and Market
    benchmark_results = compare_portfolios(
        opt_result, 
        train_returns, 
        test_returns, 
        market_train, 
        market_test, 
        risk_free_rate=risk_free_rate
    )
    
    # 8. Generate Efficient Frontier (using train data only)
    frontier_df = generate_efficient_frontier(train_ann_returns, train_cov_matrix, risk_free_rate=risk_free_rate)
    
    weight_sum = sum(opt_result.weights.values())
    
    return PortfolioAnalysisData(
        prices_df=prices_df,
        opt_result=opt_result,
        benchmark_results=benchmark_results,
        frontier_df=frontier_df,
        correlation_df=correlation_df,
        train_returns=train_returns,
        test_returns=test_returns,
        market_train=market_train,
        market_test=market_test,
        weight_sum=weight_sum
    )
