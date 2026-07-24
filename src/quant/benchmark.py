import numpy as np
import pandas as pd

from src.quant.metrics import calculate_portfolio_performance
from src.config import DEFAULT_RISK_FREE_RATE
from src.quant.models import PortfolioResult, BenchmarkResult

def generate_equal_weights(tickers: list[str]) -> np.ndarray:
    """
    Generates equal weights for a list of tickers.
    
    Args:
        tickers (list[str]): List of asset tickers.
        
    Returns:
        np.ndarray: Array of equal weights summing to 1.
    """
    n = len(tickers)
    if not tickers:
        raise ValueError("Ticker list cannot be empty.")
    return np.ones(n) / n

def compare_portfolios(optimal_result: PortfolioResult, 
                       ann_returns: pd.Series, 
                       cov_matrix: pd.DataFrame, 
                       daily_returns: pd.DataFrame,
                       risk_free_rate: float = DEFAULT_RISK_FREE_RATE) -> BenchmarkResult:
    """
    Compares the optimal portfolio against a naive equal-weight portfolio.
    
    Args:
        optimal_result (PortfolioResult): The optimized portfolio result.
        ann_returns (pd.Series): Annualized returns of the assets.
        cov_matrix (pd.DataFrame): Annualized covariance matrix.
        daily_returns (pd.DataFrame): Daily returns of the assets for max drawdown calculation.
        risk_free_rate (float): The risk-free rate.
        
    Returns:
        BenchmarkResult: A dataclass containing the comparison.
    """
    tickers = list(ann_returns.index)
    
    # --- Equal Weight Portfolio ---
    equal_weights_arr = generate_equal_weights(tickers)
    equal_weights_dict = dict(zip(tickers, equal_weights_arr))
    
    eq_metrics = calculate_portfolio_performance(
        equal_weights_arr, ann_returns, cov_matrix, daily_returns, risk_free_rate
    )
    
    equal_result = PortfolioResult(
        weights=equal_weights_dict,
        metrics=eq_metrics,
        success=True
    )
    
    # --- Comparison DataFrame ---
    optimal_metrics_dict = {
        "Return": optimal_result.metrics.expected_return if optimal_result.metrics else 0.0,
        "Volatility": optimal_result.metrics.volatility if optimal_result.metrics else 0.0,
        "Sharpe Ratio": optimal_result.metrics.sharpe_ratio if optimal_result.metrics else 0.0,
        "Max Drawdown": optimal_result.metrics.max_drawdown if optimal_result.metrics else 0.0
    }
    
    equal_metrics_dict = {
        "Return": eq_metrics.expected_return,
        "Volatility": eq_metrics.volatility,
        "Sharpe Ratio": eq_metrics.sharpe_ratio,
        "Max Drawdown": eq_metrics.max_drawdown
    }
    
    comparison_df = pd.DataFrame({
        "Optimal (Max Sharpe)": optimal_metrics_dict,
        "Naive (Equal Weight)": equal_metrics_dict
    })
    
    return BenchmarkResult(
        optimized=optimal_result,
        equal_weight=equal_result,
        comparison_df=comparison_df
    )
