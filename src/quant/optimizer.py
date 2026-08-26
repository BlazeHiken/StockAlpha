import numpy as np
import pandas as pd
from scipy.optimize import minimize
import logging

from src.quant.metrics import calculate_portfolio_performance
from src.config import DEFAULT_RISK_FREE_RATE, FRONTIER_POINTS
from src.quant.models import PortfolioResult

logger = logging.getLogger(__name__)

def get_negative_sharpe(weights: np.ndarray, ann_returns: pd.Series, 
                        cov_matrix: pd.DataFrame, risk_free_rate: float) -> float:
    """Objective function to minimize (Negative Sharpe Ratio)."""
    metrics = calculate_portfolio_performance(weights, ann_returns, cov_matrix, risk_free_rate=risk_free_rate)
    return -metrics.sharpe_ratio

def get_portfolio_volatility(weights: np.ndarray, cov_matrix: pd.DataFrame) -> float:
    """Objective function to minimize (Portfolio Volatility)."""
    cov = cov_matrix.to_numpy()
    return float(np.sqrt(weights.T @ cov @ weights))

def get_portfolio_return(weights: np.ndarray, ann_returns: pd.Series) -> float:
    """Helper to get portfolio return."""
    return float(np.sum(ann_returns * weights))

def maximize_sharpe_ratio(ann_returns: pd.Series, 
                          cov_matrix: pd.DataFrame, 
                          daily_returns: pd.DataFrame | None = None,
                          risk_free_rate: float = DEFAULT_RISK_FREE_RATE) -> PortfolioResult:
    """
    Finds the optimal portfolio weights that maximize the Sharpe Ratio.
    
    Args:
        ann_returns (pd.Series): Annualized returns of the assets.
        cov_matrix (pd.DataFrame): Annualized covariance matrix.
        daily_returns (pd.DataFrame | None): Daily returns for max drawdown calculation.
        risk_free_rate (float): The risk-free rate.
        
    Returns:
        PortfolioResult: A dataclass containing weights, metrics, and success status.
    """
    num_assets = len(ann_returns)
    args = (ann_returns, cov_matrix, risk_free_rate)
    
    # Constraints: weights sum to 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # Bounds: 0 <= weight <= 1 for each asset
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    
    # Initial guess: equal distribution
    initial_guess = np.full(num_assets, 1 / num_assets)
    
    # Optimization
    result = minimize(get_negative_sharpe, initial_guess, args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)
    
    if not result.success:
        logger.warning(f"Optimization failed: {result.message}")
        
    optimal_weights = result.x
    metrics = calculate_portfolio_performance(
        optimal_weights, ann_returns, cov_matrix, daily_returns, risk_free_rate)
        
    weights_dict = dict(zip(ann_returns.index, optimal_weights))
    
    return PortfolioResult(
        weights=weights_dict,
        in_sample_metrics=metrics,
        success=bool(result.success),
        error_message=result.message if not result.success else None
    )

def generate_efficient_frontier(ann_returns: pd.Series, cov_matrix: pd.DataFrame, 
                                num_points: int = FRONTIER_POINTS, risk_free_rate: float = DEFAULT_RISK_FREE_RATE) -> pd.DataFrame:
    """
    Traces the efficient frontier by minimizing volatility for a range of target returns.
    
    Args:
        ann_returns (pd.Series): Annualized returns of the assets.
        cov_matrix (pd.DataFrame): Annualized covariance matrix.
        num_points (int): Number of points (portfolios) to generate on the frontier.
        risk_free_rate (float): The risk-free rate for Sharpe ratio calculations.
        
    Returns:
        pd.DataFrame: A DataFrame where each row is a portfolio on the frontier.
    """
    num_assets = len(ann_returns)
    initial_guess = np.full(num_assets, 1 / num_assets)

    max_return = float(ann_returns.max())
    
    # Find Minimum Variance Portfolio
    min_var_result = minimize(
        get_portfolio_volatility,
        initial_guess,
        args=(cov_matrix,),
        method='SLSQP',
        bounds=tuple((0.0, 1.0) for _ in range(num_assets)),
        constraints={'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    )
    
    if not min_var_result.success:
        logger.warning("Minimum variance optimization failed: %s", min_var_result.message)
        min_return = float(ann_returns.min())
    else:
        min_return = get_portfolio_return(min_var_result.x, ann_returns)
    
    target_returns = np.linspace(min_return, max_return, num_points)
    frontier_portfolios = []
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    
    for target in target_returns:
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x, target=target: get_portfolio_return(x, ann_returns) - target}
        )
        
        result = minimize(
            get_portfolio_volatility, 
            initial_guess,
            args=(cov_matrix,), 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints
        )
        
        if result.success:
            weights = result.x
            metrics = calculate_portfolio_performance(
                weights, ann_returns, cov_matrix, risk_free_rate=risk_free_rate)
                
            portfolio_data = {
                'return': metrics.expected_return,
                'volatility': metrics.volatility,
                'sharpe_ratio': metrics.sharpe_ratio
            }
            # Add individual asset weights
            for i, asset in enumerate(ann_returns.index):
                portfolio_data[asset] = weights[i]
                
            frontier_portfolios.append(portfolio_data)
            
    return pd.DataFrame(frontier_portfolios)
