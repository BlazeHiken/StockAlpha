import numpy as np
import pandas as pd
from scipy.optimize import minimize
import logging

from src.quant.metrics import calculate_portfolio_performance
from src.quant.metrics import DEFAULT_RISK_FREE_RATE

logger = logging.getLogger(__name__)

def get_negative_sharpe(weights: np.ndarray, ann_returns: pd.Series, 
                        cov_matrix: pd.DataFrame, risk_free_rate: float) -> float:
    """Objective function to minimize (Negative Sharpe Ratio)."""
    _, _, sharpe = calculate_portfolio_performance(weights, ann_returns, cov_matrix, risk_free_rate)
    return -sharpe

def get_portfolio_volatility(weights: np.ndarray, cov_matrix: pd.DataFrame) -> float:
    """Objective function to minimize (Portfolio Volatility)."""
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def get_portfolio_return(weights: np.ndarray, ann_returns: pd.Series) -> float:
    """Helper to get portfolio return."""
    return np.sum(ann_returns * weights)

def maximize_sharpe_ratio(ann_returns: pd.Series, 
                            cov_matrix: pd.DataFrame, 
                            risk_free_rate: float = DEFAULT_RISK_FREE_RATE) -> dict:
    """
    Finds the optimal portfolio weights that maximize the Sharpe Ratio.
    
    Args:
        ann_returns (pd.Series): Annualized returns of the assets.
        cov_matrix (pd.DataFrame): Annualized covariance matrix.
        risk_free_rate (float): The risk-free rate (default 0.065 for Indian 10y G-Sec).
        
    Returns:
        dict: A dictionary containing 'weights', 'return', 'volatility', and 'sharpe_ratio'.
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
    opt_ret, opt_vol, opt_sharpe = calculate_portfolio_performance(
        optimal_weights, ann_returns, cov_matrix, risk_free_rate)
        
    return {
        'weights': dict(zip(ann_returns.index, optimal_weights)),
        'return': opt_ret,
        'volatility': opt_vol,
        'sharpe_ratio': opt_sharpe,
        'success': result.success
    }

def generate_efficient_frontier(ann_returns: pd.Series, cov_matrix: pd.DataFrame, 
                                num_points: int = 50, risk_free_rate: float = DEFAULT_RISK_FREE_RATE) -> pd.DataFrame:
    """
    Traces the efficient frontier by minimizing volatility for a range of target returns.
    
    Args:
        ann_returns (pd.Series): Annualized returns of the assets.
        cov_matrix (pd.DataFrame): Annualized covariance matrix.
        num_points (int): Number of points (portfolios) to generate on the frontier.
        risk_free_rate (float): The risk-free rate for Sharpe ratio calculations.
        
    Returns:
        pd.DataFrame: A DataFrame where each row is a portfolio on the frontier containing
                      its return, volatility, sharpe ratio, and asset weights.
    """
    num_assets = len(ann_returns)
    initial_guess = np.full(num_assets, 1 / num_assets)

    # Find the minimum and maximum possible returns
    # Max return is the return of the single asset with the highest return
    # Min return is the return of the minimum variance portfolio
    
    max_return = ann_returns.max()
    
    # Find Minimum Variance Portfolio to get the starting return for the efficient frontier
    min_var_result = minimize(
        get_portfolio_volatility,
        initial_guess,
        args=(cov_matrix,),
        method='SLSQP',
        bounds=tuple((0.0, 1.0) for _ in range(num_assets)),
        constraints={'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    )
    
    if not min_var_result.success:
        logger.warning(
            "Minimum variance optimization failed: %s",
            min_var_result.message
        )
        min_return = ann_returns.min()
    else:
        min_return = get_portfolio_return(min_var_result.x, ann_returns)
    
    # Create target returns between min_return and max_return
    target_returns = np.linspace(min_return, max_return, num_points)
    
    frontier_portfolios = []
    
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    
    for target in target_returns:
        # Constraints: 
        # 1. Weights sum to 1
        # 2. Portfolio return is equal to the target return
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
            port_ret, port_vol, port_sharpe = calculate_portfolio_performance(
                weights, ann_returns, cov_matrix, risk_free_rate)
                
            portfolio_data = {
                'return': port_ret,
                'volatility': port_vol,
                'sharpe_ratio': port_sharpe
            }
            # Add individual asset weights
            for i, asset in enumerate(ann_returns.index):
                portfolio_data[asset] = weights[i]
                
            frontier_portfolios.append(portfolio_data)
            
    return pd.DataFrame(frontier_portfolios)
