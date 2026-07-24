import pandas as pd
import numpy as np

from src.config import DEFAULT_RISK_FREE_RATE, TRADING_DAYS_PER_YEAR
from src.quant.models import PortfolioMetrics

def calculate_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the daily percentage returns from a DataFrame of prices.
    
    Args:
        prices (pd.DataFrame): DataFrame containing daily prices.
        
    Returns:
        pd.DataFrame: DataFrame containing daily returns, with the first row (NaN) dropped.
    """
    daily_returns = prices.pct_change()
    return daily_returns.dropna()

def calculate_annualized_returns(daily_returns: pd.DataFrame, days: int = TRADING_DAYS_PER_YEAR) -> pd.Series:
    """
    Calculates the annualized returns for each asset.
    
    Args:
        daily_returns (pd.DataFrame): DataFrame of daily returns.
        days (int): Number of trading days in a year.
        
    Returns:
        pd.Series: Annualized returns per asset.
    """
    return daily_returns.mean() * days

def calculate_annualized_volatility(daily_returns: pd.DataFrame, days: int = TRADING_DAYS_PER_YEAR) -> pd.Series:
    """
    Calculates the annualized volatility (standard deviation) for each asset.
    
    Args:
        daily_returns (pd.DataFrame): DataFrame of daily returns.
        days (int): Number of trading days in a year.
        
    Returns:
        pd.Series: Annualized volatility per asset.
    """
    return daily_returns.std() * np.sqrt(days)

def calculate_covariance_matrix(daily_returns: pd.DataFrame, days: int = TRADING_DAYS_PER_YEAR) -> pd.DataFrame:
    """
    Calculates the annualized covariance matrix of the assets.
    
    Args:
        daily_returns (pd.DataFrame): DataFrame of daily returns.
        days (int): Number of trading days in a year.
        
    Returns:
        pd.DataFrame: Annualized covariance matrix.
    """
    return daily_returns.cov() * days

def calculate_portfolio_performance(weights: np.ndarray, 
                                  ann_returns: pd.Series, 
                                  cov_matrix: pd.DataFrame, 
                                  daily_returns: pd.DataFrame | None = None,
                                  risk_free_rate: float = DEFAULT_RISK_FREE_RATE) -> PortfolioMetrics:
    """
    Calculates the expected return, volatility, Sharpe Ratio, and max drawdown of a portfolio.
    
    Args:
        weights (np.ndarray): Array of asset weights.
        ann_returns (pd.Series): Annualized returns of the assets.
        cov_matrix (pd.DataFrame): Annualized covariance matrix.
        daily_returns (pd.DataFrame | None): Daily returns of the assets for max drawdown calculation.
        risk_free_rate (float): The risk-free rate.
        
    Returns:
        PortfolioMetrics: Dataclass containing the performance metrics.
    """
    # Expected Return
    port_return = float(np.sum(ann_returns * weights))
    
    # Expected Volatility
    cov = cov_matrix.to_numpy()
    port_volatility = float(np.sqrt(weights.T @ cov @ weights))
    
    # Sharpe Ratio
    sharpe_ratio = (port_return - risk_free_rate) / port_volatility if port_volatility != 0 else 0.0
    
    # Max Drawdown
    max_drawdown = 0.0
    if daily_returns is not None:
        port_daily_returns = (daily_returns * weights).sum(axis=1)
        max_drawdown = calculate_max_drawdown(port_daily_returns)
    
    return PortfolioMetrics(
        expected_return=port_return,
        volatility=port_volatility,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown
    )

def calculate_max_drawdown(daily_returns: pd.Series) -> float:
    """
    Calculates the maximum drawdown of a single asset or portfolio's daily returns.
    
    Args:
        daily_returns (pd.Series): Daily returns of an asset or portfolio.
        
    Returns:
        float: The maximum drawdown (as a negative decimal, e.g., -0.25 for a 25% drop).
    """
    # Calculate cumulative wealth index
    cumulative_returns = (1 + daily_returns).cumprod()
    
    # Calculate running maximum
    running_max = cumulative_returns.cummax()
    
    # Calculate drawdown
    drawdown = (cumulative_returns - running_max) / running_max
    
    # Max drawdown is the minimum value (most negative)
    return float(drawdown.min())
