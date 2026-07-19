import pandas as pd
import numpy as np

DEFAULT_RISK_FREE_RATE = 0.065

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

def calculate_annualized_returns(daily_returns: pd.DataFrame, days: int = 252) -> pd.Series:
    """
    Calculates the annualized returns for each asset.
    
    Args:
        daily_returns (pd.DataFrame): DataFrame of daily returns.
        days (int): Number of trading days in a year (default 252).
        
    Returns:
        pd.Series: Annualized returns per asset.
    """
    return daily_returns.mean() * days

def calculate_annualized_volatility(daily_returns: pd.DataFrame, days: int = 252) -> pd.Series:
    """
    Calculates the annualized volatility (standard deviation) for each asset.
    
    Args:
        daily_returns (pd.DataFrame): DataFrame of daily returns.
        days (int): Number of trading days in a year (default 252).
        
    Returns:
        pd.Series: Annualized volatility per asset.
    """
    return daily_returns.std() * np.sqrt(days)

def calculate_covariance_matrix(daily_returns: pd.DataFrame, days: int = 252) -> pd.DataFrame:
    """
    Calculates the annualized covariance matrix of the assets.
    
    Args:
        daily_returns (pd.DataFrame): DataFrame of daily returns.
        days (int): Number of trading days in a year (default 252).
        
    Returns:
        pd.DataFrame: Annualized covariance matrix.
    """
    return daily_returns.cov() * days

def calculate_portfolio_performance(weights: np.ndarray, 
                                  ann_returns: pd.Series, 
                                  cov_matrix: pd.DataFrame, 
                                  risk_free_rate: float = DEFAULT_RISK_FREE_RATE) -> tuple[float, float, float]:
    """
    Calculates the expected return, volatility, and Sharpe Ratio of a portfolio.
    
    Args:
        weights (np.ndarray): Array of asset weights.
        ann_returns (pd.Series): Annualized returns of the assets.
        cov_matrix (pd.DataFrame): Annualized covariance matrix.
        risk_free_rate (float): The risk-free rate (default 0.065 for Indian 10y G-Sec).
        
    Returns:
        tuple[float, float, float]: (expected_return, volatility, sharpe_ratio)
    """
    # Expected Return
    port_return = np.sum(ann_returns * weights)
    
    # Expected Volatility
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # Sharpe Ratio
    sharpe_ratio = (port_return - risk_free_rate) / port_volatility if port_volatility != 0 else 0.0
    
    return port_return, port_volatility, sharpe_ratio

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
    return drawdown.min()
