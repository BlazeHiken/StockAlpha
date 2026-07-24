import pytest
import pandas as pd
import numpy as np

from src.quant.metrics import (
    calculate_daily_returns,
    calculate_annualized_returns,
    calculate_annualized_volatility,
    calculate_covariance_matrix,
    calculate_portfolio_performance,
    calculate_max_drawdown
)
from src.quant.optimizer import maximize_sharpe_ratio, generate_efficient_frontier

@pytest.fixture
def dummy_prices():
    """Returns a simple DataFrame with mock prices for 2 assets."""
    dates = pd.date_range("2023-01-01", periods=5)
    data = {
        "A": [100.0, 102.0, 104.04, 106.1208, 108.2432], # Steady 2% increase
        "B": [100.0, 95.0,  90.25,  85.7375,  81.4506]   # Steady 5% decrease
    }
    return pd.DataFrame(data, index=dates)

@pytest.fixture
def dummy_returns(dummy_prices):
    return calculate_daily_returns(dummy_prices)

def test_calculate_daily_returns(dummy_prices):
    returns = calculate_daily_returns(dummy_prices)
    assert len(returns) == 4 # One less than prices due to pct_change
    assert np.isclose(returns["A"].iloc[0], 0.02)
    assert np.isclose(returns["B"].iloc[0], -0.05)

def test_calculate_annualized_metrics(dummy_returns):
    ann_returns = calculate_annualized_returns(dummy_returns, days=252)
    ann_vol = calculate_annualized_volatility(dummy_returns, days=252)
    
    # Asset A has constant 2% daily return
    assert np.isclose(ann_returns["A"], 0.02 * 252)
    # Asset A has ~0 volatility because the return is perfectly constant
    assert np.isclose(ann_vol["A"], 0.0, atol=1e-5)

def test_calculate_covariance_matrix(dummy_returns):
    cov_matrix = calculate_covariance_matrix(dummy_returns)

    assert isinstance(cov_matrix, pd.DataFrame)
    assert cov_matrix.shape == (2, 2)
    assert list(cov_matrix.columns) == ["A", "B"]
    assert list(cov_matrix.index) == ["A", "B"]

def test_calculate_portfolio_performance(dummy_returns):
    ann_returns = calculate_annualized_returns(dummy_returns, days=252)
    cov_matrix = calculate_covariance_matrix(dummy_returns, days=252)
    
    weights = np.array([0.5, 0.5])
    metrics = calculate_portfolio_performance(weights, ann_returns, cov_matrix, risk_free_rate=0.0)
    
    # Equal weight return is average of the two annualized returns
    expected_ret = 0.5 * ann_returns["A"] + 0.5 * ann_returns["B"]
    assert np.isclose(metrics.expected_return, expected_ret)

def test_calculate_max_drawdown():
    # Price goes 100 -> 120 -> 90 -> 100
    # Returns:
    # 120/100 - 1 = 0.2
    # 90/120 - 1 = -0.25
    # 100/90 - 1 = 0.111
    # Max peak is 120. Drop to 90 is 30/120 = 25% drawdown.
    returns = pd.Series([0.2, -0.25, 0.11111111])
    mdd = calculate_max_drawdown(returns)
    assert np.isclose(mdd, -0.25)

def test_maximize_sharpe_ratio():
    # Asset A: 10% return, 15% vol
    # Asset B: 5% return, 10% vol
    ann_returns = pd.Series({"A": 0.10, "B": 0.05})
    # Dummy covariance matrix
    cov_matrix = pd.DataFrame({
        "A": [0.0225, 0.005],
        "B": [0.005, 0.01]
    }, index=["A", "B"])
    
    result = maximize_sharpe_ratio(ann_returns, cov_matrix, risk_free_rate=0.02)
    
    assert result.success is True
    weights = result.weights
    assert "A" in weights and "B" in weights
    
    # Check constraints
    assert np.isclose(sum(weights.values()), 1.0)
    for w in weights.values():
        assert w >= -1e-7 # Allow tiny floating point neg
    assert weights["A"] > weights["B"]

def test_generate_efficient_frontier():
    ann_returns = pd.Series({"A": 0.10, "B": 0.05})
    cov_matrix = pd.DataFrame({
        "A": [0.0225, 0.005],
        "B": [0.005, 0.01]
    }, index=["A", "B"])
    
    frontier = generate_efficient_frontier(ann_returns, cov_matrix, num_points=10)
    
    # Frontier should be a dataframe with 10 rows
    assert isinstance(frontier, pd.DataFrame)
    assert len(frontier) == 10
    
    # Should contain specific columns
    assert "return" in frontier.columns
    assert "volatility" in frontier.columns
    assert "sharpe_ratio" in frontier.columns
    assert "A" in frontier.columns
    assert "B" in frontier.columns
    for _, row in frontier.iterrows():
        assert np.isclose(row["A"] + row["B"], 1.0)
