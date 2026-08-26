import numpy as np
import pandas as pd
from typing import Optional

from src.quant.metrics import (
    calculate_portfolio_performance,
    calculate_annualized_returns,
    calculate_covariance_matrix
)
from src.config import DEFAULT_RISK_FREE_RATE
from src.quant.models import PortfolioResult, BenchmarkResult, PortfolioMetrics


def generate_equal_weights(tickers: list[str]) -> np.ndarray:
    """Generates equal weights for a list of tickers."""
    n = len(tickers)

    if not tickers:
        raise ValueError("Ticker list cannot be empty.")

    return np.ones(n) / n


def evaluate_portfolio(
    weights_dict: dict[str, float],
    daily_returns: pd.DataFrame,
    risk_free_rate: float
) -> PortfolioMetrics:
    """Evaluates a portfolio with fixed weights over a given period."""
    tickers = list(daily_returns.columns)
    weights_arr = np.array([weights_dict.get(t, 0.0) for t in tickers])

    ann_returns = calculate_annualized_returns(daily_returns)
    cov_matrix = calculate_covariance_matrix(daily_returns)

    return calculate_portfolio_performance(
        weights_arr,
        ann_returns,
        cov_matrix,
        daily_returns,
        risk_free_rate
    )


def compare_portfolios(
    optimal_result: PortfolioResult,
    train_returns: pd.DataFrame,
    test_returns: pd.DataFrame,
    market_train_returns: Optional[pd.Series] = None,
    market_test_returns: Optional[pd.Series] = None,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE
) -> BenchmarkResult:
    """
    Compares the optimal portfolio against a naive equal-weight portfolio
    and market benchmark, evaluating both in-sample and out-of-sample performance.
    """

    tickers = list(train_returns.columns)

    # --- Equal Weight Portfolio ---
    eq_weights_arr = generate_equal_weights(tickers)
    eq_weights_dict = dict(zip(tickers, eq_weights_arr))

    eq_train_metrics = evaluate_portfolio(
        eq_weights_dict,
        train_returns,
        risk_free_rate
    )

    eq_test_metrics = evaluate_portfolio(
        eq_weights_dict,
        test_returns,
        risk_free_rate
    )

    equal_result = PortfolioResult(
        weights=eq_weights_dict,
        in_sample_metrics=eq_train_metrics,
        out_of_sample_metrics=eq_test_metrics,
        success=True
    )

    # --- Market Benchmark ---
    market_result = None
    market_train_metrics = None
    market_test_metrics = None

    if market_train_returns is not None and market_test_returns is not None:
        market_train_df = market_train_returns.to_frame(name="Market")
        market_test_df = market_test_returns.to_frame(name="Market")

        market_train_metrics = evaluate_portfolio(
            {"Market": 1.0},
            market_train_df,
            risk_free_rate
        )

        market_test_metrics = evaluate_portfolio(
            {"Market": 1.0},
            market_test_df,
            risk_free_rate
        )

        market_result = PortfolioResult(
            weights={"Market": 1.0},
            in_sample_metrics=market_train_metrics,
            out_of_sample_metrics=market_test_metrics,
            success=True
        )

    # --- Optimal Portfolio Metrics ---
    optimal_metrics_dict = {
        "IS Return": (
            optimal_result.in_sample_metrics.expected_return
            if optimal_result.in_sample_metrics else 0.0
        ),
        "OOS Return": (
            optimal_result.out_of_sample_metrics.expected_return
            if optimal_result.out_of_sample_metrics else 0.0
        ),
        "IS Volatility": (
            optimal_result.in_sample_metrics.volatility
            if optimal_result.in_sample_metrics else 0.0
        ),
        "OOS Volatility": (
            optimal_result.out_of_sample_metrics.volatility
            if optimal_result.out_of_sample_metrics else 0.0
        ),
        "IS Sharpe": (
            optimal_result.in_sample_metrics.sharpe_ratio
            if optimal_result.in_sample_metrics else 0.0
        ),
        "OOS Sharpe": (
            optimal_result.out_of_sample_metrics.sharpe_ratio
            if optimal_result.out_of_sample_metrics else 0.0
        ),
        "IS Max Drawdown": (
            optimal_result.in_sample_metrics.max_drawdown
            if optimal_result.in_sample_metrics else 0.0
        ),
        "OOS Max Drawdown": (
            optimal_result.out_of_sample_metrics.max_drawdown
            if optimal_result.out_of_sample_metrics else 0.0
        ),
    }

    # --- Equal Weight Metrics ---
    equal_metrics_dict = {
        "IS Return": eq_train_metrics.expected_return,
        "OOS Return": eq_test_metrics.expected_return,
        "IS Volatility": eq_train_metrics.volatility,
        "OOS Volatility": eq_test_metrics.volatility,
        "IS Sharpe": eq_train_metrics.sharpe_ratio,
        "OOS Sharpe": eq_test_metrics.sharpe_ratio,
        "IS Max Drawdown": eq_train_metrics.max_drawdown,
        "OOS Max Drawdown": eq_test_metrics.max_drawdown,
    }

    data = {
        "Optimal (Max Sharpe)": optimal_metrics_dict,
        "Naive (Equal Weight)": equal_metrics_dict
    }

    # --- Market Metrics ---
    if market_train_metrics is not None and market_test_metrics is not None:
        market_metrics_dict = {
            "IS Return": market_train_metrics.expected_return,
            "OOS Return": market_test_metrics.expected_return,
            "IS Volatility": market_train_metrics.volatility,
            "OOS Volatility": market_test_metrics.volatility,
            "IS Sharpe": market_train_metrics.sharpe_ratio,
            "OOS Sharpe": market_test_metrics.sharpe_ratio,
            "IS Max Drawdown": market_train_metrics.max_drawdown,
            "OOS Max Drawdown": market_test_metrics.max_drawdown,
        }

        data["Market (NIFTY 50)"] = market_metrics_dict

    comparison_df = pd.DataFrame(data)

    return BenchmarkResult(
        optimized=optimal_result,
        equal_weight=equal_result,
        market_benchmark=market_result,
        comparison_df=comparison_df
    )