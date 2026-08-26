from dataclasses import dataclass
from typing import Optional
import pandas as pd

@dataclass(frozen=True)
class PortfolioMetrics:
    """Metrics representing the performance of a portfolio."""
    expected_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float

@dataclass(frozen=True)
class PortfolioResult:
    """The result of a portfolio allocation (either optimized or naive)."""
    weights: dict[str, float]
    in_sample_metrics: Optional[PortfolioMetrics] = None
    out_of_sample_metrics: Optional[PortfolioMetrics] = None
    success: bool = True
    error_message: Optional[str] = None

@dataclass(frozen=True)
class BenchmarkResult:
    """Comparison between an optimized and a baseline equal-weight portfolio and a market benchmark."""
    optimized: PortfolioResult
    equal_weight: PortfolioResult
    market_benchmark: Optional[PortfolioResult]
    comparison_df: pd.DataFrame
