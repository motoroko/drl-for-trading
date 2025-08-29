# utils/metrics.py

import numpy as np
from scipy.stats import kurtosis

def compute_deflated_sharpe_ratio(returns, benchmark_sr=0.0, num_strategies=4):
    """
    Compute the Deflated Sharpe Ratio (DSR).
    
    Args:
        returns (np.array): Array of daily returns
        benchmark_sr (float): Threshold Sharpe Ratio (e.g., 0 or from index)
        num_strategies (int): Number of strategies compared (to model selection effect)
    
    Returns:
        float: Deflated Sharpe Ratio
    """
    n = len(returns)
    if n < 2:
        return 0.0

    mean_r = np.mean(returns)
    std_r = np.std(returns)
    sr = mean_r / (std_r + 1e-8)
    
    k = kurtosis(returns, fisher=True)
    gamma = 0.5772  # Euler–Mascheroni constant

    sr_adj = sr - benchmark_sr
    denominator = np.sqrt((1 - gamma) / n * (1 + (sr ** 2) / 2 + k / 24))
    
    return sr_adj / (denominator + 1e-8)
