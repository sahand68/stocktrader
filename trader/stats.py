from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

EULER_GAMMA = 0.5772156649


@dataclass(frozen=True)
class DeflatedSharpe:
    probability: float       # P(true Sharpe beats the best-of-noise benchmark)
    sharpe: float            # annualized
    noise_benchmark: float   # annualized Sharpe the best of n_trials noise strategies would show
    n_trials: int

    @property
    def passed(self) -> bool:
        return self.probability > 0.95


def expected_max_sharpe(n_trials: int, sharpe_std: float) -> float:
    """Expected best Sharpe among n_trials strategies with no edge."""
    if n_trials < 2:
        return 0.0
    z = (1 - EULER_GAMMA) * norm.ppf(1 - 1 / n_trials) + EULER_GAMMA * norm.ppf(1 - 1 / (n_trials * np.e))
    return sharpe_std * z


def deflated_sharpe(returns: pd.Series, n_trials: int, periods_per_year: float) -> DeflatedSharpe:
    """
    Bailey & Lopez de Prado (2014), computed in per-bar units throughout and only
    annualized for display. Mixing an annualized Sharpe with a per-bar sample size
    turns the test into a fixed Sharpe cutoff.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    r = returns.dropna().to_numpy()
    t = len(r)
    if t < 3:
        raise ValueError("need at least 3 returns")
    std = r.std(ddof=1)
    sr = r.mean() / std if std > 0 else 0.0
    # Spread of Sharpe estimates across trials; the null sampling error when unknown.
    sr0 = expected_max_sharpe(n_trials, 1 / np.sqrt(t - 1))
    variance = 1 - skew(r) * sr + (kurtosis(r, fisher=False) - 1) / 4 * sr**2
    probability = float(norm.cdf((sr - sr0) * np.sqrt(t - 1) / np.sqrt(max(variance, 1e-12))))
    k = np.sqrt(periods_per_year)
    return DeflatedSharpe(probability, float(sr * k), float(sr0 * k), n_trials)
