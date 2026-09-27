from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

EULER_GAMMA = 0.5772156649


@dataclass(frozen=True)
class DeflatedSharpe:
    probability: float          # P(true Sharpe > best-of-noise benchmark)
    sharpe: float               # annualized, observed
    noise_benchmark: float      # annualized Sharpe the best of n_trials noise strategies would show
    n_trials: int
    passed: bool


def expected_max_sharpe(n_trials: int, sharpe_std: float) -> float:
    """Expected maximum of n_trials Sharpe estimates drawn from pure noise."""
    if n_trials < 2:
        return 0.0
    z = (1 - EULER_GAMMA) * norm.ppf(1 - 1 / n_trials) + EULER_GAMMA * norm.ppf(1 - 1 / (n_trials * np.e))
    return sharpe_std * z


def deflated_sharpe(
    returns: pd.Series,
    n_trials: int,
    periods_per_year: float,
    trial_sharpe_std: float | None = None,
    threshold: float = 0.95,
) -> DeflatedSharpe:
    """
    Bailey & Lopez de Prado (2014) deflated Sharpe ratio.

    Everything is computed per bar and only annualized for reporting: mixing an
    annualized Sharpe with a per-bar sample size inflates the result by sqrt(bars per year).

    returns: per-bar net returns of the strategy you intend to trade
    n_trials: every variation you tried, including the ones you threw away
    trial_sharpe_std: per-bar std of the Sharpe ratios across those trials.
        If unknown, the sampling std under the null, 1/sqrt(T-1), is used.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    r = returns.dropna().to_numpy()
    t = len(r)
    if t < 3:
        raise ValueError("need at least 3 returns")

    std = r.std(ddof=1)
    sr = r.mean() / std if std > 0 else 0.0
    if trial_sharpe_std is None:
        trial_sharpe_std = 1 / np.sqrt(t - 1)
    sr0 = expected_max_sharpe(n_trials, trial_sharpe_std)

    g3 = skew(r)
    g4 = kurtosis(r, fisher=False)
    variance = 1 - g3 * sr + (g4 - 1) / 4 * sr**2
    probability = float(norm.cdf((sr - sr0) * np.sqrt(t - 1) / np.sqrt(max(variance, 1e-12))))

    annualize = np.sqrt(periods_per_year)
    return DeflatedSharpe(
        probability=probability,
        sharpe=float(sr * annualize),
        noise_benchmark=float(sr0 * annualize),
        n_trials=n_trials,
        passed=probability > threshold,
    )
