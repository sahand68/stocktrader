import numpy as np
import pandas as pd


def sharpe_ratio(net: pd.Series, periods_per_year: float) -> float:
    """Annualized Sharpe; 0 for a strategy that never moves."""
    std = net.std()
    if not std > 0:
        return 0.0
    return float(net.mean() / std * np.sqrt(periods_per_year))


def metrics(net: pd.Series, periods_per_year: float, min_obs: int = 30) -> dict:
    """Performance of a series of per-bar net simple returns."""
    r = net.dropna()
    if len(r) < min_obs:
        raise ValueError(f"need at least {min_obs} observations, got {len(r)}")

    equity = (1 + r).clip(lower=0).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1
    max_dd = float(dd.min())

    # How long you sat underwater: the number that decides whether you'd have held on.
    underwater = dd < 0
    spell_id = (underwater != underwater.shift()).cumsum()
    longest_dd_bars = int(underwater.groupby(spell_id).sum().max())

    years = len(r) / periods_per_year
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if equity.iloc[-1] > 0 else -1.0

    return {
        "sharpe": sharpe_ratio(r, periods_per_year),
        "cagr": cagr,
        "ann_vol": float(r.std() * np.sqrt(periods_per_year)),
        "max_drawdown": max_dd,
        "longest_dd_days": longest_dd_bars * 365 / periods_per_year,
        "calmar": cagr / abs(max_dd) if max_dd < 0 else float("nan"),
        "n_obs": len(r),
    }
