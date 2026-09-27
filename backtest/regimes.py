import numpy as np
import pandas as pd

from backtest.metrics import metrics


def label_regimes(close: pd.Series, window: int = 200, slope_bars: int = 20) -> pd.Series:
    """
    Bull: price above a rising moving average. Bear: below a falling one. Chop: anything else.
    Uses only trailing data, so the label at bar t is known at the close of bar t.
    """
    ma = close.rolling(window).mean()
    slope = ma.diff(slope_bars)
    labels = np.select(
        [(close > ma) & (slope > 0), (close < ma) & (slope < 0), ma.notna() & slope.notna()],
        ["bull", "bear", "chop"],
        default="warmup",
    )
    return pd.Series(labels, index=close.index)


def metrics_by_regime(
    net: pd.Series,
    close: pd.Series,
    periods_per_year: float,
    window: int = 200,
    min_obs: int = 30,
) -> pd.DataFrame:
    """Strategy metrics split by the regime in force when each position was taken."""
    # Return at bar t comes from the position chosen at the close of t-1.
    regime = label_regimes(close, window).shift(1).reindex(net.index)
    rows = []
    for name in ("bull", "bear", "chop"):
        r = net[regime == name]
        if len(r) < min_obs:
            rows.append({"regime": name, "n_obs": len(r)})
            continue
        rows.append({"regime": name, **metrics(r, periods_per_year, min_obs)})
    return pd.DataFrame(rows).set_index("regime")
