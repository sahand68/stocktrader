import numpy as np
import pandas as pd
import pytest


def make_ohlcv(n: int = 3000, drift: float = 0.0, vol: float = 0.01, seed: int = 0) -> pd.DataFrame:
    """Hourly geometric random walk with UTC bar-open timestamps."""
    rng = np.random.default_rng(seed)
    close = 20_000 * np.exp(np.cumsum(drift + vol * rng.standard_normal(n)))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(vol * rng.standard_normal(n)) * close
    idx = pd.date_range("2024-01-01", periods=n, freq="60min", tz="UTC")
    return pd.DataFrame({
        "Open": open_,
        "High": np.maximum(open_, close) + spread,
        "Low": np.minimum(open_, close) - spread,
        "Close": close,
        "Volume": rng.uniform(100, 1000, n),
    }, index=idx)


@pytest.fixture
def ohlcv():
    return make_ohlcv()
