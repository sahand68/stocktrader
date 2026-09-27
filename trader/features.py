"""
Scale-free features computed from trailing bars only: row t uses bars up to and including t.

Nothing here carries a price level, timestamp or symbol, so a model that saw market
history during training cannot recognize which episode it is looking at.
"""
import numpy as np
import pandas as pd

RECENT_BARS = 10


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return 100 - 100 / (1 + gain / loss)


def build_features(bars: pd.DataFrame) -> pd.DataFrame:
    close, volume = bars["close"], bars["volume"]
    ret = close.pct_change()
    vol20 = ret.rolling(20).std()

    f = pd.DataFrame(index=bars.index)
    f["ret_1"] = ret
    f["ret_5"] = close.pct_change(5)
    f["ret_20"] = close.pct_change(20)
    f["vol_20"] = vol20
    f["vol_ratio"] = ret.rolling(5).std() / ret.rolling(50).std()
    # Distance from each moving average in units of per-bar volatility.
    for n in (20, 50, 200):
        f[f"z_sma{n}"] = (close / close.rolling(n).mean() - 1) / vol20
    f["rsi_14"] = rsi(close)
    low, high = bars["low"].rolling(20).min(), bars["high"].rolling(20).max()
    f["range_pos_20"] = (close - low) / (high - low)
    log_vol = np.log(volume.where(volume > 0))
    f["volume_z_20"] = (log_vol - log_vol.rolling(20).mean()) / log_vol.rolling(20).std()
    for k in range(RECENT_BARS):
        f[f"ret_lag{k}"] = ret.shift(k)
    return f.replace([np.inf, -np.inf], np.nan)


def ready(features: pd.DataFrame) -> pd.Series:
    """Bars with every feature defined (after the 200-bar warm-up)."""
    return features.notna().all(axis=1)
