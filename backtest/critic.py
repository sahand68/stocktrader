"""
Mechanical versions of the checks a skeptical reviewer runs on every backtest.
Each returns a Check; anything FAIL blocks the strategy.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.engine import Config
from backtest.regimes import label_regimes
from utils import TIMEFRAME_MINUTES

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


def lookahead_probe(df: pd.DataFrame, strategy, params: dict, n_probes: int = 8) -> Check:
    """
    Recompute the signal on truncated history and compare it with the signal from
    the full history. A causal signal cannot change when future bars are removed;
    any difference is look-ahead or repainting (centered windows, bfill, full-sample scaling).
    """
    full = strategy.signal(df, params)
    cuts = np.linspace(len(df) // 2, len(df) - 2, n_probes).astype(int)
    for cut in cuts:
        partial = strategy.signal(df.iloc[: cut + 1], params)
        expected = full.iloc[: cut + 1]
        diff = ~np.isclose(partial.to_numpy(), expected.to_numpy(), equal_nan=True)
        if diff.any():
            ts = expected.index[int(np.argmax(diff))]
            return Check("Look-ahead", FAIL,
                         f"signal at {ts} changes when data after {df.index[cut]} is removed")
    return Check("Look-ahead", PASS, f"signal unchanged on {n_probes} truncated histories")


def cost_check(cfg: Config) -> Check:
    per_side = cfg.fee_bps + cfg.slippage_bps
    if cfg.fee_bps <= 0 or cfg.slippage_bps <= 0:
        return Check("Costs", FAIL, f"fee {cfg.fee_bps} bps, slippage {cfg.slippage_bps} bps: both must be charged")
    if per_side < 10:
        return Check("Costs", WARN, f"{per_side:g} bps per side is below typical retail spot taker costs")
    return Check("Costs", PASS, f"{per_side:g} bps per side on every change in position")


def data_check(df: pd.DataFrame, interval: str) -> Check:
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex) or idx.tz is None or str(idx.tz) != "UTC":
        return Check("Data alignment", FAIL, "index must be a UTC DatetimeIndex")
    if not idx.is_monotonic_increasing or idx.has_duplicates:
        return Check("Data alignment", FAIL, "timestamps are out of order or duplicated")
    step = pd.Timedelta(minutes=TIMEFRAME_MINUTES[interval])
    gaps = idx.to_series().diff().dropna()
    missing = int(((gaps / step).round() - 1).clip(lower=0).sum())
    if missing:
        return Check("Data alignment", WARN,
                     f"{missing} missing {interval} bars (exchange outages); returns span the gaps")
    return Check("Data alignment", PASS, f"UTC, bar-open timestamps, no gaps in {len(idx)} bars")


def regime_check(close: pd.Series, test_index: pd.DatetimeIndex, window: int = 200) -> Check:
    labels = label_regimes(close, window).reindex(test_index)
    share = labels.value_counts(normalize=True)
    present = [r for r in ("bull", "bear") if share.get(r, 0) >= 0.05]
    summary = ", ".join(f"{r} {share.get(r, 0):.0%}" for r in ("bull", "bear", "chop"))
    if len(present) < 2:
        return Check("Sample regimes", WARN, f"test period lacks a bull or bear phase ({summary})")
    return Check("Sample regimes", PASS, summary)


def sharpe_check(sharpe: float) -> Check:
    """Very high Sharpe on liquid crypto is far more often leakage than edge."""
    if sharpe > 3:
        return Check("Sharpe sanity", FAIL, f"out-of-sample Sharpe {sharpe:.2f} > 3: assume leakage")
    if sharpe > 2:
        return Check("Sharpe sanity", WARN, f"out-of-sample Sharpe {sharpe:.2f} > 2: look for leakage first")
    return Check("Sharpe sanity", PASS, f"out-of-sample Sharpe {sharpe:.2f}")


def selection_note() -> Check:
    return Check("Survivorship", WARN,
                 "single-asset test on a coin that still trades; coins that died are not in the sample")
