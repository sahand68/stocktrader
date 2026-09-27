"""
Mechanical versions of what a hostile reviewer checks on every backtest.
Any FAIL blocks the strategy.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from trader.backtest import Costs
from trader.data import bar_minutes

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


def lookahead_probe(bars: pd.DataFrame, strategy, params: dict, n_probes: int = 5) -> Check:
    """
    Recompute inputs and positions on truncated history and compare them with the
    full-history run. Removing future bars cannot change a causal signal, so any
    difference is look-ahead: centered windows, backfilling, full-sample scaling.
    """
    full_inputs = strategy.inputs(bars)
    full = strategy.positions(full_inputs, params)
    numeric = full_inputs.select_dtypes("number").columns
    cuts = np.linspace(len(bars) // 2, len(bars) - 2, n_probes).astype(int)
    for cut in cuts:
        part_inputs = strategy.inputs(bars.iloc[: cut + 1])
        part = strategy.positions(part_inputs, params)
        same_inputs = np.isclose(part_inputs[numeric].to_numpy(dtype=float),
                                 full_inputs[numeric].iloc[: cut + 1].to_numpy(dtype=float), equal_nan=True)
        same_positions = np.isclose(part.to_numpy(), full.iloc[: cut + 1].to_numpy())
        if not (same_inputs.all() and same_positions.all()):
            bad = ~same_positions if not same_positions.all() else ~same_inputs.all(axis=1)
            ts = bars.index[int(np.argmax(bad))]
            return Check("Look-ahead", FAIL, f"the signal at {ts} changes when bars after {bars.index[cut]} are removed")
    return Check("Look-ahead", PASS, f"inputs and positions unchanged on {n_probes} truncated histories")


def anonymity_check(bars: pd.DataFrame, strategy) -> Check:
    """
    A language model trained on market history may recognize an episode from a date
    or a price level and "predict" what it remembers. So the state sent to it must
    not change when every price is doubled or every date moves ten years.
    (Doubling is exact in floating point, so any difference is real.)
    """
    if not hasattr(strategy, "states"):
        return Check("Anonymized state", PASS, "strategy sends nothing to a model")
    _, base = strategy.states(bars)

    rescaled = bars.copy()
    rescaled[["open", "high", "low", "close"]] *= 2
    if strategy.states(rescaled)[1] != base:
        return Check("Anonymized state", FAIL, "state changes with the price level")

    shifted = bars.copy()
    shifted.index = shifted.index - pd.Timedelta(days=3653)
    if strategy.states(shifted)[1] != base:
        return Check("Anonymized state", FAIL, "state changes with the date")

    return Check("Anonymized state", PASS, f"{len(base)} states identical under 2x prices and a 10-year date shift")


def cost_check(costs: Costs) -> Check:
    if costs.fee_bps <= 0 or costs.slippage_bps <= 0:
        return Check("Costs", FAIL, "fees and slippage must both be charged")
    per_side = costs.fee_bps + costs.slippage_bps
    if per_side < 10:
        return Check("Costs", WARN, f"{per_side:g} bps per side is below typical spot taker costs")
    return Check("Costs", PASS, f"{per_side:g} bps per side on every change in position")


def data_check(bars: pd.DataFrame, interval: str) -> Check:
    idx = bars.index
    if not isinstance(idx, pd.DatetimeIndex) or str(idx.tz) != "UTC":
        return Check("Data alignment", FAIL, "index must be a UTC DatetimeIndex")
    if not idx.is_monotonic_increasing or idx.has_duplicates:
        return Check("Data alignment", FAIL, "timestamps out of order or duplicated")
    step = pd.Timedelta(minutes=bar_minutes(interval))
    missing = int(((idx.to_series().diff().dropna() / step).round() - 1).clip(lower=0).sum())
    if missing:
        return Check("Data alignment", WARN, f"{missing} missing {interval} bars; returns span the gaps")
    return Check("Data alignment", PASS, f"{len(idx)} consecutive UTC {interval} bars")


def label_regimes(close: pd.Series, window: int = 200, slope_bars: int = 20) -> pd.Series:
    """Bull: above a rising moving average. Bear: below a falling one. Chop: the rest."""
    ma = close.rolling(window).mean()
    slope = ma.diff(slope_bars)
    labels = np.select(
        [(close > ma) & (slope > 0), (close < ma) & (slope < 0), slope.notna()],
        ["bull", "bear", "chop"],
        default="warmup",
    )
    return pd.Series(labels, index=close.index)


def regime_check(close: pd.Series, test_index: pd.Index) -> Check:
    share = label_regimes(close).reindex(test_index).value_counts(normalize=True)
    summary = ", ".join(f"{r} {share.get(r, 0):.0%}" for r in ("bull", "bear", "chop"))
    if share.get("bull", 0) < 0.05 or share.get("bear", 0) < 0.05:
        return Check("Sample regimes", WARN, f"test period lacks a bull or a bear phase ({summary})")
    return Check("Sample regimes", PASS, summary)


def sharpe_check(sharpe: float) -> Check:
    if sharpe > 3:
        return Check("Sharpe sanity", FAIL, f"out-of-sample Sharpe {sharpe:.2f} > 3: assume leakage")
    if sharpe > 2:
        return Check("Sharpe sanity", WARN, f"out-of-sample Sharpe {sharpe:.2f} > 2: look for leakage first")
    return Check("Sharpe sanity", PASS, f"out-of-sample Sharpe {sharpe:.2f}")


def model_version_check(inputs: pd.DataFrame, test_index: pd.Index) -> Check:
    if "model" not in inputs:
        return Check("Model version", PASS, "no model involved")
    versions = sorted(inputs["model"].reindex(test_index).dropna().unique())
    if len(versions) > 1:
        return Check("Model version", WARN,
                     f"answers come from {', '.join(versions)}; pin one version so thresholds stay calibrated")
    return Check("Model version", PASS, versions[0] if versions else "none")


def selection_note(symbol: str) -> Check:
    return Check("Survivorship", WARN,
                 f"{symbol} still trades; coins that died are not in this single-asset sample")
