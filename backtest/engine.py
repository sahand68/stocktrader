from dataclasses import dataclass

import pandas as pd

from utils import TIMEFRAME_MINUTES


def periods_per_year(interval: str) -> float:
    """Crypto trades 24/7/365."""
    return 365 * 1440 / TIMEFRAME_MINUTES[interval]


@dataclass(frozen=True)
class Config:
    initial_capital: float = 10_000
    fee_bps: float = 10.0        # per side; spot taker fee on most major exchanges
    slippage_bps: float = 5.0    # per side
    max_leverage: float = 1.0
    periods_per_year: float = 365 * 24  # hourly bars; use periods_per_year(interval)


def backtest(prices: pd.Series, signal: pd.Series, cfg: Config) -> pd.DataFrame:
    """
    prices: close prices, datetime index
    signal: target position in [-max_leverage, +max_leverage], computed only
            from data available at the close of that bar
    """
    if not prices.index.equals(signal.index):
        raise ValueError("prices and signal must share the same index")

    # A signal computed at the close of bar t can only be held from bar t+1.
    position = signal.shift(1).fillna(0).clip(-cfg.max_leverage, cfg.max_leverage)

    # Simple returns, not log returns: position * log return misstates P&L
    # for shorts and leverage on the large moves crypto routinely makes.
    returns = prices.pct_change().fillna(0)
    gross = position * returns

    turnover = position.diff().abs().fillna(position.abs())
    costs = turnover * (cfg.fee_bps + cfg.slippage_bps) / 1e4

    net = gross - costs
    # A bar that loses 100% or more wipes the account; it cannot go negative.
    growth = (1 + net).clip(lower=0).cumprod()
    equity = cfg.initial_capital * growth

    return pd.DataFrame({
        "position": position,
        "gross": gross,
        "costs": costs,
        "net": net,
        "equity": equity,
    })
