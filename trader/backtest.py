from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Costs:
    fee_bps: float = 10.0       # per side; spot taker fee on most major exchanges
    slippage_bps: float = 5.0   # per side

    @property
    def per_side(self) -> float:
        return (self.fee_bps + self.slippage_bps) / 1e4


def run(close: pd.Series, target: pd.Series, costs: Costs) -> pd.DataFrame:
    """
    close: bar closes
    target: position wanted at each bar's close, decided from data up to that close

    A decision made at the close of bar t is held over bar t+1.
    """
    if not close.index.equals(target.index):
        raise ValueError("close and target must share an index")

    position = target.shift(1).fillna(0.0).clip(-1, 1)
    # Simple returns: position times log return misstates short P&L on big moves.
    gross = position * close.pct_change().fillna(0.0)
    turnover = position.diff().fillna(position).abs()
    cost = turnover * costs.per_side
    net = gross - cost
    equity = (1 + net).clip(lower=0).cumprod()
    return pd.DataFrame({"position": position, "gross": gross, "cost": cost, "net": net, "equity": equity})


def sharpe(net: pd.Series, periods_per_year: float) -> float:
    std = net.std()
    return float(net.mean() / std * np.sqrt(periods_per_year)) if std > 0 else 0.0


def metrics(net: pd.Series, periods_per_year: float) -> dict:
    r = net.dropna()
    if len(r) < 2:
        raise ValueError("need at least two returns")
    equity = (1 + r).clip(lower=0).cumprod()
    drawdown = equity / equity.cummax() - 1
    underwater = drawdown < 0
    spells = underwater.groupby((underwater != underwater.shift()).cumsum()).sum()
    years = len(r) / periods_per_year
    final = float(equity.iloc[-1])
    cagr = final ** (1 / years) - 1 if final > 0 else -1.0
    max_dd = float(drawdown.min())
    return {
        "sharpe": sharpe(r, periods_per_year),
        "cagr": cagr,
        "total_return": final - 1,
        "max_drawdown": max_dd,
        "longest_drawdown_days": float(spells.max()) * 365 / periods_per_year,
        "calmar": cagr / -max_dd if max_dd < 0 else float("nan"),
        "bars": len(r),
    }
