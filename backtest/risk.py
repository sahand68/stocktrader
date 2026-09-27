import pandas as pd

from backtest.metrics import sharpe_ratio


def position_size(capital: float, entry: float, stop: float,
                  risk_pct: float = 0.01, max_position_pct: float = 0.20) -> dict:
    """
    Size a trade so hitting the stop loses risk_pct of capital,
    capped at max_position_pct of capital in notional.
    """
    risk_per_unit = abs(entry - stop)
    if risk_per_unit == 0:
        raise ValueError("stop cannot equal entry")

    units = capital * risk_pct / risk_per_unit
    cap = capital * max_position_pct
    if units * entry > cap:
        units = cap / entry
    notional = units * entry

    return {
        "units": units,
        "notional": notional,
        "pct_of_capital": notional / capital,
        "loss_if_stopped": units * risk_per_unit,
    }


def losing_streak_drawdown(risk_pct: float, streak: int) -> float:
    """Drawdown after `streak` consecutive stopped-out trades at fixed fractional risk."""
    return 1 - (1 - risk_pct) ** streak


def health_check(live_returns: pd.Series, backtest_sharpe: float, backtest_max_dd: float,
                 periods_per_year: float, window: int = 30) -> dict:
    """
    Run on every new bar of live returns. Decide the thresholds before deploying.

    backtest_max_dd: worst out-of-sample drawdown as a negative fraction, e.g. -0.25
    window: bars of recent live returns to judge Sharpe decay on
    """
    live_sharpe = sharpe_ratio(live_returns.tail(window), periods_per_year)

    equity = (1 + live_returns).cumprod()
    current_dd = float(equity.iloc[-1] / equity.cummax().iloc[-1] - 1)

    alerts = []
    if backtest_sharpe > 0 and live_sharpe < 0.5 * backtest_sharpe:
        alerts.append("SHARPE_DECAY")
    if current_dd < 1.5 * backtest_max_dd:
        alerts.append("DRAWDOWN_EXCEEDED")

    return {
        "live_sharpe": live_sharpe,
        "current_dd": current_dd,
        "alerts": alerts,
        "action": "HALT" if alerts else "CONTINUE",
    }
