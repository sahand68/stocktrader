"""
Strategies for walk-forward testing.

Every strategy exposes:
    grid()                 -> list of parameter dicts it chooses between
    fit(train, cfg)        -> params, chosen using the training window only
    signal(history, params)-> target position per bar, using data up to that bar only
"""
from itertools import product

import numpy as np
import pandas as pd

from backtest.engine import Config, backtest
from backtest.metrics import sharpe_ratio
from utils import add_technical_indicators


class GridStrategy:
    name = ""

    def __init__(self, allow_short: bool = False):
        self.allow_short = allow_short

    def grid(self) -> list[dict]:
        raise NotImplementedError

    def signal(self, history: pd.DataFrame, params: dict) -> pd.Series:
        raise NotImplementedError

    def fit(self, train: pd.DataFrame, cfg: Config) -> dict:
        """Pick the grid point with the best after-cost Sharpe on the training window."""
        best, best_sharpe = None, -np.inf
        for params in self.grid():
            bt = backtest(train["Close"], self.signal(train, params), cfg)
            s = sharpe_ratio(bt["net"], cfg.periods_per_year)
            if s > best_sharpe:
                best, best_sharpe = params, s
        return best

    def _floor(self) -> float:
        return -1.0 if self.allow_short else 0.0


class SmaCross(GridStrategy):
    name = "SMA crossover"

    def __init__(self, fast=(10, 20, 50), slow=(100, 200), allow_short: bool = False):
        super().__init__(allow_short)
        self.fast, self.slow = fast, slow

    def grid(self) -> list[dict]:
        return [{"fast": f, "slow": s} for f, s in product(self.fast, self.slow) if f < s]

    def signal(self, history: pd.DataFrame, params: dict) -> pd.Series:
        close = history["Close"]
        fast = close.rolling(params["fast"]).mean()
        slow = close.rolling(params["slow"]).mean()
        sig = pd.Series(np.where(fast > slow, 1.0, self._floor()), index=close.index)
        return sig.where(slow.notna(), 0.0)


class RsiReversion(GridStrategy):
    name = "RSI mean reversion"

    def __init__(self, lower=(20, 25, 30, 35), upper=(50, 60, 70), period: int = 14, allow_short: bool = False):
        super().__init__(allow_short)
        self.lower, self.upper, self.period = lower, upper, period

    def grid(self) -> list[dict]:
        return [{"lower": lo, "upper": hi} for lo, hi in product(self.lower, self.upper)]

    def signal(self, history: pd.DataFrame, params: dict) -> pd.Series:
        delta = history["Close"].diff()
        gain = delta.clip(lower=0).rolling(self.period).mean()
        loss = (-delta.clip(upper=0)).rolling(self.period).mean()
        rsi = 100 - 100 / (1 + gain / loss)

        long = self._hold(rsi < params["lower"], rsi > params["upper"])
        if not self.allow_short:
            return long
        # Mirror image for shorts: enter above 100-lower, exit below 100-upper.
        short = self._hold(rsi > 100 - params["lower"], rsi < 100 - params["upper"])
        return long - short

    @staticmethod
    def _hold(enter: pd.Series, exit_: pd.Series) -> pd.Series:
        """1 from an entry bar until the next exit bar, else 0."""
        state = pd.Series(np.nan, index=enter.index)
        state[exit_] = 0.0
        state[enter] = 1.0
        return state.ffill().fillna(0.0)


class XGBoostDirection(GridStrategy):
    """
    Gradient-boosted forecast of the next bar's return from trailing indicators.
    Goes long only when the forecast beats the round-trip cost of trading.
    """
    name = "XGBoost forecast"

    FEATURES = [
        "RSI", "%K", "Williams_R", "CMF", "ADX", "BB_width",
        "macd_rel", "atr_rel", "sma20_gap", "sma50_gap", "ret_1", "ret_5", "ret_20",
    ]

    def __init__(self, n_estimators: int = 200, max_depth: int = 3, learning_rate: float = 0.05,
                 allow_short: bool = False):
        super().__init__(allow_short)
        self.model_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": 0.8,
            "random_state": 42,
        }

    def grid(self) -> list[dict]:
        return [self.model_params]

    @classmethod
    def features(cls, history: pd.DataFrame) -> pd.DataFrame:
        f = add_technical_indicators(history[["Open", "High", "Low", "Close", "Volume"]].copy())
        close = f["Close"]
        f["macd_rel"] = f["MACD"] / close
        f["atr_rel"] = f["ATR"] / close
        f["sma20_gap"] = close / f["SMA_20"] - 1
        f["sma50_gap"] = close / f["SMA_50"] - 1
        for n in (1, 5, 20):
            f[f"ret_{n}"] = close.pct_change(n)
        return f[cls.FEATURES].replace([np.inf, -np.inf], np.nan)

    def fit(self, train: pd.DataFrame, cfg: Config) -> dict:
        import xgboost as xgb

        x = self.features(train)
        y = train["Close"].pct_change().shift(-1)
        # The last bar's target is the first return of the test window: drop it.
        x, y = x.iloc[:-1], y.iloc[:-1]
        keep = y.notna()
        model = xgb.XGBRegressor(objective="reg:squarederror", **self.model_params)
        model.fit(x[keep], y[keep])
        return {"model": model, "hurdle": 2 * (cfg.fee_bps + cfg.slippage_bps) / 1e4}

    def signal(self, history: pd.DataFrame, params: dict) -> pd.Series:
        forecast = params["model"].predict(self.features(history))
        sig = np.where(forecast > params["hurdle"], 1.0, 0.0)
        if self.allow_short:
            sig = np.where(forecast < -params["hurdle"], -1.0, sig)
        return pd.Series(sig, index=history.index)


STRATEGIES = {cls.name: cls for cls in (SmaCross, RsiReversion, XGBoostDirection)}
