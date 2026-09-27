from dataclasses import dataclass

import pandas as pd

from backtest.engine import Config, backtest
from backtest.metrics import metrics


@dataclass
class WalkForwardResult:
    folds: pd.DataFrame        # one row per test window
    oos: pd.DataFrame          # backtest of the stitched out-of-sample signal
    params: list[dict]         # parameters chosen in each fold

    @property
    def positive_folds(self) -> int:
        return int((self.folds["sharpe"] > 0).sum())

    @property
    def worst_fold_sharpe(self) -> float:
        return float(self.folds["sharpe"].min())


def walk_forward(
    df: pd.DataFrame,
    strategy,
    cfg: Config,
    train_bars: int,
    test_bars: int,
    min_obs: int = 30,
) -> WalkForwardResult:
    """
    Fit on the past, trade the next window, roll forward by one window.

    Each fold's signal is computed on all history up to the end of its test window,
    so indicators are warmed up, then only the test window is kept. The strategy
    must therefore be causal; backtest.critic.lookahead_probe checks that.
    """
    if test_bars < min_obs:
        raise ValueError(f"test window of {test_bars} bars is below the {min_obs}-bar minimum")
    n_folds = (len(df) - train_bars) // test_bars
    if n_folds < 2:
        raise ValueError(
            f"{len(df)} bars only fit {n_folds} fold(s) of {train_bars} train + {test_bars} test; "
            "fetch a longer period or shrink the windows"
        )

    signals, params_used, bounds = [], [], []
    for k in range(n_folds):
        start = k * test_bars
        test_start = start + train_bars
        test_end = test_start + test_bars

        params = strategy.fit(df.iloc[start:test_start], cfg)
        signal = strategy.signal(df.iloc[:test_end], params).iloc[test_start:test_end]

        signals.append(signal)
        params_used.append(params)
        bounds.append((df.index[test_start], df.index[test_end - 1]))

    oos_signal = pd.concat(signals)
    oos = backtest(df["Close"].loc[oos_signal.index], oos_signal, cfg)

    rows = []
    for (start, end), params in zip(bounds, params_used):
        fold_net = oos["net"].loc[start:end]
        shown = {k: v for k, v in params.items() if isinstance(v, (int, float, str))}
        rows.append({"start": start, "end": end, **metrics(fold_net, cfg.periods_per_year, min_obs), **shown})

    return WalkForwardResult(folds=pd.DataFrame(rows), oos=oos, params=params_used)
