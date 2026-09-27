from dataclasses import dataclass

import pandas as pd

from trader import backtest
from trader.backtest import Costs

MIN_TEST_BARS = 30


@dataclass
class WalkForward:
    folds: pd.DataFrame     # one row per test window
    oos: pd.DataFrame       # backtest of the stitched out-of-sample positions
    live_params: dict       # fitted on the most recent training window, for paper trading

    @property
    def positive_folds(self) -> int:
        return int((self.folds["sharpe"] > 0).sum())


def fit(strategy, inputs: pd.DataFrame, close: pd.Series, costs: Costs, periods_per_year: float) -> dict:
    """The grid point with the best after-cost Sharpe on this window."""
    def score(params: dict) -> float:
        bt = backtest.run(close, strategy.positions(inputs, params), costs)
        return backtest.sharpe(bt["net"], periods_per_year)

    return max(strategy.grid(), key=score)


def walk_forward(
    bars: pd.DataFrame,
    strategy,
    costs: Costs,
    periods_per_year: float,
    train_bars: int,
    test_bars: int,
    inputs: pd.DataFrame | None = None,
) -> WalkForward:
    """
    Fit on a trailing window, trade the next one, roll forward by one window.
    `inputs` must be causal (checked by critic.lookahead_probe), so computing them
    once over all bars and slicing is the same as recomputing them fold by fold.
    """
    if test_bars < MIN_TEST_BARS:
        raise ValueError(f"test window of {test_bars} bars is below the {MIN_TEST_BARS}-bar minimum")
    n_folds = (len(bars) - train_bars) // test_bars
    if n_folds < 2:
        raise ValueError(
            f"{len(bars)} bars fit {max(n_folds, 0)} fold(s) of {train_bars} train + {test_bars} test bars; "
            "use more days or shorter windows"
        )

    inputs = strategy.inputs(bars) if inputs is None else inputs
    close = bars["close"]
    targets, rows = [], []
    for k in range(n_folds):
        train = slice(k * test_bars, k * test_bars + train_bars)
        test = slice(train.stop, train.stop + test_bars)
        params = fit(strategy, inputs.iloc[train], close.iloc[train], costs, periods_per_year)
        targets.append(strategy.positions(inputs.iloc[: test.stop], params).iloc[test])
        rows.append({"start": bars.index[test.start], "end": bars.index[test.stop - 1], **params})

    target = pd.concat(targets)
    oos = backtest.run(close.loc[target.index], target, costs)
    for row in rows:
        row.update(backtest.metrics(oos["net"].loc[row["start"] : row["end"]], periods_per_year))

    recent = slice(len(bars) - train_bars, len(bars))
    live_params = fit(strategy, inputs.iloc[recent], close.iloc[recent], costs, periods_per_year)
    return WalkForward(folds=pd.DataFrame(rows), oos=oos, live_params=live_params)
