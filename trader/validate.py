"""
The three gates a strategy clears before it may paper trade:
  1. the critic finds no leakage
  2. the deflated Sharpe clears the multiple-testing bar
  3. out-of-sample performance survives walk-forward
"""
from dataclasses import asdict, dataclass

import pandas as pd

from trader import backtest, critic
from trader.backtest import Costs
from trader.stats import DeflatedSharpe, deflated_sharpe
from trader.walkforward import WalkForward, walk_forward

MIN_POSITIVE_FOLD_SHARE = 0.6


@dataclass
class Report:
    strategy: str
    config: dict
    walk_forward: WalkForward
    oos: dict
    buy_and_hold: dict
    regimes: pd.DataFrame
    checks: list[critic.Check]
    dsr: DeflatedSharpe
    gates: dict[str, bool]

    @property
    def approved(self) -> bool:
        return all(self.gates.values())

    def to_dict(self) -> dict:
        folds = self.walk_forward.folds.copy()
        folds[["start", "end"]] = folds[["start", "end"]].map(lambda t: t.isoformat())
        return {
            "strategy": self.strategy,
            "approved": self.approved,
            "config": self.config,
            "gates": self.gates,
            "live_params": self.walk_forward.live_params,
            "oos": self.oos,
            "buy_and_hold": self.buy_and_hold,
            "deflated_sharpe": {**asdict(self.dsr), "passed": self.dsr.passed},
            "checks": [asdict(c) for c in self.checks],
            "folds": folds.to_dict(orient="records"),
            "regimes": self.regimes.reset_index().to_dict(orient="records"),
        }


def by_regime(net: pd.Series, close: pd.Series, periods_per_year: float, min_bars: int = 30) -> pd.DataFrame:
    """Out-of-sample metrics split by the regime in force when each position was chosen."""
    regime = critic.label_regimes(close).shift(1).reindex(net.index)
    rows = []
    for name in ("bull", "bear", "chop"):
        r = net[regime == name]
        rows.append({"regime": name, **(backtest.metrics(r, periods_per_year) if len(r) >= min_bars
                                        else {"bars": len(r)})})
    return pd.DataFrame(rows).set_index("regime")


def validate(
    bars: pd.DataFrame,
    strategy,
    costs: Costs,
    interval: str,
    periods_per_year: float,
    train_bars: int,
    test_bars: int,
    n_trials: int,
    config: dict,
) -> Report:
    inputs = strategy.inputs(bars)
    wf = walk_forward(bars, strategy, costs, periods_per_year, train_bars, test_bars, inputs=inputs)
    net = wf.oos["net"]
    oos = backtest.metrics(net, periods_per_year)

    close = bars["close"]
    hold = backtest.run(close.loc[net.index], pd.Series(1.0, index=net.index), costs)

    checks = [
        critic.lookahead_probe(bars, strategy, wf.live_params),
        critic.anonymity_check(bars, strategy),
        critic.cost_check(costs),
        critic.data_check(bars, interval),
        critic.regime_check(close, net.index),
        critic.sharpe_check(oos["sharpe"]),
        critic.model_version_check(inputs, net.index),
        critic.selection_note(config.get("symbol", "this asset")),
    ]
    dsr = deflated_sharpe(net, n_trials, periods_per_year)
    n_folds = len(wf.folds)
    gates = {
        "critic finds no leakage": all(c.status != critic.FAIL for c in checks),
        "deflated Sharpe > 0.95": dsr.passed,
        "walk-forward survives": oos["sharpe"] > 0 and wf.positive_folds >= MIN_POSITIVE_FOLD_SHARE * n_folds,
    }
    return Report(
        strategy=strategy.name,
        config=config,
        walk_forward=wf,
        oos=oos,
        buy_and_hold=backtest.metrics(hold["net"], periods_per_year),
        regimes=by_regime(net, close, periods_per_year),
        checks=checks,
        dsr=dsr,
        gates=gates,
    )
