"""
The three gates a strategy must clear before it touches money:
  1. the critic finds no leakage
  2. the deflated Sharpe clears the multiple-testing bar
  3. out-of-sample performance survives walk-forward
"""
from dataclasses import dataclass

import pandas as pd

from backtest import critic
from backtest.engine import Config
from backtest.metrics import metrics
from backtest.regimes import metrics_by_regime
from backtest.stats import DeflatedSharpe, deflated_sharpe
from backtest.walkforward import WalkForwardResult, walk_forward


@dataclass
class Report:
    walk_forward: WalkForwardResult
    oos_metrics: dict
    regimes: pd.DataFrame
    checks: list[critic.Check]
    dsr: DeflatedSharpe
    gates: dict[str, bool]

    @property
    def approved(self) -> bool:
        return all(self.gates.values())


def validate(
    df: pd.DataFrame,
    strategy,
    cfg: Config,
    interval: str,
    train_bars: int,
    test_bars: int,
    n_trials: int,
    min_positive_fold_share: float = 0.6,
) -> Report:
    wf = walk_forward(df, strategy, cfg, train_bars, test_bars)
    oos_net = wf.oos["net"]
    oos_metrics = metrics(oos_net, cfg.periods_per_year)

    checks = [
        critic.lookahead_probe(df, strategy, wf.params[-1]),
        critic.cost_check(cfg),
        critic.data_check(df, interval),
        critic.regime_check(df["Close"], oos_net.index),
        critic.sharpe_check(oos_metrics["sharpe"]),
        critic.selection_note(),
    ]
    dsr = deflated_sharpe(oos_net, n_trials, cfg.periods_per_year)

    n_folds = len(wf.folds)
    gates = {
        "Critic finds no leakage": not any(c.status == critic.FAIL for c in checks),
        "Deflated Sharpe > 0.95": dsr.passed,
        "Walk-forward survives": (
            oos_metrics["sharpe"] > 0
            and wf.positive_folds >= min_positive_fold_share * n_folds
        ),
    }
    return Report(
        walk_forward=wf,
        oos_metrics=oos_metrics,
        regimes=metrics_by_regime(oos_net, df["Close"], cfg.periods_per_year),
        checks=checks,
        dsr=dsr,
        gates=gates,
    )
