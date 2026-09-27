import json

from test_critic import Clairvoyant
from trader.backtest import Costs
from trader.data import periods_per_year
from trader.strategies import JevDirection, SmaTrend
from trader.validate import validate

HOURLY = periods_per_year("1h")


def run(bars, strategy, n_trials=1):
    return validate(bars, strategy, Costs(), "1h", HOURLY, train_bars=1000, test_bars=250,
                    n_trials=n_trials, config={"symbol": "TEST/USDT"})


def test_random_walk_is_rejected(bars):
    report = run(bars, SmaTrend(), n_trials=10)
    assert not report.approved
    assert not report.gates["deflated Sharpe > 0.95"]


def test_clairvoyant_strategy_is_blocked_by_the_critic(bars):
    report = run(bars, Clairvoyant())
    assert not report.gates["critic finds no leakage"]
    assert not report.approved
    failed = {c.name for c in report.checks if c.status == "FAIL"}
    assert {"Look-ahead", "Sharpe sanity"} <= failed


def test_jev_end_to_end_and_report_serializes(bars, oracle):
    report = run(bars, JevDirection(oracle, "1h"))
    data = json.loads(json.dumps(report.to_dict(), default=float))
    assert data["strategy"] == "jev"
    assert set(data["gates"]) == {"critic finds no leakage", "deflated Sharpe > 0.95", "walk-forward survives"}
    assert "threshold" in data["live_params"]
    assert {c["name"] for c in data["checks"]} >= {"Look-ahead", "Anonymized state", "Model version"}
    assert len(data["folds"]) == (len(bars) - 1000) // 250
    # every check after the first pass hits the answer cache
    assert oracle.calls == len(bars) - 199
