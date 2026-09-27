import json

import numpy as np
import pandas as pd
import pytest

from conftest import FakeExchange, make_bars
from trader import cli
from trader.backtest import Costs
from trader.data import periods_per_year
from trader.paper import KillSwitch, PaperTrader

LENIENT = KillSwitch(max_drawdown=-0.9, min_sharpe=-1e9, window_bars=50)


class Scripted:
    """Target position read from a schedule keyed by bar time."""
    name = "scripted"

    def __init__(self, schedule: dict):
        self.schedule = schedule

    def inputs(self, bars):
        return pd.DataFrame({"t": np.arange(len(bars))}, index=bars.index)

    def positions(self, inputs, params):
        return pd.Series([self.schedule.get(ts, 0.0) for ts in inputs.index], index=inputs.index)


def make_trader(tmp_path, bars, now, strategy, kill=LENIENT):
    ex = FakeExchange(bars, now)
    trader = PaperTrader(ex, "BTC/USDT", "1h", strategy, {}, Costs(10, 5), kill, periods_per_year("1h"),
                         tmp_path / "account.json", tmp_path / "journal.jsonl", capital=10_000)
    return trader, ex


def test_buys_at_the_ask_with_fees_and_only_once_per_bar(tmp_path):
    bars = make_bars(600)
    now = bars.index[500] + pd.Timedelta(minutes=1)
    trader, ex = make_trader(tmp_path, bars, now, Scripted({bars.index[499]: 1.0}))

    entry = trader.step(now.to_pydatetime())
    ask = ex.fetch_ticker("BTC/USDT")["ask"]
    assert entry["bar"] == bars.index[499].isoformat()
    assert entry["fill"]["price"] == ask
    assert entry["fill"]["fee"] == pytest.approx(entry["fill"]["units"] * ask * 10 / 1e4)
    assert trader.account.units == pytest.approx(10_000 / ((ask + ex.fetch_ticker("")["bid"]) / 2))
    assert trader.step(now.to_pydatetime()) is None             # same bar again: nothing to do


def test_state_survives_restart(tmp_path):
    bars = make_bars(600)
    now = bars.index[500] + pd.Timedelta(minutes=1)
    trader, _ = make_trader(tmp_path, bars, now, Scripted({bars.index[499]: 1.0}))
    trader.step(now.to_pydatetime())

    restarted, _ = make_trader(tmp_path, bars, now, Scripted({}))
    assert restarted.account.units == trader.account.units
    assert restarted.step(now.to_pydatetime()) is None
    later = now + pd.Timedelta(hours=1)
    restarted.client.now = later
    entry = restarted.step(later.to_pydatetime())
    assert entry["fill"]["units"] < 0 and entry["fill"]["price"] == restarted.client.fetch_ticker("")["bid"]
    lines = (tmp_path / "journal.jsonl").read_text().splitlines()
    assert len(lines) == 2 and json.loads(lines[1])["target"] == 0.0


def test_kill_switch_flattens_and_halts(tmp_path):
    bars = make_bars(600)
    bars.loc[bars.index[501]:, ["open", "high", "low", "close"]] *= 0.5     # crash while long
    start = bars.index[500] + pd.Timedelta(minutes=1)
    always_long = Scripted({ts: 1.0 for ts in bars.index})
    trader, ex = make_trader(tmp_path, bars, start, always_long, KillSwitch(-0.2, -1e9, 50))

    trader.step(start.to_pydatetime())
    ex.now = start + pd.Timedelta(hours=2)
    entry = trader.step(ex.now.to_pydatetime())
    assert "drawdown" in entry["halted"]
    assert entry["target"] == 0.0 and trader.account.units == pytest.approx(0.0)
    ex.now += pd.Timedelta(hours=1)
    assert trader.step(ex.now.to_pydatetime()) is None


def test_kill_switch_from_report():
    report = {"oos": {"max_drawdown": -0.2}, "folds": [{"sharpe": 0.5, "bars": 720}, {"sharpe": -0.4, "bars": 720}]}
    kill = KillSwitch.from_report(report)
    assert kill.max_drawdown == pytest.approx(-0.3)
    assert kill.min_sharpe == -0.4 and kill.window_bars == 720


def test_paper_refuses_rejected_and_placebo_reports(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"approved": False, "config": {"placebo": False}}))
    with pytest.raises(SystemExit, match="rejected"):
        cli.main(["paper", str(path)])
    path.write_text(json.dumps({"approved": True, "config": {"placebo": True}}))
    with pytest.raises(SystemExit, match="placebo"):
        cli.main(["paper", str(path)])


def test_network_errors_are_retried_not_fatal(tmp_path, monkeypatch):
    import ccxt
    from trader import paper

    bars = make_bars(600)
    trader, ex = make_trader(tmp_path, bars, bars.index[500], Scripted({}))
    outcomes = iter([ccxt.NetworkError("timeout"), ccxt.NetworkError("timeout"), {"bar": "x"}])

    def flaky_step(now=None):
        o = next(outcomes)
        if isinstance(o, Exception):
            raise o
        return o

    monkeypatch.setattr(trader, "step", flaky_step)
    sleeps = []
    monkeypatch.setattr(paper.time, "sleep", sleeps.append)
    seen = []
    trader.run(max_bars=1, on_entry=seen.append)
    assert [e.get("consecutive") for e in seen] == [1, 2, None]
    assert len(sleeps) == 2        # waits after each failure, not after the last bar


def test_repeated_errors_eventually_stop(tmp_path, monkeypatch):
    import ccxt
    from trader import paper

    bars = make_bars(600)
    trader, _ = make_trader(tmp_path, bars, bars.index[500], Scripted({}))
    monkeypatch.setattr(trader, "step", lambda now=None: (_ for _ in ()).throw(ccxt.NetworkError("down")))
    monkeypatch.setattr(paper.time, "sleep", lambda s: None)
    with pytest.raises(ccxt.NetworkError):
        trader.run(on_entry=lambda e: None, max_consecutive_errors=3)
