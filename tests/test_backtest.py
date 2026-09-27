import numpy as np
import pandas as pd
import pytest

from conftest import make_bars
from trader import backtest
from trader.backtest import Costs
from trader.data import periods_per_year
from trader.stats import deflated_sharpe, expected_max_sharpe
from trader.strategies import SmaTrend
from trader.trials import TrialLedger
from trader.walkforward import walk_forward

HOURLY = periods_per_year("1h")
FREE = Costs(fee_bps=0, slippage_bps=0)


def test_position_is_previous_bars_target(bars):
    target = pd.Series(np.random.default_rng(1).choice([0.0, 1.0], len(bars)), index=bars.index)
    bt = backtest.run(bars["close"], target, Costs())
    assert bt["position"].iloc[0] == 0
    assert (bt["position"].iloc[1:].to_numpy() == target.iloc[:-1].to_numpy()).all()


def test_knowing_the_current_bar_is_worthless_after_the_shift(bars):
    same_bar = (bars["close"].pct_change() > 0).astype(float)
    bt = backtest.run(bars["close"], same_bar, FREE)
    assert abs(backtest.sharpe(bt["net"], HOURLY)) < 2


def test_costs_charged_per_unit_of_turnover(bars):
    flips = pd.Series(np.tile([1.0, 0.0], len(bars) // 2), index=bars.index)
    bt = backtest.run(bars["close"], flips, Costs(10, 5))
    assert bt["cost"].sum() == pytest.approx((len(bars) - 1) * 15 / 1e4)


def test_equity_floors_at_zero():
    idx = pd.date_range("2025", periods=4, freq="D", tz="UTC")
    bt = backtest.run(pd.Series([100.0, 100, 300, 900], idx), pd.Series(-1.0, idx), Costs())
    assert bt["equity"].iloc[-1] == 0
    assert (bt["equity"] >= 0).all()


def test_misaligned_target_rejected(bars):
    with pytest.raises(ValueError):
        backtest.run(bars["close"], pd.Series(1.0, index=bars.index[1:]), Costs())


def test_drawdown_depth_and_duration():
    m = backtest.metrics(pd.Series([0.1, -0.5, 0.0, 0.2, 1.0]), periods_per_year=365)
    assert m["max_drawdown"] == pytest.approx(-0.5)
    assert m["longest_drawdown_days"] == pytest.approx(3)


def test_noise_benchmark_grows_with_trials():
    assert expected_max_sharpe(1, 1.0) == 0
    assert expected_max_sharpe(10, 1.0) < expected_max_sharpe(100, 1.0) < expected_max_sharpe(1000, 1.0)


def test_best_of_many_noise_strategies_rejected():
    rng = np.random.default_rng(7)
    trials = [pd.Series(rng.normal(0, 0.01, 2000)) for _ in range(200)]
    best = max(trials, key=lambda r: r.mean() / r.std())
    assert deflated_sharpe(best, 1, 365).passed          # brilliant if you forget the other 199
    assert not deflated_sharpe(best, 200, 365).passed


def test_real_edge_survives_deflation():
    r = pd.Series(np.random.default_rng(3).normal(0.002, 0.01, 2000))
    assert deflated_sharpe(r, 100, 365).passed


def test_moderate_edge_is_not_a_fixed_sharpe_cutoff():
    # The widely copied version compares an annualized Sharpe with a z-score and
    # rejects any annual Sharpe below ~2.4 regardless of sample size.
    r = pd.Series(np.random.default_rng(5).normal(0.0008, 0.01, 365 * 8))
    dsr = deflated_sharpe(r, 5, 365)
    assert dsr.sharpe < 2.4 and dsr.passed


def test_walk_forward_folds_and_boundaries(bars):
    wf = walk_forward(bars, SmaTrend(), Costs(), HOURLY, train_bars=1000, test_bars=250)
    assert len(wf.folds) == (len(bars) - 1000) // 250
    assert wf.oos.index[0] == bars.index[1000]
    assert (wf.folds["start"].iloc[1:].to_numpy() > wf.folds["end"].iloc[:-1].to_numpy()).all()
    assert set(wf.live_params) == {"fast", "slow"}


def test_fit_never_sees_the_test_window(bars):
    seen = []

    class Spy(SmaTrend):
        def positions(self, inputs, params):
            seen.append(inputs.index[-1])
            return super().positions(inputs, params)

    wf = walk_forward(bars, Spy(), Costs(), HOURLY, train_bars=1000, test_bars=500)
    fold_starts = list(wf.folds["start"])
    grid = len(Spy().grid())
    for k, start in enumerate(fold_starts):
        fitting = seen[k * (grid + 1): k * (grid + 1) + grid]
        assert all(ts < start for ts in fitting)


def test_walk_forward_needs_two_folds(bars):
    with pytest.raises(ValueError, match="fold"):
        walk_forward(bars, SmaTrend(), Costs(), HOURLY, train_bars=2600, test_bars=300)


def test_walk_forward_rejects_tiny_test_windows(bars):
    with pytest.raises(ValueError, match="minimum"):
        walk_forward(bars, SmaTrend(), Costs(), HOURLY, train_bars=1000, test_bars=10)


def test_trial_ledger_counts_distinct_configs_per_dataset(tmp_path):
    ledger = TrialLedger(tmp_path / "trials.jsonl")
    btc, eth = {"symbol": "BTC/USDT"}, {"symbol": "ETH/USDT"}
    assert ledger.record(btc, {"threshold": 0.1}) == 1
    assert ledger.record(btc, {"threshold": 0.1}) == 1
    assert ledger.record(btc, {"threshold": 0.2}) == 2
    assert ledger.record(eth, {"threshold": 0.1}) == 1
    assert TrialLedger(tmp_path / "trials.jsonl").record(btc, {"threshold": 0.3}) == 3


def test_sma_trend_long_flat_or_short():
    b = make_bars(600, drift=0.001, seed=2)
    s = SmaTrend(allow_short=True)
    pos = s.positions(s.inputs(b), {"fast": 20, "slow": 200})
    assert set(pos.unique()) <= {-1.0, 0.0, 1.0}
    assert (pos.iloc[:199] == 0).all()
