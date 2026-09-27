import numpy as np
import pandas as pd
import pytest

from backtest import critic
from backtest.engine import Config, backtest, periods_per_year
from backtest.metrics import metrics
from backtest.regimes import label_regimes
from backtest.risk import health_check, losing_streak_drawdown, position_size
from backtest.stats import deflated_sharpe, expected_max_sharpe
from backtest.strategies import GridStrategy, RsiReversion, SmaCross, XGBoostDirection
from backtest.validate import validate
from backtest.walkforward import walk_forward
from conftest import make_ohlcv

HOURLY = periods_per_year("1h")
CFG = Config(periods_per_year=HOURLY)


class Cheat(GridStrategy):
    """Holds the coin exactly when the next bar goes up."""
    name = "cheat"

    def grid(self):
        return [{}]

    def signal(self, history, params):
        nxt = history["Close"].pct_change().shift(-1)
        return (nxt > 0).astype(float)


class CenteredSma(GridStrategy):
    """Centered window: each value averages bars on both sides."""
    name = "centered"

    def grid(self):
        return [{}]

    def signal(self, history, params):
        close = history["Close"]
        ma = close.rolling(21, center=True).mean()
        return (close > ma).astype(float)


# --- engine -----------------------------------------------------------------

def test_position_lags_signal_by_one_bar(ohlcv):
    signal = pd.Series(np.random.default_rng(1).choice([0.0, 1.0], len(ohlcv)), index=ohlcv.index)
    bt = backtest(ohlcv["Close"], signal, CFG)
    assert bt["position"].iloc[0] == 0
    assert (bt["position"].iloc[1:].to_numpy() == signal.iloc[:-1].to_numpy()).all()


def test_foresight_signal_is_neutralized_by_the_shift(ohlcv):
    # Knowing the current bar's return only helps if you could trade before it happened.
    same_bar = (ohlcv["Close"].pct_change() > 0).astype(float)
    bt = backtest(ohlcv["Close"], same_bar, Config(periods_per_year=HOURLY, fee_bps=0, slippage_bps=0))
    assert abs(metrics(bt["net"], HOURLY)["sharpe"]) < 2


def test_costs_charged_once_per_unit_of_turnover(ohlcv):
    signal = pd.Series(1.0, index=ohlcv.index)
    bt = backtest(ohlcv["Close"], signal, CFG)
    assert bt["costs"].sum() == pytest.approx((CFG.fee_bps + CFG.slippage_bps) / 1e4)


def test_equity_never_negative():
    prices = pd.Series([100.0, 100.0, 300.0, 900.0], index=pd.date_range("2024", periods=4, freq="D", tz="UTC"))
    bt = backtest(prices, pd.Series(-1.0, index=prices.index), Config(periods_per_year=365))
    assert (bt["equity"] >= 0).all()
    assert bt["equity"].iloc[-1] == 0


def test_backtest_rejects_misaligned_signal(ohlcv):
    with pytest.raises(ValueError):
        backtest(ohlcv["Close"], pd.Series(1.0, index=ohlcv.index[1:]), CFG)


# --- metrics ----------------------------------------------------------------

def test_drawdown_and_duration():
    net = pd.Series([0.1, -0.5, 0.0, 0.2, 1.0] + [0.0] * 30)
    m = metrics(net, periods_per_year=365)
    assert m["max_drawdown"] == pytest.approx(-0.5)
    assert m["longest_dd_days"] == pytest.approx(3)


def test_metrics_refuses_tiny_samples():
    with pytest.raises(ValueError):
        metrics(pd.Series([0.01] * 5), 365)


# --- deflated Sharpe ----------------------------------------------------------

def test_noise_benchmark_grows_with_trials():
    assert expected_max_sharpe(1, 1.0) == 0
    assert expected_max_sharpe(10, 1.0) < expected_max_sharpe(100, 1.0) < expected_max_sharpe(1000, 1.0)


def test_best_of_many_noise_strategies_is_rejected():
    rng = np.random.default_rng(7)
    trials = [pd.Series(rng.normal(0, 0.01, 2000)) for _ in range(200)]
    best = max(trials, key=lambda r: r.mean() / r.std())
    single = deflated_sharpe(best, n_trials=1, periods_per_year=365)
    honest = deflated_sharpe(best, n_trials=200, periods_per_year=365)
    assert single.passed          # looks brilliant if you forget the other 199
    assert not honest.passed
    assert honest.noise_benchmark > 0


def test_real_edge_survives_deflation():
    r = pd.Series(np.random.default_rng(3).normal(0.002, 0.01, 2000))
    assert deflated_sharpe(r, n_trials=100, periods_per_year=365).passed


def test_sharpe_reported_annualized():
    r = pd.Series(np.random.default_rng(4).normal(0.001, 0.01, 500))
    dsr = deflated_sharpe(r, n_trials=1, periods_per_year=365)
    assert dsr.sharpe == pytest.approx(r.mean() / r.std() * np.sqrt(365))


# --- walk-forward -------------------------------------------------------------

def test_walk_forward_short_test_windows(ohlcv):
    # The original version crashed whenever a fold had fewer than 100 bars.
    wf = walk_forward(ohlcv, SmaCross(), CFG, train_bars=1000, test_bars=60)
    assert len(wf.folds) == (len(ohlcv) - 1000) // 60
    assert wf.oos.index[0] == ohlcv.index[1000]
    assert {"sharpe", "fast", "slow"} <= set(wf.folds.columns)


def test_walk_forward_needs_two_folds(ohlcv):
    with pytest.raises(ValueError, match="fold"):
        walk_forward(ohlcv, SmaCross(), CFG, train_bars=2500, test_bars=400)


def test_fit_only_sees_training_window(ohlcv):
    seen = []

    class Spy(SmaCross):
        def fit(self, train, cfg):
            seen.append(train.index[-1])
            return super().fit(train, cfg)

    wf = walk_forward(ohlcv, Spy(), CFG, train_bars=1000, test_bars=500)
    for last_train_bar, start in zip(seen, wf.folds["start"]):
        assert last_train_bar < start


# --- critic -------------------------------------------------------------------

def test_probe_passes_causal_strategies(ohlcv):
    assert critic.lookahead_probe(ohlcv, SmaCross(), {"fast": 10, "slow": 100}).status == critic.PASS
    assert critic.lookahead_probe(ohlcv, RsiReversion(), {"lower": 30, "upper": 60}).status == critic.PASS


@pytest.mark.parametrize("strategy", [Cheat(), CenteredSma()])
def test_probe_catches_lookahead(ohlcv, strategy):
    assert critic.lookahead_probe(ohlcv, strategy, {}).status == critic.FAIL


def test_cost_check():
    assert critic.cost_check(Config(fee_bps=0)).status == critic.FAIL
    assert critic.cost_check(Config(fee_bps=2, slippage_bps=1)).status == critic.WARN
    assert critic.cost_check(Config()).status == critic.PASS


def test_data_check(ohlcv):
    assert critic.data_check(ohlcv, "1h").status == critic.PASS
    assert critic.data_check(ohlcv.drop(ohlcv.index[100:103]), "1h").status == critic.WARN
    assert critic.data_check(ohlcv.tz_localize(None), "1h").status == critic.FAIL
    assert critic.data_check(ohlcv.iloc[::-1], "1h").status == critic.FAIL


def test_sharpe_check():
    assert critic.sharpe_check(1.2).status == critic.PASS
    assert critic.sharpe_check(2.5).status == critic.WARN
    assert critic.sharpe_check(4.0).status == critic.FAIL


def test_regimes_are_causal(ohlcv):
    full = label_regimes(ohlcv["Close"])
    partial = label_regimes(ohlcv["Close"].iloc[:1500])
    assert (full.iloc[:1500] == partial).all()


# --- strategies ---------------------------------------------------------------

def test_rsi_short_entries_do_not_cancel_longs(ohlcv):
    params = {"lower": 30, "upper": 60}
    long_only = RsiReversion().signal(ohlcv, params)
    both = RsiReversion(allow_short=True).signal(ohlcv, params)
    assert set(both.unique()) <= {-1.0, 0.0, 1.0}
    assert (both[long_only == 1] >= 0).all()
    assert (both == -1).any()


def test_xgboost_walk_forward(ohlcv):
    pytest.importorskip("xgboost")
    strategy = XGBoostDirection(n_estimators=20)
    wf = walk_forward(ohlcv, strategy, CFG, train_bars=1500, test_bars=500)
    assert len(wf.folds) == 3
    assert critic.lookahead_probe(ohlcv, strategy, wf.params[-1], n_probes=3).status == critic.PASS


# --- gates ----------------------------------------------------------------------

def test_noise_fails_the_gates(ohlcv):
    report = validate(ohlcv, SmaCross(), CFG, "1h", train_bars=1000, test_bars=500, n_trials=10)
    assert not report.approved
    assert not report.gates["Deflated Sharpe > 0.95"]


def test_cheat_is_blocked_by_the_critic(ohlcv):
    report = validate(ohlcv, Cheat(), CFG, "1h", train_bars=1000, test_bars=500, n_trials=1)
    assert not report.gates["Critic finds no leakage"]
    assert not report.approved


# --- risk ---------------------------------------------------------------------

def test_position_size_risk_and_cap():
    s = position_size(10_000, entry=100, stop=95, risk_pct=0.01)
    assert s["loss_if_stopped"] == pytest.approx(100)
    tight = position_size(10_000, entry=100, stop=99.9, risk_pct=0.01, max_position_pct=0.2)
    assert tight["notional"] == pytest.approx(2_000)
    with pytest.raises(ValueError):
        position_size(10_000, 100, 100)


def test_losing_streaks():
    assert losing_streak_drawdown(0.01, 12) == pytest.approx(0.114, abs=1e-3)
    assert losing_streak_drawdown(0.05, 12) == pytest.approx(0.46, abs=1e-2)


def test_health_check():
    good = pd.Series(np.random.default_rng(5).normal(0.002, 0.01, 200))
    assert health_check(good, 1.0, -0.2, 365)["action"] == "CONTINUE"
    crash = pd.concat([good, pd.Series([-0.1] * 5)], ignore_index=True)
    result = health_check(crash, 1.0, -0.2, 365)
    assert result["action"] == "HALT"
    assert "DRAWDOWN_EXCEEDED" in result["alerts"]
