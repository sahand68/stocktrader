import pandas as pd
import pytest

from trader import critic
from trader.backtest import Costs
from trader.jev import make_state
from trader.strategies import JevDirection, SmaTrend


class Clairvoyant(SmaTrend):
    """Long exactly when the next bar goes up."""
    name = "clairvoyant"

    def inputs(self, bars):
        return pd.DataFrame({"next": bars["close"].pct_change().shift(-1)})

    def grid(self):
        return [{}]

    def positions(self, inputs, params):
        return (inputs["next"] > 0).astype(float)


class CenteredAverage(Clairvoyant):
    name = "centered"

    def inputs(self, bars):
        close = bars["close"]
        return pd.DataFrame({"next": close / close.rolling(21, center=True).mean() - 1})


class PriceLeakingJev(JevDirection):
    def states(self, bars):
        index, states = super().states(bars)
        return index, [{**s, "last_close": round(float(bars.loc[t, "close"]), 2)} for t, s in zip(index, states)]


class DateLeakingJev(JevDirection):
    def states(self, bars):
        index, states = super().states(bars)
        return index, [{**s, "weekday": t.day_name(), "month": t.month} for t, s in zip(index, states)]


def test_probe_passes_causal_strategies(bars, oracle):
    assert critic.lookahead_probe(bars, SmaTrend(), {"fast": 20, "slow": 100}).status == critic.PASS
    assert critic.lookahead_probe(bars, JevDirection(oracle, "1h"), {"threshold": 0.1}).status == critic.PASS


@pytest.mark.parametrize("strategy", [Clairvoyant(), CenteredAverage()])
def test_probe_catches_lookahead(bars, strategy):
    assert critic.lookahead_probe(bars, strategy, {}).status == critic.FAIL


def test_anonymity_passes_real_states(bars, oracle):
    assert critic.anonymity_check(bars, JevDirection(oracle, "1h")).status == critic.PASS
    assert critic.anonymity_check(bars, SmaTrend()).status == critic.PASS


@pytest.mark.parametrize("leaky, reason", [(PriceLeakingJev, "price"), (DateLeakingJev, "date")])
def test_anonymity_catches_identifying_states(bars, oracle, leaky, reason):
    check = critic.anonymity_check(bars, leaky(oracle, "1h"))
    assert check.status == critic.FAIL and reason in check.detail


def test_cost_check():
    assert critic.cost_check(Costs(0, 5)).status == critic.FAIL
    assert critic.cost_check(Costs(3, 2)).status == critic.WARN
    assert critic.cost_check(Costs(10, 5)).status == critic.PASS


def test_data_check(bars):
    assert critic.data_check(bars, "1h").status == critic.PASS
    assert critic.data_check(bars.drop(bars.index[100:104]), "1h").status == critic.WARN
    assert critic.data_check(bars.tz_localize(None), "1h").status == critic.FAIL
    assert critic.data_check(bars.iloc[::-1], "1h").status == critic.FAIL


def test_sharpe_check():
    assert critic.sharpe_check(1.1).status == critic.PASS
    assert critic.sharpe_check(2.4).status == critic.WARN
    assert critic.sharpe_check(3.5).status == critic.FAIL


def test_regimes_use_trailing_data_only(bars):
    full = critic.label_regimes(bars["close"])
    assert (full.iloc[:1500] == critic.label_regimes(bars["close"].iloc[:1500])).all()
    assert set(full.unique()) <= {"bull", "bear", "chop", "warmup"}


def test_model_version_check(bars):
    idx = bars.index[:10]
    one = pd.DataFrame({"model": ["jev-1.13.0"] * 10}, index=idx)
    two = pd.DataFrame({"model": ["jev-1.13.0"] * 5 + ["jev-1.14.0"] * 5}, index=idx)
    assert critic.model_version_check(one, idx).status == critic.PASS
    assert critic.model_version_check(two, idx).status == critic.WARN


def test_make_state_ignores_levels(bars):
    from trader.features import build_features, ready
    f = build_features(bars)
    g = build_features(bars.assign(**{c: bars[c] * 2 for c in ("open", "high", "low", "close")}))
    row = f[ready(f)].index[5]
    assert make_state(f.loc[row], "1h") == make_state(g.loc[row], "1h")
