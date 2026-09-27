import json

import numpy as np
import pytest
from typesafe_sdk import TypeSafeBadRequestError

from conftest import FakeJev, make_oracle
from trader.features import build_features, ready
from trader.jev import direction_question, make_state
from trader.strategies import JevDirection


def test_request_shape_and_parsed_answers(bars, fake_jev, oracle):
    strategy = JevDirection(oracle, "1h")
    inputs = strategy.inputs(bars)

    body = fake_jev.requests[0]
    assert body["model"] == "jev-latest"
    q = body["questions"]["direction"]
    assert q["type"] == "choice" and set(q["criteria"]) == {"up", "down", "unclear"}

    answered = inputs.dropna()
    assert len(answered) == len(fake_jev.requests) == ready(build_features(bars)).sum()
    assert np.allclose(answered[["p_up", "p_down", "p_unclear"]].sum(axis=1), 1)
    assert (answered["model"] == "jev-1.13.0").all()   # the version that answered, not the alias
    assert inputs.iloc[:199].isna().all().all()        # warm-up bars are never sent


def test_cache_makes_reruns_free(bars, fake_jev, tmp_path):
    JevDirection(make_oracle(fake_jev, tmp_path), "1h").inputs(bars)
    first = len(fake_jev.requests)

    rerun = make_oracle(fake_jev, tmp_path)                  # new process, same cache file
    JevDirection(rerun, "1h").inputs(bars)
    assert len(fake_jev.requests) == first
    assert rerun.calls == 0 and rerun.cost_usd == 0


def test_cache_is_keyed_by_model(bars, fake_jev, tmp_path):
    JevDirection(make_oracle(fake_jev, tmp_path, "jev-latest"), "1h").inputs(bars.iloc[:300])
    before = len(fake_jev.requests)
    JevDirection(make_oracle(fake_jev, tmp_path, "jev-1.13.0"), "1h").inputs(bars.iloc[:300])
    assert len(fake_jev.requests) == 2 * before


def test_cost_accounting(bars, oracle):
    JevDirection(oracle, "1h").inputs(bars.iloc[:400])
    assert oracle.input_tokens == 300 * oracle.calls
    assert oracle.cost_usd == pytest.approx(oracle.input_tokens * 0.042 / 1e6)


def test_failed_calls_raise_but_keep_answers_already_paid_for(bars, tmp_path):
    flaky = FakeJev(fail_after=50)
    with pytest.raises(TypeSafeBadRequestError):
        JevDirection(make_oracle(flaky, tmp_path), "1h").inputs(bars.iloc[:400])
    healthy = FakeJev()
    JevDirection(make_oracle(healthy, tmp_path), "1h").inputs(bars.iloc[:400])
    assert len(healthy.requests) == 201 - 50


def test_state_is_anonymized(bars):
    f = build_features(bars)
    row = f[ready(f)].iloc[0]
    text = json.dumps(make_state(row, "1h"))
    assert "2025" not in text
    for value in bars[["open", "high", "low", "close"]].iloc[:260].to_numpy().ravel()[::50]:
        assert f"{value:.2f}" not in text


def test_question_mentions_the_bar_interval():
    assert "4h" in direction_question("4h").instructions


def test_threshold_and_shorts(oracle, bars):
    s = JevDirection(oracle, "1h")
    inputs = s.inputs(bars)
    edge = inputs["p_up"] - inputs["p_down"]
    loose = s.positions(inputs, {"threshold": 0.0})
    tight = s.positions(inputs, {"threshold": 0.3})
    assert (loose[edge > 0] == 1).all() and (loose[edge <= 0] == 0).all()
    assert tight.sum() <= loose.sum()
    both = JevDirection(oracle, "1h", allow_short=True).positions(inputs, {"threshold": 0.1})
    assert (both[edge < -0.1] == -1).all()


def test_requests_are_paced_under_the_rate_limit(bars, fake_jev, tmp_path):
    import time
    oracle = make_oracle(fake_jev, tmp_path)
    oracle.min_gap = 60 / 6000          # 100 requests per second
    started = time.monotonic()
    JevDirection(oracle, "1h").inputs(bars.iloc[:250])     # 51 states
    assert time.monotonic() - started >= 50 * oracle.min_gap
