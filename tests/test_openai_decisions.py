import base64
import json

import numpy as np
import pandas as pd
import pytest

from conftest import FakeDecisions, make_openai_oracle
from trader import critic
from trader.compare import skill
from trader.features import build_features, ready
from trader.jev import direction_question, make_state
from trader.openai_decisions import CHART_BARS, chart_window
from trader.strategies import OpenAIDirection


@pytest.fixture
def short_bars(bars):
    return bars.iloc[:230]   # 31 decision bars after the 200-bar warm-up; each chart takes ~0.1 s to draw


@pytest.mark.parametrize("evidence", ["state", "chart", "state-chart"])
def test_request_shape(short_bars, tmp_path, evidence):
    fake = FakeDecisions()
    OpenAIDirection(make_openai_oracle(fake, tmp_path), "1h", evidence).inputs(short_bars)

    body = fake.requests[0]
    assert body["model"] == "gpt-6-luna"
    q = body["questions"][0]
    assert q["type"] == "choice" and "1h" in q["instructions"]
    jev = direction_question("1h")
    assert {c["value"]: c["description"] for c in q["choices"]} == jev.criteria
    if evidence == "state":
        assert q["instructions"] == jev.instructions

    parts = {p["type"]: p for p in body["input"][0]["content"]}
    features = build_features(short_bars)
    states = [make_state(row, "1h") for _, row in features[ready(features)].iterrows()]
    assert ("input_text" in parts) == (evidence != "chart")
    assert ("input_image" in parts) == (evidence != "state")
    if "input_text" in parts:
        assert json.loads(parts["input_text"]["text"]) in states   # requests run concurrently, in any order
    if "input_image" in parts:
        url = parts["input_image"]["image_url"]
        assert base64.b64decode(url.removeprefix("data:image/png;base64,"))[:4] == b"\x89PNG"


def test_answers_on_the_same_bars_as_jev(short_bars, tmp_path, oracle):
    from trader.strategies import JevDirection

    fake = FakeDecisions()
    openai = OpenAIDirection(make_openai_oracle(fake, tmp_path), "1h", "chart").inputs(short_bars)
    jev = JevDirection(oracle, "1h").inputs(short_bars)
    assert openai.dropna().index.equals(jev.dropna().index)
    assert np.allclose(openai.dropna()[["p_up", "p_down", "p_unclear"]].sum(axis=1), 1)
    assert (openai.dropna()["model"] == fake.model).all()


def test_refusal_reads_as_no_edge(short_bars, tmp_path):
    fake = FakeDecisions(refuse_every=4)
    oracle = make_openai_oracle(fake, tmp_path)
    strategy = OpenAIDirection(oracle, "1h", "state")
    inputs = strategy.inputs(short_bars)
    refused = inputs.dropna()[["p_up", "p_down", "p_unclear"]].sum(axis=1) == 0
    assert refused.sum() == oracle.refusals == len(fake.requests) // 4
    assert (strategy.positions(inputs, {"threshold": 0.0})[refused[refused].index] == 0).all()
    assert "refused" in oracle.describe_usage() and "$" not in oracle.describe_usage()


def test_cache_is_keyed_by_evidence_and_model(short_bars, tmp_path):
    fake = FakeDecisions()
    OpenAIDirection(make_openai_oracle(fake, tmp_path), "1h", "chart").inputs(short_bars)
    n = len(fake.requests)
    OpenAIDirection(make_openai_oracle(fake, tmp_path), "1h", "chart").inputs(short_bars)
    assert len(fake.requests) == n
    OpenAIDirection(make_openai_oracle(fake, tmp_path), "1h", "state").inputs(short_bars)
    OpenAIDirection(make_openai_oracle(fake, tmp_path, "gpt-6-luna-pinned"), "1h", "chart").inputs(short_bars)
    assert len(fake.requests) == 3 * n


@pytest.mark.parametrize("evidence", ["chart", "state-chart"])
def test_charts_pass_the_critic(short_bars, tmp_path, evidence):
    strategy = OpenAIDirection(make_openai_oracle(FakeDecisions(), tmp_path), "1h", evidence)
    assert critic.anonymity_check(short_bars, strategy).status == critic.PASS
    assert critic.lookahead_probe(short_bars, strategy, {"threshold": 0.0}).status == critic.PASS


def test_chart_window_is_rebased_and_causal(bars):
    ohlc, volume = bars[["open", "high", "low", "close"]].to_numpy(), bars["volume"].to_numpy()
    rows = np.array(chart_window(ohlc, volume, 250))
    assert rows.shape == (CHART_BARS, 5)
    assert rows[-1, 3] == 100
    assert rows[:, 4].mean() == pytest.approx(1, abs=0.01)
    changed = ohlc.copy()
    changed[251:] *= 3
    assert chart_window(changed, volume, 250) == rows.tolist()


def test_skill_is_paired_and_finds_a_real_signal():
    rng = np.random.default_rng(1)
    ret = pd.Series(rng.normal(0, 0.01, 2000))
    edges = pd.DataFrame({
        "noise": rng.uniform(-0.5, 0.5, 2000),
        "informed": np.sign(ret.shift(-1)).fillna(0) * 0.3 * (rng.random(2000) < 0.6)
                    + rng.uniform(-0.1, 0.1, 2000),
    })
    edges.iloc[:10, 0] = np.nan   # bars one model never answered are dropped for every model
    table = skill(edges, ret.shift(-1))
    assert (table["bars"] == 1990 - 1).all()   # minus the last bar, which has no next return
    assert table.loc["noise", "hit_p"] > 0.01
    assert table.loc["informed", "hit_rate"] > 0.6 and table.loc["informed", "ic"] > 0.2
    assert table.loc["informed", "vs_ref_p"] < 1e-6
    assert "vs_ref_p" not in table.columns or np.isnan(table.loc["noise", "vs_ref_p"])
