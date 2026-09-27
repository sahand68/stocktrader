import json

import httpx2
import numpy as np
import pandas as pd
import pytest
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from trader.jev import JevOracle

HOUR_MS = 3_600_000


def make_bars(n: int = 3000, drift: float = 0.0, vol: float = 0.01, seed: int = 0,
              start: str = "2025-01-01") -> pd.DataFrame:
    """Hourly geometric random walk, UTC bar-open timestamps."""
    rng = np.random.default_rng(seed)
    close = 50_000 * np.exp(np.cumsum(drift + vol * rng.standard_normal(n)))
    open_ = np.r_[close[0], close[:-1]]
    wick = np.abs(vol * rng.standard_normal(n)) * close
    return pd.DataFrame({
        "open": open_,
        "high": np.maximum(open_, close) + wick,
        "low": np.minimum(open_, close) - wick,
        "close": close,
        "volume": rng.uniform(100, 1000, n),
    }, index=pd.date_range(start, periods=n, freq="60min", tz="UTC", name="time"))


@pytest.fixture
def bars():
    return make_bars()


class FakeJev:
    """
    Stands in for api.typesafe.ai behind the real SDK. Its "view" is a fixed function
    of the state, so answers are deterministic and use only what the state contains.
    """

    def __init__(self, model: str = "jev-1.13.0", fail_after: int | None = None):
        self.model = model
        self.fail_after = fail_after
        self.requests: list[dict] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        if self.fail_after is not None and len(self.requests) > self.fail_after:
            return httpx2.Response(400, json={"error": {"message": "bad request"}})
        state = body["state"]
        tilt = float(np.tanh(state["trend_z"]["vs_sma_20"] / 3))
        p_up, p_down = 0.4 + 0.3 * tilt, 0.4 - 0.3 * tilt
        probs = {"up": p_up, "down": p_down, "unclear": 1 - p_up - p_down}
        best = max(probs, key=probs.get)
        return httpx2.Response(200, json={
            "model": self.model,
            "usage": {"input_tokens": 300, "output_tokens": 1},
            "answers": {name: {"type": "choice", "choice": best, "confidence": probs[best], "probabilities": probs}
                        for name in body["questions"]},
        })


@pytest.fixture
def fake_jev():
    return FakeJev()


def make_oracle(fake: FakeJev, tmp_path, model: str = "jev-latest") -> JevOracle:
    return JevOracle(
        model=model,
        cache_path=tmp_path / "jev.sqlite",
        requests_per_minute=1e9,
        client_factory=lambda: AsyncTypeSafeClient(
            api_key="test-key", model=model,
            transport=httpx2.MockTransport(fake), retry=RetryPolicy(max_retries=0),
        ),
        progress=False,
    )


@pytest.fixture
def oracle(fake_jev, tmp_path):
    return make_oracle(fake_jev, tmp_path)


class FakeExchange:
    """ccxt-shaped exchange serving hourly candles up to and including the one still forming."""

    def __init__(self, bars: pd.DataFrame, now: pd.Timestamp, page: int = 500):
        self.bars, self.now, self.page = bars, now, page
        self.ohlcv_calls = 0

    def fetch_ohlcv(self, symbol, timeframe, since=None):
        self.ohlcv_calls += 1
        rows = self.bars[(self.bars.index >= pd.Timestamp(since, unit="ms", tz="UTC")) & (self.bars.index <= self.now)]
        return [[int(ts.timestamp() * 1000), *row] for ts, row in zip(rows.index, rows.to_numpy().tolist())][: self.page]

    def fetch_ticker(self, symbol):
        last = float(self.bars.loc[: self.now, "close"].iloc[-1])
        return {"bid": last * 0.9995, "ask": last * 1.0005, "last": last}
