from types import SimpleNamespace

import pandas as pd
import pytest

import utils

HOUR_MS = 3_600_000


class FakeExchange:
    """Serves hourly candles up to and including the one currently forming."""
    page = 500

    def __init__(self, config):
        now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
        self.forming = now_ms - now_ms % HOUR_MS
        self.calls = 0

    def fetch_ohlcv(self, symbol, timeframe, since=None):
        self.calls += 1
        start = since - since % HOUR_MS
        if self.calls > 1:
            start -= HOUR_MS  # overlap the previous page by one bar, as some exchanges do
        stamps = list(range(start, self.forming + 1, HOUR_MS))[: self.page]
        return [[t, 1.0, 2.0, 0.5, 1.5, 10.0] for t in stamps]


@pytest.fixture
def fake_ccxt(monkeypatch):
    monkeypatch.setattr(utils, "ccxt", SimpleNamespace(binance=FakeExchange))


def test_paginates_and_drops_forming_bar(fake_ccxt):
    df = utils.get_crypto_data("BTC/USDT", period="60d", interval="1h")
    assert str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing and not df.index.has_duplicates
    assert len(df) >= 60 * 24 - 1
    assert df.index[-1] + pd.Timedelta(hours=1) <= pd.Timestamp.now(tz="UTC")
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_refuses_oversized_downloads(fake_ccxt):
    with pytest.raises(ValueError, match="bars"):
        utils.get_crypto_data("BTC/USDT", period="1y", interval="1h", max_bars=1000)


def test_rejects_unknown_interval(fake_ccxt):
    with pytest.raises(ValueError):
        utils.get_crypto_data("BTC/USDT", interval="3h")


def test_bars_per_day():
    assert utils.bars_per_day("1h") == 24
    assert utils.bars_per_day("4h") == 6
    assert utils.bars_per_day("1d") == 1
