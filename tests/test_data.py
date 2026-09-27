import pandas as pd
import pytest

from conftest import FakeExchange, make_bars
from trader import data


def test_fetch_paginates_and_drops_the_forming_bar():
    bars = make_bars(2000)
    now = bars.index[1500] + pd.Timedelta(minutes=20)      # bar 1500 is still forming
    ex = FakeExchange(bars, now, page=300)
    got = data.fetch_bars(ex, "BTC/USDT", "1h", bars.index[0].to_pydatetime(), now.to_pydatetime())
    assert ex.ohlcv_calls > 1
    assert got.index[-1] == bars.index[1499]
    pd.testing.assert_frame_equal(got, bars.iloc[:1500], check_freq=False, check_index_type=False)


def test_fetch_refuses_oversized_downloads():
    bars = make_bars(2000)
    ex = FakeExchange(bars, bars.index[-1], page=500)
    with pytest.raises(ValueError, match="bars"):
        data.fetch_bars(ex, "BTC/USDT", "1h", bars.index[0].to_pydatetime(), bars.index[-1].to_pydatetime(), max_bars=600)


def test_load_bars_caches_and_extends(tmp_path, monkeypatch):
    now = pd.Timestamp.now(tz="UTC").floor("h")
    bars = make_bars(24 * 40, start=str(now - pd.Timedelta(hours=24 * 40 - 1)))
    ex = FakeExchange(bars, now - pd.Timedelta(days=5))
    monkeypatch.setattr(data, "make_exchange", lambda name: ex)

    first = data.load_bars("binance", "BTC/USDT", "1h", days=30, cache_dir=tmp_path)
    assert (tmp_path / "binance" / "BTC-USDT_1h.csv").exists()

    ex.now = now + pd.Timedelta(minutes=30)
    calls = ex.ohlcv_calls
    second = data.load_bars("binance", "BTC/USDT", "1h", days=30, cache_dir=tmp_path)
    assert ex.ohlcv_calls - calls == 1                          # only the new bars were fetched
    assert second.index[-1] > first.index[-1]
    assert second.index.is_monotonic_increasing and not second.index.has_duplicates
    assert str(second.index.tz) == "UTC"


def test_intervals():
    assert data.periods_per_year("1h") == 365 * 24
    assert data.periods_per_year("1d") == 365
    with pytest.raises(ValueError):
        data.bar_minutes("3h")
