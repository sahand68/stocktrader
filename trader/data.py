"""OHLCV bars from public exchange endpoints. No API key needed."""
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ccxt
import pandas as pd

# Bar length in minutes for every supported interval.
TIMEFRAMES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}

COLUMNS = ["open", "high", "low", "close", "volume"]


def bar_minutes(interval: str) -> int:
    if interval not in TIMEFRAMES:
        raise ValueError(f"unsupported interval {interval!r}; use one of {', '.join(TIMEFRAMES)}")
    return TIMEFRAMES[interval]


def periods_per_year(interval: str) -> float:
    """Crypto trades 24/7/365."""
    return 365 * 1440 / bar_minutes(interval)


def make_exchange(exchange: str):
    return getattr(ccxt, exchange)({"enableRateLimit": True})


def fetch_bars(
    client,
    symbol: str,
    interval: str,
    since: datetime,
    until: datetime | None = None,
    max_bars: int = 200_000,
) -> pd.DataFrame:
    """
    Closed bars from `since` up to `until` (default now), indexed by bar open time in UTC.
    The bar that is still forming is dropped: its close would change after a decision used it.
    """
    bar_ms = bar_minutes(interval) * 60_000
    until = until or datetime.now(timezone.utc)
    until_ms = int(until.timestamp() * 1000)
    cursor = int(since.timestamp() * 1000)

    rows = []
    while cursor < until_ms:
        batch = client.fetch_ohlcv(symbol, interval, since=cursor)
        if not batch:
            break
        rows.extend(batch)
        if len(rows) > max_bars:
            raise ValueError(f"more than {max_bars:,} bars requested; use a larger interval or fewer days")
        next_cursor = batch[-1][0] + bar_ms
        if next_cursor <= cursor:
            break
        cursor = next_cursor

    if not rows:
        return pd.DataFrame(columns=COLUMNS, index=pd.DatetimeIndex([], tz="UTC", name="time"), dtype=float)

    df = pd.DataFrame(rows, columns=["time", *COLUMNS]).drop_duplicates("time").sort_values("time")
    df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("time"), unit="ms", utc=True), name="time")
    closed = df.index + pd.Timedelta(milliseconds=bar_ms) <= pd.Timestamp(until)
    return df[closed].astype(float)


def load_bars(
    exchange: str,
    symbol: str,
    interval: str,
    days: int,
    cache_dir: Path = Path("data"),
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch `days` of bars, reusing and extending a CSV cache so reruns only download new bars."""
    path = cache_dir / exchange / f"{symbol.replace('/', '-')}_{interval}.csv"
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)

    cached = None
    if path.exists() and not refresh:
        cached = pd.read_csv(path, index_col="time", parse_dates=["time"])
        cached.index = pd.DatetimeIndex(cached.index, tz="UTC") if cached.index.tz is None else cached.index

    client = make_exchange(exchange)
    # The first cached bar opens at most one bar after the requested start.
    covers_start = cached is not None and len(cached) and cached.index[0] <= start + timedelta(minutes=bar_minutes(interval))
    if covers_start:
        new = fetch_bars(client, symbol, interval, cached.index[-1].to_pydatetime(), now)
        bars = pd.concat([cached, new])
    else:
        bars = fetch_bars(client, symbol, interval, start, now)
    bars = bars[~bars.index.duplicated(keep="last")].sort_index()

    path.parent.mkdir(parents=True, exist_ok=True)
    bars.to_csv(path)
    return bars[bars.index >= start]
