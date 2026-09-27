"""
Paper trading: live bars and live quotes, simulated fills, no orders sent.

After each bar closes, the strategy decides a target position from the bars so far;
the difference is filled at the current ask (buying) or bid (selling) plus fees.
The kill conditions come from the approved backtest and are fixed before the first
trade, because nobody is objective at the moment they fire.
"""
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd
from typesafe_sdk import TypeSafeAPIConnectionError, TypeSafeAPIError

from trader.backtest import Costs, sharpe
from trader.data import bar_minutes, fetch_bars

# Worth retrying next bar; anything else is a bug and should stop the loop.
TRANSIENT = (ccxt.NetworkError, TypeSafeAPIConnectionError, TypeSafeAPIError)


@dataclass
class Account:
    cash: float
    units: float = 0.0
    peak_equity: float = 0.0
    last_bar: str | None = None
    halted: str | None = None
    equity_history: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class KillSwitch:
    max_drawdown: float     # negative fraction, e.g. -0.3
    min_sharpe: float       # rolling live Sharpe floor
    window_bars: int        # bars the Sharpe floor is judged over

    @classmethod
    def from_report(cls, report: dict) -> "KillSwitch":
        folds = report["folds"]
        return cls(
            max_drawdown=1.5 * report["oos"]["max_drawdown"],
            min_sharpe=min(f["sharpe"] for f in folds),
            window_bars=int(np.median([f["bars"] for f in folds])),
        )

    def check(self, equity: list[float], peak: float, periods_per_year: float) -> str | None:
        e = np.asarray(equity)
        drawdown = e[-1] / peak - 1
        if drawdown < self.max_drawdown:
            return f"drawdown {drawdown:.1%} beyond the {self.max_drawdown:.1%} limit"
        if len(e) > self.window_bars:
            recent = np.diff(e[-self.window_bars - 1:]) / e[-self.window_bars - 1:-1]
            live = sharpe(pd.Series(recent), periods_per_year)
            if live < self.min_sharpe:
                return f"live Sharpe {live:.2f} over {self.window_bars} bars is below the worst backtest fold ({self.min_sharpe:.2f})"
        return None


class PaperTrader:
    def __init__(self, client, symbol: str, interval: str, strategy, params: dict, costs: Costs,
                 kill: KillSwitch, periods_per_year: float, state_path: Path, journal_path: Path,
                 capital: float = 10_000, warmup_bars: int = 400):
        self.client = client
        self.symbol = symbol
        self.interval = interval
        self.strategy = strategy
        self.params = params
        self.costs = costs
        self.kill = kill
        self.periods_per_year = periods_per_year
        self.state_path = state_path
        self.journal_path = journal_path
        self.warmup = timedelta(minutes=bar_minutes(interval) * warmup_bars)
        self.account = (Account(**json.loads(state_path.read_text())) if state_path.exists()
                        else Account(cash=capital, peak_equity=capital))

    def step(self, now: datetime | None = None) -> dict | None:
        """Act on the newest closed bar if it hasn't been seen. Returns the journal entry."""
        if self.account.halted:
            return None
        now = now or datetime.now(timezone.utc)
        bars = fetch_bars(self.client, self.symbol, self.interval, now - self.warmup, now)
        if bars.empty or (self.account.last_bar and bars.index[-1].isoformat() <= self.account.last_bar):
            return None

        started = time.perf_counter()
        inputs = self.strategy.inputs(bars)
        target = float(self.strategy.positions(inputs, self.params).iloc[-1])
        decision_ms = (time.perf_counter() - started) * 1000

        quote = self.client.fetch_ticker(self.symbol)
        # Some exchanges leave bid/ask empty in the ticker; fall back to the last trade.
        bid, ask = quote.get("bid") or quote["last"], quote.get("ask") or quote["last"]
        mid = (bid + ask) / 2
        acct = self.account
        equity = acct.cash + acct.units * mid
        acct.equity_history = (acct.equity_history + [equity])[-(self.kill.window_bars + 1):]
        acct.peak_equity = max(acct.peak_equity, equity)

        reason = self.kill.check(acct.equity_history, acct.peak_equity, self.periods_per_year)
        if reason:
            acct.halted = reason
            target = 0.0

        fill = None
        delta = target * equity / mid - acct.units
        if abs(delta * mid) > 1e-6 * equity:
            price = ask if delta > 0 else bid
            notional = delta * price
            fee = abs(notional) * self.costs.fee_bps / 1e4
            acct.cash -= notional + fee
            acct.units += delta
            fill = {"units": delta, "price": price, "fee": fee}

        acct.last_bar = bars.index[-1].isoformat()
        entry = {
            "bar": acct.last_bar,
            "decided_at": now.isoformat(),
            "decision_ms": round(decision_ms, 1),
            "target": target,
            "inputs": {k: v for k, v in inputs.iloc[-1].items() if isinstance(v, (int, float, str))},
            "bid": bid,
            "ask": ask,
            "fill": fill,
            "equity": acct.cash + acct.units * mid,
            "halted": acct.halted,
        }
        self._save(entry)
        return entry

    def _save(self, entry: dict) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(asdict(self.account)))
        with self.journal_path.open("a") as f:
            f.write(json.dumps(entry, default=float) + "\n")

    def run(self, max_bars: int | None = None, on_entry=print, max_consecutive_errors: int = 10) -> None:
        bar = timedelta(minutes=bar_minutes(self.interval))
        seen = errors = 0
        while not self.account.halted and (max_bars is None or seen < max_bars):
            try:
                entry = self.step()
                errors = 0
            except TRANSIENT as e:
                # The position stays as it is; the next bar tries again.
                errors += 1
                on_entry({"error": f"{type(e).__name__}: {e}", "consecutive": errors})
                if errors >= max_consecutive_errors:
                    raise
                entry = None
            if entry:
                seen += 1
                on_entry(entry)
            if self.account.halted or (max_bars is not None and seen >= max_bars):
                break
            now = datetime.now(timezone.utc)
            epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
            next_close = epoch + ((now - epoch) // bar + 1) * bar
            time.sleep(max(1.0, (next_close - now).total_seconds() + 3))
        if self.account.halted:
            on_entry({"halted": self.account.halted})
