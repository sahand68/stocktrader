"""
TypeSafe Jev as a per-bar direction model.

Each closed bar becomes an anonymized state, Jev answers one Choice question
(up / down / unclear) with calibrated probabilities, and code decides what to do
with them. Answers are cached on disk by (model, question, state), so a rerun of
the same backtest costs nothing and returns identical numbers.
"""
import asyncio
import hashlib
import json
import sqlite3
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from typesafe_sdk import AsyncTypeSafeClient, Choice, RetryPolicy

from trader.features import RECENT_BARS

PRICE_PER_MILLION_INPUT_TOKENS = 0.042   # listed price; output tokens are free
REQUESTS_PER_MINUTE = 1000                # under the listed 1,200/min account limit
LABELS = ("up", "down", "unclear")


def direction_question(interval: str) -> Choice:
    return Choice(
        instructions=(
            f"The state summarizes the most recent closed {interval} bars of a liquid crypto market: "
            "returns, volatility, distance from moving averages in volatility units, RSI, "
            "position in the recent range and relative volume. "
            "Will the next bar close higher or lower than the last close?"
        ),
        criteria={
            "up": "The next bar is more likely to close higher.",
            "down": "The next bar is more likely to close lower.",
            "unclear": "The indicators give no directional read.",
        },
    )


def _r(x: float, digits: int = 2) -> float:
    return round(float(x), digits)


def make_state(row: pd.Series, interval: str) -> dict:
    """One bar's features as Jev state. Percentages and z-scores only: no prices, dates or symbol."""
    return {
        "bar_interval": interval,
        "returns_pct": {
            "last_bar": _r(row["ret_1"] * 100),
            "last_5_bars": _r(row["ret_5"] * 100),
            "last_20_bars": _r(row["ret_20"] * 100),
        },
        "recent_bar_returns_pct_oldest_first": [_r(row[f"ret_lag{k}"] * 100) for k in reversed(range(RECENT_BARS))],
        "volatility": {
            "per_bar_pct_20_bars": _r(row["vol_20"] * 100),
            "short_vs_long_ratio": _r(row["vol_ratio"]),
        },
        "trend_z": {
            "vs_sma_20": _r(row["z_sma20"]),
            "vs_sma_50": _r(row["z_sma50"]),
            "vs_sma_200": _r(row["z_sma200"]),
        },
        "rsi_14": _r(row["rsi_14"], 1),
        "position_in_20_bar_range": _r(row["range_pos_20"]),
        "volume_z_20": _r(row["volume_z_20"]),
    }


@dataclass(frozen=True)
class Answer:
    probabilities: dict[str, float]
    model: str             # versioned model that answered, not the alias requested
    input_tokens: int
    latency_ms: float | None = None   # None when served from cache

    @property
    def edge(self) -> float:
        return self.probabilities.get("up", 0.0) - self.probabilities.get("down", 0.0)


class AnswerCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

    def get_many(self, keys: list[str]) -> dict[str, Answer]:
        found = {}
        for i in range(0, len(keys), 500):
            chunk = keys[i : i + 500]
            marks = ",".join("?" * len(chunk))
            for key, value in self.db.execute(f"SELECT key, value FROM answers WHERE key IN ({marks})", chunk):
                found[key] = Answer(**{**json.loads(value), "latency_ms": None})
        return found

    def put(self, key: str, answer: Answer) -> None:
        value = {"probabilities": answer.probabilities, "model": answer.model, "input_tokens": answer.input_tokens}
        self.db.execute("INSERT OR REPLACE INTO answers VALUES (?, ?)", (key, json.dumps(value)))

    def commit(self) -> None:
        self.db.commit()


class JevOracle:
    """
    Ask Jev about many states concurrently, with a disk cache in front.

    client_factory builds the AsyncTypeSafeClient; tests pass one with a mock transport.
    Subclasses for other decision APIs override `_query` and `key`.
    """

    label = "jev"
    price_per_million_input_tokens: float | None = PRICE_PER_MILLION_INPUT_TOKENS

    def __init__(
        self,
        model: str = "jev-latest",
        cache_path: Path = Path(".trader/jev_cache.sqlite"),
        concurrency: int = 16,
        requests_per_minute: float = REQUESTS_PER_MINUTE,
        client_factory: Callable[[], AsyncTypeSafeClient] | None = None,
        progress: bool = True,
    ):
        self.model = model
        self.cache = AnswerCache(cache_path)
        self.concurrency = concurrency
        self.min_gap = 60 / requests_per_minute
        self.client_factory = client_factory or (
            lambda: AsyncTypeSafeClient(model=model, retry=RetryPolicy(max_retries=6, backoff_max=30.0))
        )
        self.progress = progress
        self.calls = 0
        self.input_tokens = 0

    def key(self, question: Choice, state: dict) -> str:
        payload = {"model": self.model, "question": question.model_dump(), "state": state}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    @property
    def cost_usd(self) -> float | None:
        if self.price_per_million_input_tokens is None:
            return None
        return self.input_tokens * self.price_per_million_input_tokens / 1e6

    def describe_usage(self) -> str:
        cost = "" if self.cost_usd is None else f", ${self.cost_usd:.4f}"
        return f"{self.label}: {self.calls} new calls, {self.input_tokens:,} input tokens{cost}"

    def ask(self, question: Choice, states: list[dict]) -> list[Answer]:
        keys = [self.key(question, s) for s in states]
        answers = self.cache.get_many(list(set(keys)))
        missing = {k: s for k, s in zip(keys, states) if k not in answers}
        if missing:
            answers.update(asyncio.run(self._ask_remote(question, missing)))
        return [answers[k] for k in keys]

    async def _ask_remote(self, question: Choice, states: dict[str, dict]) -> dict[str, Answer]:
        limit = asyncio.Semaphore(self.concurrency)
        pace = asyncio.Lock()
        next_start = [0.0]
        done: dict[str, Answer] = {}

        async def wait_turn() -> None:
            """Space request starts min_gap apart so bursts stay under the rate limit."""
            async with pace:
                delay = next_start[0] - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                next_start[0] = max(next_start[0], time.monotonic()) + self.min_gap

        async with self.client_factory() as client:
            async def one(key: str, state: dict) -> None:
                async with limit:
                    await wait_turn()
                    started = time.perf_counter()
                    probabilities, model, tokens = await self._query(client, question, state)
                    latency = (time.perf_counter() - started) * 1000
                result = Answer(probabilities, model, tokens, latency)
                self.cache.put(key, result)
                self.calls += 1
                self.input_tokens += tokens
                done[key] = result
                if self.progress and len(done) % max(1, len(states) // 10) == 0:
                    print(f"  {self.label}: {len(done)}/{len(states)} answered", file=sys.stderr)

            try:
                await asyncio.gather(*(one(k, s) for k, s in states.items()))
            finally:
                self.cache.commit()   # keep every answer already paid for, even if a call failed
        return done

    async def _query(self, client, question, state) -> tuple[dict[str, float], str, int]:
        """One request: (probability per label, versioned model that answered, input tokens)."""
        response = await client.system_one(state=state, questions={"direction": question})
        answer = response.choices["direction"]
        return dict(answer.probabilities), response.model, response.usage.input_tokens or 0
