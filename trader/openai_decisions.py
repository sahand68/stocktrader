"""
OpenAI's Decisions API as a per-bar direction model, run the same way as Jev so the
two compare cleanly: same labels, same threshold rule, same cache and pacing.

The evidence can be Jev's anonymized state, a candlestick chart of recent bars, or both.
Charts are drawn from bars rebased so the last close is 100, with no time axis, so they
carry no price level or date either.
"""
import asyncio
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from openai import AsyncOpenAI

from trader.jev import JevOracle, direction_question

EVIDENCE = ("state", "chart", "state-chart")
CHART_BARS = 96
CHART_VERSION = 1   # part of the cache key: bump when the drawing changes


def decision_question(interval: str, evidence: str) -> dict:
    """Jev's question and labels, reworded only where the evidence differs."""
    jev = direction_question(interval)
    chart = (f"The image is a candlestick chart of the last {CHART_BARS} closed {interval} bars of a liquid crypto "
             "market, with relative volume underneath. Prices are rebased so the last close is 100 and the time "
             "axis is unlabeled.")
    ask = "Will the next bar close higher or lower than the last close?"
    instructions = {"state": jev.instructions, "chart": f"{chart} {ask}", "state-chart": f"{chart} {jev.instructions}"}
    return {
        "type": "choice",
        "name": "direction",
        "instructions": instructions[evidence],
        "choices": [{"value": label, "description": text} for label, text in jev.criteria.items()],
    }


def chart_window(ohlc: np.ndarray, volume: np.ndarray, end: int) -> list[list[float]]:
    """The CHART_BARS bars ending at row `end`, rebased to a last close of 100, volume relative to the window mean."""
    window = ohlc[end - CHART_BARS + 1 : end + 1] * (100 / ohlc[end, 3])
    vol = volume[end - CHART_BARS + 1 : end + 1]
    vol = vol / (vol.mean() or 1)
    return np.column_stack([window, vol]).round(2).tolist()


def render_chart(rows: list[list[float]]) -> bytes:
    """Candles and volume, each drawn as one line collection: patches per bar render ~10x slower."""
    o, h, l, c, v = np.asarray(rows).T
    x = np.arange(len(c))
    colors = np.where(c >= o, "#26a69a", "#ef5350")
    fig = Figure(figsize=(8, 5), dpi=100)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.97, bottom=0.03, hspace=0.05)
    price, vol = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    price.vlines(x, l, h, colors=colors, linewidth=1)
    price.vlines(x, np.minimum(o, c), np.maximum(o, c) + 1e-3, colors=colors, linewidth=4)
    price.axhline(100, color="#888", linewidth=0.6, linestyle=":")
    price.set_ylabel("price (last close = 100)")
    vol.vlines(x, 0, v, colors=colors, linewidth=4)
    vol.set_ylim(bottom=0)
    vol.set_ylabel("rel. volume")
    vol.set_xticks([])
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    return buf.getvalue()


class OpenAIOracle(JevOracle):
    """
    Decisions API calls behind JevOracle's cache, pacing and concurrency.
    A refused question comes back as empty probabilities, which every strategy reads as no edge.
    """

    label = "openai"
    price_per_million_input_tokens = None   # not published for the Decisions API yet

    def __init__(
        self,
        model: str = "gpt-6-luna",
        cache_path: Path = Path(".trader/openai_cache.sqlite"),
        concurrency: int = 16,
        requests_per_minute: float = 500,
        client_factory=None,
        progress: bool = True,
    ):
        super().__init__(model, cache_path, concurrency, requests_per_minute,
                         client_factory or (lambda: AsyncOpenAI(max_retries=6)), progress)
        self.refusals = 0

    def key(self, question: dict, state: dict) -> str:
        payload = {"api": "openai-decisions", "model": self.model, "question": question, "state": state,
                   "chart_version": CHART_VERSION}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def describe_usage(self) -> str:
        return f"{super().describe_usage()}, {self.refusals} refused"

    @staticmethod
    def content(evidence: dict) -> list[dict]:
        parts = []
        if "state" in evidence:
            parts.append({"type": "input_text", "text": json.dumps(evidence["state"])})
        if "bars" in evidence:
            png = base64.b64encode(render_chart(evidence["bars"])).decode()
            parts.append({"type": "input_image", "image_url": f"data:image/png;base64,{png}", "detail": "high"})
        return parts

    async def _query(self, client, question, state) -> tuple[dict[str, float], str, int]:
        content = await asyncio.to_thread(self.content, state)   # drawing a chart blocks; keep it off the loop
        response = await client.decisions.create(
            model=self.model, input=[{"role": "user", "content": content}], questions=[question]
        )
        answer = response.answers[0]
        if answer.type != "choice":
            self.refusals += 1
            return {}, response.model, response.usage.input_tokens
        return {p.value: p.probability for p in answer.probabilities}, response.model, response.usage.input_tokens
