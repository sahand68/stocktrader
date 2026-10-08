"""
A strategy turns bars into target positions in two steps:

    inputs(bars)              per-bar inputs; row t may only use bars up to t
    positions(inputs, params) target position per bar in [-1, 1]

grid() lists the parameter sets that walk-forward chooses between on each
training window. Keeping the expensive, causal step (features, Jev calls) apart
from the cheap parameter step lets walk-forward refit thresholds without
re-querying anything.
"""
import numpy as np
import pandas as pd

from trader.features import build_features, ready
from trader.jev import LABELS, JevOracle, direction_question, make_state
from trader.openai_decisions import CHART_BARS, EVIDENCE, OpenAIOracle, chart_window, decision_question


class BuyAndHold:
    name = "buy-and-hold"

    def inputs(self, bars: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(index=bars.index)

    def grid(self) -> list[dict]:
        return [{}]

    def positions(self, inputs: pd.DataFrame, params: dict) -> pd.Series:
        return pd.Series(1.0, index=inputs.index)


class SmaTrend:
    """Long above the slow moving average's trend, flat (or short) below it."""
    name = "sma-trend"

    def __init__(self, fast=(10, 20, 50), slow=(100, 200), allow_short: bool = False):
        self.fast, self.slow, self.allow_short = fast, slow, allow_short

    def inputs(self, bars: pd.DataFrame) -> pd.DataFrame:
        close = bars["close"]
        return pd.DataFrame({f"sma{n}": close.rolling(n).mean() for n in {*self.fast, *self.slow}})

    def grid(self) -> list[dict]:
        return [{"fast": f, "slow": s} for f in self.fast for s in self.slow if f < s]

    def positions(self, inputs: pd.DataFrame, params: dict) -> pd.Series:
        fast, slow = inputs[f"sma{params['fast']}"], inputs[f"sma{params['slow']}"]
        pos = np.where(fast > slow, 1.0, -1.0 if self.allow_short else 0.0)
        return pd.Series(pos, index=inputs.index).where(slow.notna(), 0.0)


class JevDirection:
    """
    Jev's probability that the next bar closes up minus the probability it closes down.
    Long when that edge clears the threshold, short when it clears it the other way
    (if shorts are allowed), flat otherwise. The threshold is the only fitted parameter.
    """
    name = "jev"

    def __init__(self, oracle: JevOracle, interval: str,
                 thresholds=(0.0, 0.1, 0.2, 0.3, 0.4), allow_short: bool = False):
        self.oracle = oracle
        self.interval = interval
        self.thresholds = thresholds
        self.allow_short = allow_short
        self.question = direction_question(interval)

    def states(self, bars: pd.DataFrame) -> tuple[pd.Index, list[dict]]:
        features = build_features(bars)
        rows = features[ready(features)]
        return rows.index, [make_state(row, self.interval) for _, row in rows.iterrows()]

    def inputs(self, bars: pd.DataFrame) -> pd.DataFrame:
        index, states = self.states(bars)
        answers = self.oracle.ask(self.question, states)
        out = pd.DataFrame(
            [[a.probabilities.get(label, 0.0) for label in LABELS] + [a.model] for a in answers],
            columns=[f"p_{label}" for label in LABELS] + ["model"],
            index=index,
        )
        return out.reindex(bars.index)

    def grid(self) -> list[dict]:
        return [{"threshold": t} for t in self.thresholds]

    def positions(self, inputs: pd.DataFrame, params: dict) -> pd.Series:
        edge = (inputs["p_up"] - inputs["p_down"]).fillna(0.0)
        pos = np.where(edge > params["threshold"], 1.0, 0.0)
        if self.allow_short:
            pos = np.where(edge < -params["threshold"], -1.0, pos)
        return pd.Series(pos, index=inputs.index)


class OpenAIDirection(JevDirection):
    """
    JevDirection with OpenAI's Decisions API answering instead. `evidence` picks what it sees:
    Jev's state ("state"), a chart of the last CHART_BARS bars ("chart"), or both ("state-chart").
    It answers on exactly the bars Jev does, so the two can be compared bar for bar.
    """

    def __init__(self, oracle: OpenAIOracle, interval: str, evidence: str = "chart",
                 thresholds=(0.0, 0.1, 0.2, 0.3, 0.4), allow_short: bool = False):
        if evidence not in EVIDENCE:
            raise ValueError(f"evidence must be one of {EVIDENCE}")
        super().__init__(oracle, interval, thresholds, allow_short)
        self.name = f"openai-{evidence}"
        self.evidence = evidence
        self.question = decision_question(interval, evidence)

    def states(self, bars: pd.DataFrame) -> tuple[pd.Index, list[dict]]:
        features = build_features(bars)
        rows = features[ready(features)]
        assert len(rows) == 0 or bars.index.get_loc(rows.index[0]) >= CHART_BARS - 1
        ohlc = bars[["open", "high", "low", "close"]].to_numpy()
        volume = bars["volume"].to_numpy()
        evidence = []
        for t, (_, row) in zip(bars.index.get_indexer(rows.index), rows.iterrows()):
            e = {}
            if self.evidence != "chart":
                e["state"] = make_state(row, self.interval)
            if self.evidence != "state":
                e["bars"] = chart_window(ohlc, volume, t)
            evidence.append(e)
        return rows.index, evidence
