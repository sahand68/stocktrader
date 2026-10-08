"""
Forecast skill of direction models on the same bars, before any threshold or cost.

The backtest asks whether a signal pays after fees. This asks the narrower question:
does one model read the next bar better than another? Every model is scored on the
bars all of them answered, so the comparison is paired.
"""
import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr


def skill(edges: pd.DataFrame, next_return: pd.Series) -> pd.DataFrame:
    """
    edges: p_up - p_down per bar, one column per model (NaN where a model was not asked).
    next_return: the return of the bar after each decision.

    A model "calls" a bar when its edge is non-zero. hit_rate is over called bars;
    ic is the rank correlation of edge with the next return over all bars, so it rewards
    confidence that tracks the size of the move, not just its sign. The first column is
    the reference for the paired test (McNemar on bars both models called).
    """
    data = edges.join(next_return.rename("ret"), how="inner").dropna()
    data = data[data["ret"] != 0]
    ret = data.pop("ret")
    up = ret > 0
    reference = data.columns[0]
    ref_hit = (data[reference] > 0) == up

    rows = []
    for name in data.columns:
        edge = data[name]
        called = edge != 0
        hit = (edge > 0) == up
        k, n = int((hit & called).sum()), int(called.sum())
        ic, ic_p = spearmanr(edge, ret) if edge.nunique() > 1 else (np.nan, np.nan)
        row = {
            "model": name,
            "bars": len(edge),
            "called": called.mean(),
            "hit_rate": k / n if n else np.nan,
            "hit_p": binomtest(k, n).pvalue if n else np.nan,
            "ic": ic,
            "ic_p": ic_p,
            "gross_bps_per_call": float((np.sign(edge) * ret)[called].mean() * 1e4) if n else np.nan,
        }
        if name != reference:
            both = called & (data[reference] != 0)
            wins = int((hit & ~ref_hit & both).sum())
            losses = int((~hit & ref_hit & both).sum())
            row["vs_ref_p"] = binomtest(wins, wins + losses).pvalue if wins + losses else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")
