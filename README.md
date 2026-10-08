# trader

Crypto trading signals from [TypeSafe Jev](https://typesafe.ai), put through a walk-forward backtest that has to reject a strategy before it will approve one, then paper traded on live data.

Nothing here sends orders. Paper trading uses live bars and live bid/ask quotes and simulates the fills.

## How Jev is used

After every closed bar, the bar history is turned into an anonymized state and Jev answers one `Choice` question:

```json
{
  "bar_interval": "1h",
  "returns_pct": {"last_bar": 0.21, "last_5_bars": -0.64, "last_20_bars": 1.9},
  "recent_bar_returns_pct_oldest_first": [0.1, -0.3, 0.05, 0.4, -0.2, 0.0, -0.5, 0.3, 0.1, 0.21],
  "volatility": {"per_bar_pct_20_bars": 0.48, "short_vs_long_ratio": 1.12},
  "trend_z": {"vs_sma_20": 0.8, "vs_sma_50": 1.9, "vs_sma_200": 3.4},
  "rsi_14": 57.3,
  "position_in_20_bar_range": 0.71,
  "volume_z_20": -0.4
}
```

The question asks whether the next bar closes `up`, `down`, or is `unclear`. Jev returns calibrated probabilities, and code goes long when `p_up - p_down` clears a threshold (and short past the mirror threshold, if shorts are enabled). The threshold is the only fitted parameter, and walk-forward refits it on each training window.

The state has no prices, dates or symbol. A model trained on market history could otherwise recognize the episode and "predict" what it remembers. The critic checks this on every run.

Answers are cached in `.trader/jev_cache.sqlite`, keyed by model, question and state, so rerunning a backtest costs nothing. At the listed $0.042 per million input tokens, a first run over two years of hourly bars is about 17,000 calls and roughly $0.25. Requests are paced to 1,000 per minute to stay under the account rate limit, so that first run takes about 18 minutes.

## OpenAI Decisions as a challenger

OpenAI's Decisions API answers the same `Choice` question with the same labels, threshold rule, cache and request pacing, so any difference comes from the model and what it sees. It also accepts images, so there are three variants:

| strategy | evidence |
|---|---|
| `openai-state` | Jev's JSON state, unchanged |
| `openai-chart` | a candlestick chart of the last 96 bars with volume, rebased so the last close is 100, no time axis |
| `openai-state-chart` | both |

All of them answer on exactly the bars Jev answers. The charts pass the same anonymity and look-ahead checks as the state. Refused questions read as no edge, so the strategy stays flat on those bars. Answers are cached in `.trader/openai_cache.sqlite`.

`trader compare` scores models on the next bar before any threshold or cost: hit rate when a model makes a call, rank IC of `p_up - p_down` against the next return, and a paired McNemar test against the first model listed. The cost gate is strict on hourly bars, so this is where a difference in forecasting skill shows up first.

## The three gates

`trader backtest` approves a strategy only if all three pass:

1. **The critic finds no leakage.**
   - Look-ahead: inputs and positions are recomputed on truncated history and must not change.
   - Anonymity: states must be identical after doubling every price and shifting every date by ten years.
   - Also checked: costs, data gaps, whether the test period covers both bull and bear phases, a Sharpe above 3 (fail), and whether answers came from more than one Jev version.
2. **The deflated Sharpe exceeds 0.95.** This is Bailey & López de Prado's correction for how many variations were tried. Every distinct configuration you run on a dataset is logged in `.trader/trials.jsonl` and counted automatically.
3. **It survives walk-forward.** Out-of-sample Sharpe must be above zero, with at least 60% of folds positive.

## Setup

```bash
pip install -e ".[dev]"
export TYPESAFE_API_KEY=...        # from console.typesafe.ai
export OPENAI_API_KEY=...          # only for openai-* strategies; the Decisions API is in limited preview
```

Market data comes from public exchange endpoints via [ccxt](https://github.com/ccxt/ccxt), so no exchange key is needed. binance.com is geo-blocked in some countries, including the US. Use `--exchange binanceus` or `--exchange coinbaseexchange --symbol BTC/USD` there.

## Usage

```bash
trader fetch --symbol BTC/USDT --interval 1h --days 730

trader backtest --strategy jev --symbol BTC/USDT --interval 1h --days 730
trader backtest --strategy sma-trend          # a baseline Jev has to beat
trader backtest --strategy jev --placebo      # same pipeline on a random walk; should be rejected
trader backtest --strategy openai-chart       # the challenger, through the same gates

trader compare --strategies jev@jev-1.13.0,openai-state,openai-chart

trader paper reports/jev_BTC-USDT_1h_<timestamp>.json
```

`backtest` prints the gates, out-of-sample metrics against buy-and-hold, each walk-forward fold and a per-regime breakdown. It also writes a JSON report.

`paper` only runs an approved report (`--force` overrides this). It uses the threshold fitted on the most recent training window, trades once per closed bar, and journals every decision to `.trader/paper/journal.jsonl`. Its state survives restarts. It flattens and stops if drawdown passes 1.5x the backtest's worst, or if live Sharpe over a fold-length window falls below the worst backtest fold. Both limits are fixed before the first trade.

Pin a Jev version (`--model jev-1.13.0`) once you rely on a threshold. Aliases such as `jev-latest` move, and a threshold calibrated on one version may not hold on the next.

## Tests

```bash
pytest
```

The tests need no network access. Jev and Decisions calls go through the real `typesafe-sdk` and `openai` clients against in-process mock servers, and exchange calls go to a fake ccxt exchange.
