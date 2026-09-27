# Crypto Trader

A crypto analysis and backtesting app: technical indicators, ML forecasts (LSTM and XGBoost), and a walk-forward backtester that has to reject a strategy before it will approve one.

## Features

- OHLCV data from public exchange endpoints via [ccxt](https://github.com/ccxt/ccxt) (Binance, Binance US, Coinbase, OKX, Bybit). No API key needed.
- Technical analysis: moving averages, RSI, Bollinger Bands, support and resistance
- ML forecasts with LSTM-with-attention and XGBoost models
- Backtest mode, which puts every strategy through three gates:
  1. **Critic**: automated checks for look-ahead (the signal is recomputed on truncated history and must not change), costs, data gaps, regime coverage, and implausibly high Sharpe
  2. **Deflated Sharpe** (Bailey & López de Prado): corrects for how many variations you tried
  3. **Walk-forward**: parameters are fitted on a trailing window and traded on the next one, rolled across the whole history
- Position sizing by fixed fractional risk

## Project Structure

```
stocktrader/
├── app.py                 # Streamlit application
├── backtest/
│   ├── engine.py          # Vectorized backtest: next-bar execution, fees and slippage on turnover
│   ├── metrics.py         # Sharpe, CAGR, drawdown depth and duration
│   ├── stats.py           # Deflated Sharpe ratio
│   ├── walkforward.py     # Rolling fit/trade folds
│   ├── critic.py          # Leakage and sanity checks
│   ├── regimes.py         # Bull/bear/chop split on a 200-bar moving average
│   ├── strategies.py      # SMA crossover, RSI mean reversion, XGBoost forecast
│   ├── risk.py            # Position sizing and live health check
│   └── validate.py        # Runs the three gates
├── models/
│   ├── lstm_model.py      # LSTM model implementation
│   ├── xgboost_model.py   # XGBoost model implementation
│   ├── trainer.py         # Model training utilities
│   └── utils.py           # Model-specific utilities
├── utils/
│   ├── __init__.py        # Exchange data and technical indicators
│   └── plotting.py        # Visualization utilities
└── tests/                 # pytest suite, runs on synthetic data
```

## Installation

```bash
git clone https://github.com/sahand68/stocktrader.git
cd stocktrader
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

Open `http://localhost:8501`, pick an exchange, a symbol such as `BTC/USDT`, a period and an interval.

binance.com is geo-blocked in some countries, including the US. Use `binanceus` or `coinbaseexchange` there (Coinbase quotes most coins in USD, e.g. `BTC/USD`).

### Backtest mode

Choose a strategy, costs, and the train/test window lengths in the sidebar. The app runs a walk-forward test and shows the gates, the out-of-sample equity curve against buy and hold, the per-fold results, the critic's checks, and a per-regime breakdown.

Every distinct configuration you run in a session is counted as a trial for the deflated Sharpe. Enter variations you tried before the session in the sidebar too, since understating them only fools you.

## Tests

```bash
pytest
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to all contributors
- Inspired by various trading strategies and ML implementations
