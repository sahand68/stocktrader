import streamlit as st
from utils import (
    get_crypto_data,
    bars_per_day,
    add_technical_indicators,
    find_support_resistance_levels,
    determine_trend
)
from utils.plotting import create_candlestick_plot, create_prediction_figure
from models.trainer import ModelTrainer
from models.utils import get_bullish_bearish_confidence
from backtest.engine import Config, periods_per_year
from backtest.risk import losing_streak_drawdown, position_size
from backtest.strategies import STRATEGIES
from backtest.validate import validate
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
import io
import plotly.graph_objects as go

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a string buffer to capture logs
log_buffer = io.StringIO()
log_handler = logging.StreamHandler(log_buffer)
log_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_handler.setFormatter(formatter)
logging.getLogger().addHandler(log_handler)

st.set_page_config(page_title="Crypto Analysis App", layout="wide")


def fmt_price(x):
    """Dollar price with enough digits for sub-cent coins."""
    return f"${x:,.2f}" if abs(x) >= 1 else f"${x:.6g}"


# Title and description
st.title("📈 Crypto Analysis App")
st.markdown("""
Technical analysis, ML forecasts and walk-forward backtests on crypto markets.
Data comes from public exchange endpoints, so no API key is needed.
""")

# Sidebar inputs
st.sidebar.header("Input Parameters")

# Analysis mode selection
analysis_mode = st.sidebar.radio(
    "Select Analysis Mode",
    ["Technical Analysis", "ML Forecast", "Both", "Backtest"]
)

exchange = st.sidebar.selectbox(
    "Exchange",
    ["binance", "binanceus", "coinbaseexchange", "okx", "bybit"],
    help="binance.com is geo-blocked in some countries (including the US); use binanceus or coinbaseexchange there."
)
ticker = st.sidebar.text_input("Symbol", value="BTC/USDT", help="Exchange symbol, e.g. BTC/USDT or ETH/USD").upper()

# Time period selection
period_options = {
    "1 Day": "1d",
    "5 Days": "5d",
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "Year to Date": "ytd",
    "1 Year": "1y",
    "2 Years": "2y",
    "5 Years": "5y",
    "10 Years": "10y",
    "All Time": "max"
}

selected_period = st.sidebar.select_slider(
    "Select Time Period",
    options=list(period_options.keys()),
    value="1 Year"  # Changed default to 1 Year for better ML training
)
period = period_options[selected_period]

# Interval selection based on period
if selected_period in ["1 Day", "5 Days"]:
    interval_options = ["1m", "5m", "15m", "30m", "1h"]
    default_interval = "5m"
else:
    interval_options = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
    default_interval = "1h"

interval = st.sidebar.selectbox(
    "Select Interval",
    interval_options,
    index=interval_options.index(default_interval)
)

# ML Model Settings in Sidebar - Only show if ML is selected
if analysis_mode in ["ML Forecast", "Both"]:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ML Model Settings")

    model_type = st.sidebar.radio(
        "Select Model Type",
        ["lstm_attention", "xgboost"],
        format_func=lambda x: "LSTM" if x == "lstm_attention" else "XGBoost"
    )

    forecast_days = st.sidebar.radio(
        "Forecast Horizon",
        [1, 3],
        format_func=lambda x: f"{x} Day{'s' if x > 1 else ''} Ahead"
    )

if analysis_mode == "Backtest":
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Backtest Settings")
    strategy_name = st.sidebar.selectbox("Strategy", list(STRATEGIES))
    allow_short = st.sidebar.checkbox("Allow shorts", value=False)
    fee_bps = st.sidebar.number_input("Fee per side (bps)", min_value=0.0, value=10.0, step=1.0)
    slippage_bps = st.sidebar.number_input("Slippage per side (bps)", min_value=0.0, value=5.0, step=1.0)
    train_days = st.sidebar.number_input("Train window (days)", min_value=7, value=180, step=10)
    test_days = st.sidebar.number_input("Test window (days)", min_value=1, value=30, step=5)
    prior_trials = st.sidebar.number_input(
        "Variations tried before this session", min_value=0, value=0, step=1,
        help="Every strategy, parameter or setting you have already looked at on this data. "
             "Understating it only fools you."
    )

# Add date range info
st.sidebar.markdown("---")
st.sidebar.markdown("### Selected Range Info")

# Display approximate date range based on period
end_date = datetime.now().strftime("%Y-%m-%d")
if period == "max":
    date_info = "From the beginning to present"
elif period == "ytd":
    start = datetime(datetime.now().year, 1, 1).strftime("%Y-%m-%d")
    date_info = f"From {start} to present"
elif period == "1d":
    date_info = "Last 24 hours"
elif period == "5d":
    date_info = "Last 5 days"
elif period == "1mo":
    date_info = "Last 30 days"
elif period == "3mo":
    date_info = "Last 90 days"
elif period == "6mo":
    date_info = "Last 180 days"
elif period == "1y":
    date_info = "Last 365 days"
elif period == "2y":
    date_info = "Last 2 years"
elif period == "5y":
    date_info = "Last 5 years"
elif period == "10y":
    date_info = "Last 10 years"

st.sidebar.markdown(f"**Time Range:** {date_info}")
st.sidebar.markdown(f"**Current Date:** {end_date}")


def render_backtest(df):
    """Walk-forward backtest with the three gates: critic, deflated Sharpe, walk-forward."""
    cfg = Config(fee_bps=fee_bps, slippage_bps=slippage_bps, periods_per_year=periods_per_year(interval))
    strategy = STRATEGIES[strategy_name](allow_short=allow_short)
    train_bars = int(train_days * bars_per_day(interval))
    test_bars = int(test_days * bars_per_day(interval))

    # Every distinct configuration looked at this session counts as a trial.
    tried = st.session_state.setdefault("tried_configs", set())
    tried.add((exchange, ticker, interval, period, strategy_name, allow_short,
               fee_bps, slippage_bps, train_days, test_days))
    n_trials = int(prior_trials) + len(tried)

    with st.spinner(f"Walk-forward testing {strategy_name}..."):
        report = validate(df, strategy, cfg, interval, train_bars, test_bars, n_trials)

    wf = report.walk_forward
    m = report.oos_metrics

    st.subheader(f"{strategy_name}: {'APPROVED' if report.approved else 'REJECTED'}")
    for gate, passed in report.gates.items():
        (st.success if passed else st.error)(f"{'✅' if passed else '❌'} {gate}")

    cols = st.columns(5)
    cols[0].metric("OOS Sharpe", f"{m['sharpe']:.2f}")
    cols[1].metric("CAGR", f"{m['cagr']:.1%}")
    cols[2].metric("Max drawdown", f"{m['max_drawdown']:.1%}")
    cols[3].metric("Longest drawdown", f"{m['longest_dd_days']:.0f} days")
    cols[4].metric("Positive folds", f"{wf.positive_folds}/{len(wf.folds)}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wf.oos.index, y=wf.oos["equity"], name="Strategy (after costs)"))
    oos_close = df["Close"].loc[wf.oos.index]
    fig.add_trace(go.Scatter(x=oos_close.index, y=cfg.initial_capital * oos_close / oos_close.iloc[0],
                             name="Buy and hold", line=dict(dash="dot")))
    for start in wf.folds["start"]:
        fig.add_vline(x=start, line_width=1, line_dash="dot", opacity=0.3)
    fig.update_layout(title="Out-of-sample equity (dotted lines mark fold starts)",
                      yaxis_title="Equity ($)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Multiple-testing correction")
    dsr = report.dsr
    dcols = st.columns(3)
    dcols[0].metric("Deflated Sharpe probability", f"{dsr.probability:.3f}")
    dcols[1].metric("Best-of-noise Sharpe", f"{dsr.noise_benchmark:.2f}")
    dcols[2].metric("Trials counted", dsr.n_trials)
    st.caption("Probability that the true Sharpe beats what the best of this many random strategies would show. "
               "Needs to exceed 0.95.")

    st.subheader("Critic")
    st.dataframe(pd.DataFrame([vars(c) for c in report.checks]), use_container_width=True, hide_index=True)

    st.subheader("Walk-forward folds")
    st.caption("Judge on the worst fold and the share of positive folds, not the average.")
    st.dataframe(wf.folds, use_container_width=True, hide_index=True)

    st.subheader("By market regime (200-bar moving average)")
    st.dataframe(report.regimes, use_container_width=True)


def render_position_sizer():
    st.markdown("---")
    st.subheader("Position sizing")
    st.caption("Size for the path: a real edge still kills an account that bets too much per trade.")
    cols = st.columns(4)
    capital = cols[0].number_input("Capital ($)", min_value=1.0, value=10_000.0)
    entry = cols[1].number_input("Entry price", min_value=0.0, value=100.0, format="%.6g")
    stop = cols[2].number_input("Stop price", min_value=0.0, value=95.0, format="%.6g")
    risk_pct = cols[3].number_input("Risk per trade (%)", min_value=0.1, max_value=10.0, value=1.0) / 100
    if entry > 0 and entry != stop:
        size = position_size(capital, entry, stop, risk_pct)
        st.write(f"Buy **{size['units']:.6g}** units ({fmt_price(size['notional'])}, "
                 f"{size['pct_of_capital']:.1%} of capital); a stop-out loses {fmt_price(size['loss_if_stopped'])}.")
    st.write(f"Twelve losses in a row at this risk: **{losing_streak_drawdown(risk_pct, 12):.1%}** drawdown.")


if st.sidebar.button("Analyze"):
    try:
        # Clear previous logs
        log_buffer.truncate(0)
        log_buffer.seek(0)
        
        with st.spinner('Fetching market data...'):
            logger.info(f"Starting analysis for {ticker} on {exchange}")
            
            # Get market data first
            logger.info(f"Fetching market data with period={period} and interval={interval}")
            df = get_crypto_data(ticker, period, interval, exchange)
            
            if df.empty:
                logger.error("No data available")
                st.error(f"No data available for {ticker} on {exchange} with the selected period and interval.")
            else:
                logger.info(f"Successfully fetched {len(df)} rows of data")
                
                # Show basic market info
                st.subheader(f"{ticker} Analysis")
                
                # Calculate basic metrics
                current_price = df['Close'].iloc[-1]
                previous_close = df['Close'].iloc[-2] if len(df) > 1 else current_price
                price_change = ((current_price - previous_close) / previous_close) * 100
                avg_volume = df['Volume'].mean() if 'Volume' in df.columns else 'N/A'
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Current Price",
                        fmt_price(current_price),
                        f"{price_change:+.2f}%"
                    )
                with col2:
                    high_price = df['High'].max()
                    st.metric("Period High", fmt_price(high_price))
                with col3:
                    low_price = df['Low'].min()
                    st.metric("Period Low", fmt_price(low_price))

                # Add technical indicators if needed
                if analysis_mode in ["Technical Analysis", "Both"]:
                    df = add_technical_indicators(df)
                
                # Create layout based on selected mode
                if analysis_mode == "Technical Analysis":
                    # Single column layout for technical analysis
                    # Display candlestick chart
                    st.subheader("Interactive Price Chart")
                    fig = create_candlestick_plot(df, ticker)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Technical Analysis Section
                    st.subheader("Technical Analysis")
                    
                    # Display trend analysis
                    trend = determine_trend(df)
                    if trend == "Bullish":
                        st.success(f"Current Trend: {trend} 📈")
                    elif trend == "Bearish":
                        st.error(f"Current Trend: {trend} 📉")
                    else:
                        st.info(f"Current Trend: {trend} ↔️")
                    
                    # Display RSI
                    rsi = df['RSI'].iloc[-1]
                    st.metric("RSI (14)", f"{rsi:.2f}")
                    
                    # Support Levels
                    st.subheader("Support & Resistance Levels")
                    support_levels, resistance_levels = find_support_resistance_levels(df)
                    if support_levels or resistance_levels:
                        st.subheader("Support Levels")
                        for i, level in enumerate(support_levels[-3:], 1):
                            st.metric(f"Support Level {i}", fmt_price(level))
                        st.subheader("Resistance Levels")
                        for i, level in enumerate(resistance_levels[-3:], 1):
                            st.metric(f"Resistance Level {i}", fmt_price(level))
                    else:
                        st.write("No support/resistance levels found in the current timeframe.")
                    
                    # Additional Technical Metrics
                    with st.expander("Additional Technical Metrics"):
                        metrics_col1, metrics_col2 = st.columns(2)
                        with metrics_col1:
                            st.metric("SMA (20)", fmt_price(df['SMA_20'].iloc[-1]))
                            st.metric("Bollinger Upper", fmt_price(df['BB_upper'].iloc[-1]))
                        with metrics_col2:
                            st.metric("EMA (20)", fmt_price(df['EMA_20'].iloc[-1]))
                            st.metric("Bollinger Lower", fmt_price(df['BB_lower'].iloc[-1]))
                
                elif analysis_mode == "ML Forecast":
                    # Single column layout for ML forecast
                    # Add technical indicators before creating plot
                    df = add_technical_indicators(df)
                    
                    # Display candlestick chart
                    st.subheader("Interactive Price Chart")
                    fig = create_candlestick_plot(df, ticker)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate support and resistance levels
                    support_levels, resistance_levels = find_support_resistance_levels(df)
                    
                    # ML Model Section
                    st.subheader(f"ML Price Prediction ({forecast_days} Day{'s' if forecast_days > 1 else ''} Ahead)")
                    
                    predicted_price = None  # Initialize prediction variables
                    with st.spinner(f"Training {model_type} model..."):
                        try:
                            # Initialize model trainer with interval
                            trainer = ModelTrainer(model_type=model_type, forecast_days=forecast_days, interval=interval)
                            
                            logger.info(f"Training new {model_type} model for {ticker}")
                            
                            # First prepare the data
                            trainer.prepare_data(df)
                            
                            # Then train the model (no arguments needed)
                            trainer.train()
                            
                            # Get validation MSE from the model's training history
                            val_mse = trainer.best_val_loss if hasattr(trainer, 'best_val_loss') else None
                            
                            # Make prediction after training - returns list of (timestamp, price, sentiment, confidence) tuples
                            predictions = trainer.predict(df)
                            
                            # Get the final predicted values
                            final_timestamp, final_price, predicted_sentiment, model_confidence = predictions[-1]
                            
                            # Calculate prediction metrics for the final price
                            pred_change = ((final_price - current_price) / current_price) * 100
                            
                            # Use the model's sentiment and confidence directly
                            direction = predicted_sentiment
                            confidence = model_confidence

                            # Show prediction metrics in columns
                            pred_cols = st.columns(2)
                            with pred_cols[0]:
                                st.metric(
                                    f"Final Predicted Price ({forecast_days}d)",
                                    fmt_price(final_price),
                                    f"{pred_change:+.2f}%"
                                )
                                mse_display = f"{val_mse:.4f}" if val_mse is not None else "N/A"
                                st.metric("Model MSE", mse_display)
                            with pred_cols[1]:
                                st.metric("Prediction Direction", direction)
                                st.metric("Confidence Score", f"{confidence:.2f}")

                            # Create chart column for prediction visualization
                            chart_col = st.container()
                            
                            # Add prediction and support/resistance chart
                            with chart_col:
                                st.subheader("Prediction & Support/Resistance Levels")
                                
                                # Separate timestamps and prices for plotting
                                future_timestamps = [p[0] for p in predictions]  # Get timestamps
                                predicted_prices = [p[1] for p in predictions]   # Get prices
                                
                                pred_fig = create_prediction_figure(
                                    df, predicted_prices, future_timestamps, forecast_days,
                                    support_levels, resistance_levels, current_price,
                                    confidence, pred_change
                                )
                                st.plotly_chart(pred_fig, use_container_width=True)
                            
                        except Exception as model_error:
                            logger.error(f"Error in ML prediction: {str(model_error)}", exc_info=True)
                            st.error(f"Error in ML prediction: {str(model_error)}")
                
                elif analysis_mode == "Backtest":
                    render_backtest(df)

                else:  # Both
                    # Add technical indicators
                    df = add_technical_indicators(df)
                    
                    # Create two columns for the layout
                    chart_col, analysis_col = st.columns([2, 1])
                    
                    with chart_col:
                        # Display candlestick chart
                        st.subheader("Interactive Price Chart")
                        fig = create_candlestick_plot(df, ticker)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with analysis_col:
                        # Display trend analysis
                        st.subheader("Trend Analysis")
                        trend = determine_trend(df)
                        
                        if trend == "Bullish":
                            st.success(f"Current Trend: {trend} 📈")
                        elif trend == "Bearish":
                            st.error(f"Current Trend: {trend} 📉")
                        else:
                            st.info(f"Current Trend: {trend} ↔️")
                        
                        # Display RSI
                        rsi = df['RSI'].iloc[-1]
                        st.metric("RSI (14)", f"{rsi:.2f}")
                        
                        # Support Levels
                        st.subheader("Support & Resistance Levels")
                        support_levels, resistance_levels = find_support_resistance_levels(df)
                        if support_levels or resistance_levels:
                            st.subheader("Support Levels")
                            for i, level in enumerate(support_levels[-3:], 1):
                                st.metric(f"Support Level {i}", fmt_price(level))
                            st.subheader("Resistance Levels")
                            for i, level in enumerate(resistance_levels[-3:], 1):
                                st.metric(f"Resistance Level {i}", fmt_price(level))
                        else:
                            st.write("No support/resistance levels found in the current timeframe.")
                        
                        # ML Model Section
                        st.markdown("---")
                        st.subheader(f"ML Price Prediction ({forecast_days} Day{'s' if forecast_days > 1 else ''} Ahead)")
                        
                        predicted_price = None  # Initialize prediction variables
                        with st.spinner(f"Training {model_type} model..."):
                            try:
                                # Calculate support and resistance levels
                                support_levels, resistance_levels = find_support_resistance_levels(df)
                                
                                # Initialize model trainer with interval
                                trainer = ModelTrainer(model_type=model_type, forecast_days=forecast_days, interval=interval)
                                
                                logger.info(f"Training new {model_type} model for {ticker}")
                                
                                # First prepare the data
                                trainer.prepare_data(df)
                                
                                # Then train the model (no arguments needed)
                                trainer.train()
                                
                                # Get validation MSE from the model's training history
                                val_mse = trainer.best_val_loss if hasattr(trainer, 'best_val_loss') else None
                                
                                # Make prediction after training - returns list of (timestamp, price, sentiment, confidence) tuples
                                predictions = trainer.predict(df)
                                
                                # Get the final predicted values
                                final_timestamp, final_price, predicted_sentiment, model_confidence = predictions[-1]
                                
                                # Calculate prediction metrics for the final price
                                pred_change = ((final_price - current_price) / current_price) * 100
                                
                                # Use the model's sentiment and confidence directly
                                direction = predicted_sentiment
                                confidence = model_confidence

                                # Show prediction metrics in columns
                                pred_cols = st.columns(2)
                                with pred_cols[0]:
                                    st.metric(
                                        f"Final Predicted Price ({forecast_days}d)",
                                        fmt_price(final_price),
                                        f"{pred_change:+.2f}%"
                                    )
                                    mse_display = f"{val_mse:.4f}" if val_mse is not None else "N/A"
                                    st.metric("Model MSE", mse_display)
                                with pred_cols[1]:
                                    st.metric("Prediction Direction", direction)
                                    st.metric("Confidence Score", f"{confidence:.2f}")
                                
                                # Add prediction and support/resistance chart
                                with chart_col:
                                    st.subheader("Prediction & Support/Resistance Levels")
                                    
                                    # Separate timestamps and prices for plotting
                                    future_timestamps = [p[0] for p in predictions]  # Get timestamps
                                    predicted_prices = [p[1] for p in predictions]   # Get prices
                                    
                                    pred_fig = create_prediction_figure(
                                        df, predicted_prices, future_timestamps, forecast_days,
                                        support_levels, resistance_levels, current_price,
                                        confidence, pred_change
                                    )
                                    st.plotly_chart(pred_fig, use_container_width=True)
                                
                            except Exception as model_error:
                                logger.error(f"Error in ML prediction: {str(model_error)}", exc_info=True)
                                st.error(f"Error in ML prediction: {str(model_error)}")
                        
                        # Display additional metrics in expandable section
                        with st.expander("Additional Technical Metrics"):
                            metrics_col1, metrics_col2 = st.columns(2)
                            
                            with metrics_col1:
                                st.metric("SMA (20)", fmt_price(df['SMA_20'].iloc[-1]))
                                st.metric("Bollinger Upper", fmt_price(df['BB_upper'].iloc[-1]))
                            
                            with metrics_col2:
                                st.metric("EMA (20)", fmt_price(df['EMA_20'].iloc[-1]))
                                st.metric("Bollinger Lower", fmt_price(df['BB_lower'].iloc[-1]))
                
                # Display data summary
                with st.expander("Data Summary"):
                    start_time = df.index[0].strftime("%Y-%m-%d %H:%M")
                    end_time = df.index[-1].strftime("%Y-%m-%d %H:%M")
                    st.markdown(f"**Total data points:** {len(df)}")
                    st.markdown(f"**Date range:** {start_time} to {end_time}")
                    st.markdown(f"**Interval:** {interval}")

    except Exception as e:
        logger.error(f"Error analyzing {ticker}: {str(e)}", exc_info=True)
        st.error(f"Error analyzing {ticker}: {str(e)}")
        st.info("Check that the symbol is listed on the selected exchange and try again.")
    
    finally:
        # Display logs in an expander
        with st.expander("Debug Logs", expanded=True):
            st.text(log_buffer.getvalue())

if analysis_mode == "Backtest":
    render_position_sizer()

# Footer
st.markdown("---")
st.markdown("### About")
st.markdown("""
This app combines technical analysis, machine learning forecasts and walk-forward backtesting for crypto markets.
- **LSTM Model**: Deep learning model that captures long-term dependencies in time series data
- **XGBoost Model**: Gradient boosting model that excels at feature-based prediction
- **Forecast Horizon**: Choose between 1-day and 3-day ahead predictions
""") 