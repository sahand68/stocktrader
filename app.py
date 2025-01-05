import streamlit as st
import os
from dotenv import load_dotenv
from utils import (
    get_stock_data,
    add_technical_indicators,
    find_support_resistance_levels,
    determine_trend
)
from utils.plotting import create_candlestick_plot, create_prediction_figure
from models.trainer import ModelTrainer
from models.utils import get_bullish_bearish_confidence
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
import io
import plotly.graph_objects as go

# Load environment variables
load_dotenv()

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

st.set_page_config(page_title="Stock Analysis App", layout="wide")

# Check for API key
if not os.getenv('POLYGON_API_KEY'):
    st.error("Please set your Polygon API key in the .env file as POLYGON_API_KEY")
    st.stop()

# Title and description
st.title("📈 Stock Analysis App")
st.markdown("""
This app performs real-time stock analysis using technical indicators and ML models.
Enter a stock ticker and select your preferred timeframe to get started!
""")

# Sidebar inputs
st.sidebar.header("Input Parameters")

# Analysis mode selection
analysis_mode = st.sidebar.radio(
    "Select Analysis Mode",
    ["Technical Analysis", "ML Forecast", "Both"]
)

# Stock ticker input
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").upper()

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
    interval_options = ["1m", "5m", "15m", "30m", "1h", "1d", "5d", "1wk", "1mo"]
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

if st.sidebar.button("Analyze Stock"):
    try:
        # Clear previous logs
        log_buffer.truncate(0)
        log_buffer.seek(0)
        
        with st.spinner('Fetching stock data...'):
            logger.info(f"Starting analysis for {ticker}")
            
            # Get stock data first
            logger.info(f"Fetching stock data with period={period} and interval={interval}")
            df = get_stock_data(ticker, period, interval)
            
            if df.empty:
                logger.error("No data available")
                st.error(f"No data available for {ticker} with the selected period and interval. Try a different interval.")
            else:
                logger.info(f"Successfully fetched {len(df)} rows of data")
                
                # Show basic stock info
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
                        f"${current_price:.2f}",
                        f"{price_change:+.2f}%"
                    )
                with col2:
                    high_price = df['High'].max()
                    st.metric("Period High", f"${high_price:.2f}")
                with col3:
                    low_price = df['Low'].min()
                    st.metric("Period Low", f"${low_price:.2f}")

                # Add technical indicators if needed
                if analysis_mode in ["Technical Analysis", "Both"]:
                    df = add_technical_indicators(df)
                
                # Create layout based on selected mode
                if analysis_mode == "Technical Analysis":
                    # Single column layout for technical analysis
                    # Display candlestick chart
                    st.subheader("Interactive Stock Chart")
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
                            st.metric(f"Support Level {i}", f"${level:.2f}")
                        st.subheader("Resistance Levels")
                        for i, level in enumerate(resistance_levels[-3:], 1):
                            st.metric(f"Resistance Level {i}", f"${level:.2f}")
                    else:
                        st.write("No support/resistance levels found in the current timeframe.")
                    
                    # Additional Technical Metrics
                    with st.expander("Additional Technical Metrics"):
                        metrics_col1, metrics_col2 = st.columns(2)
                        with metrics_col1:
                            st.metric("SMA (20)", f"${df['SMA_20'].iloc[-1]:.2f}")
                            st.metric("Bollinger Upper", f"${df['BB_upper'].iloc[-1]:.2f}")
                        with metrics_col2:
                            st.metric("EMA (20)", f"${df['EMA_20'].iloc[-1]:.2f}")
                            st.metric("Bollinger Lower", f"${df['BB_lower'].iloc[-1]:.2f}")
                
                elif analysis_mode == "ML Forecast":
                    # Single column layout for ML forecast
                    # Add technical indicators before creating plot
                    df = add_technical_indicators(df)
                    
                    # Display candlestick chart
                    st.subheader("Interactive Stock Chart")
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
                                    f"${final_price:.2f}",
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
                
                else:  # Both
                    # Add technical indicators
                    df = add_technical_indicators(df)
                    
                    # Create two columns for the layout
                    chart_col, analysis_col = st.columns([2, 1])
                    
                    with chart_col:
                        # Display candlestick chart
                        st.subheader("Interactive Stock Chart")
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
                                st.metric(f"Support Level {i}", f"${level:.2f}")
                            st.subheader("Resistance Levels")
                            for i, level in enumerate(resistance_levels[-3:], 1):
                                st.metric(f"Resistance Level {i}", f"${level:.2f}")
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
                                        f"${final_price:.2f}",
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
                                st.metric("SMA (20)", f"${df['SMA_20'].iloc[-1]:.2f}")
                                st.metric("Bollinger Upper", f"${df['BB_upper'].iloc[-1]:.2f}")
                            
                            with metrics_col2:
                                st.metric("EMA (20)", f"${df['EMA_20'].iloc[-1]:.2f}")
                                st.metric("Bollinger Lower", f"${df['BB_lower'].iloc[-1]:.2f}")
                
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
        st.info("Please check if the ticker symbol is correct and try again.")
    
    finally:
        # Display logs in an expander
        with st.expander("Debug Logs", expanded=True):
            st.text(log_buffer.getvalue())

# Footer
st.markdown("---")
st.markdown("### About")
st.markdown("""
This app combines traditional technical analysis with machine learning to provide stock price predictions.
- **LSTM Model**: Deep learning model that captures long-term dependencies in time series data
- **XGBoost Model**: Gradient boosting model that excels at feature-based prediction
- **Forecast Horizon**: Choose between 1-day and 3-day ahead predictions
""") 