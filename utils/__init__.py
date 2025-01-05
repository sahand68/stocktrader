from .plotting import *

from polygon import RESTClient
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
from typing import Optional
import dotenv

# Load environment variables
dotenv.load_dotenv()

def get_stock_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch stock data from Polygon.io API.
    
    Args:
        ticker: Stock symbol
        period: Time period (e.g., "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y")
        interval: Time interval ("1m", "5m", "15m", "30m", "1h", "1d")
    """
    try:
        api_key = os.getenv('POLYGON_API_KEY')
        if not api_key:
            raise ValueError("POLYGON_API_KEY environment variable not set")
        
        client = RESTClient(api_key)
        
        # Convert period to start date
        end_date = datetime.now()
        if period.endswith('d'):
            days = int(period[:-1])
            start_date = end_date - timedelta(days=days)
        elif period.endswith('mo'):
            months = int(period[:-2])
            start_date = end_date - timedelta(days=months * 30)
        elif period.endswith('y'):
            years = int(period[:-1])
            start_date = end_date - timedelta(days=years * 365)
        else:
            start_date = end_date - timedelta(days=365)  # Default to 1 year
        
        # Convert interval to Polygon timespan
        timespan_map = {
            "1m": "minute",
            "5m": "minute",
            "15m": "minute",
            "30m": "minute",
            "1h": "hour",
            "1d": "day"
        }
        
        multiplier_map = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 1,
            "1d": 1
        }
        
        timespan = timespan_map.get(interval, "day")
        multiplier = multiplier_map.get(interval, 1)
        
        # Fetch data from Polygon
        aggs = []
        for a in client.list_aggs(
            ticker=ticker,
            multiplier=multiplier,
            timespan=timespan,
            from_=start_date.strftime("%Y-%m-%d"),
            to=end_date.strftime("%Y-%m-%d"),
            limit=50000
        ):
            aggs.append({
                'timestamp': pd.Timestamp(a.timestamp, unit='ms'),
                'Open': a.open,
                'High': a.high,
                'Low': a.low,
                'Close': a.close,
                'Volume': a.volume
            })
        
        if not aggs:
            logging.warning(f"No data found for {ticker}")
            return pd.DataFrame()
        
        df = pd.DataFrame(aggs)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        return df
        
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame()

def add_technical_indicators(df):
    """Add technical indicators to the dataframe."""
    if df.empty:
        return df
    
    # Existing indicators
    # Calculate SMA
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Calculate EMA
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # Calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    
    # Calculate MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # New Volatility Indicators
    # Average True Range (ATR)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    # Bollinger Bands Width
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
    
    # New Momentum Indicators
    # Stochastic Oscillator
    low_min = df['Low'].rolling(14).min()
    high_max = df['High'].rolling(14).max()
    df['%K'] = ((df['Close'] - low_min) / (high_max - low_min)) * 100
    df['%D'] = df['%K'].rolling(3).mean()
    
    # Williams %R
    df['Williams_R'] = ((high_max - df['Close']) / (high_max - low_min)) * -100
    
    # New Volume Indicators
    # On-Balance Volume (OBV) - Enhanced
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    # Chaikin Money Flow
    mf_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    mf_volume = mf_multiplier * df['Volume']
    df['CMF'] = mf_volume.rolling(20).sum() / df['Volume'].rolling(20).sum()
    
    # Price and Volume Trend (PVT)
    df['PVT'] = (df['Close'].pct_change() * df['Volume']).fillna(0).cumsum()
    
    # Volatility and Trend
    # Average Directional Index (ADX)
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    tr = true_range
    plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / tr.ewm(alpha=1/14).mean())
    minus_di = abs(100 * (minus_dm.ewm(alpha=1/14).mean() / tr.ewm(alpha=1/14).mean()))
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    df['ADX'] = dx.ewm(alpha=1/14).mean()
    
    return df

def find_support_resistance_levels(df, window=20):
    """Find support and resistance levels using local minima and maxima."""
    if len(df) < window:
        return [], []
    
    # Initialize lists for support and resistance levels
    support_levels = []
    resistance_levels = []
    
    # Get high and low prices
    high_prices = df['High'].values
    low_prices = df['Low'].values
    
    # Find local minima and maxima
    for i in range(window, len(df) - window):
        # Check for local minimum (support)
        if all(low_prices[i] <= low_prices[i-window:i]) and \
           all(low_prices[i] <= low_prices[i+1:i+window+1]):
            support_levels.append(low_prices[i])
        
        # Check for local maximum (resistance)
        if all(high_prices[i] >= high_prices[i-window:i]) and \
           all(high_prices[i] >= high_prices[i+1:i+window+1]):
            resistance_levels.append(high_prices[i])
    
    # Remove duplicates and sort
    support_levels = sorted(list(set(support_levels)))
    resistance_levels = sorted(list(set(resistance_levels)))
    
    return support_levels, resistance_levels

def determine_trend(df, window=20):
    """Determine the current trend using moving averages and price action."""
    if len(df) < window:
        return "Insufficient data"
    
    # Get current price and moving averages
    current_price = df['Close'].iloc[-1]
    sma_20 = df['SMA_20'].iloc[-1]
    sma_50 = df['SMA_50'].iloc[-1]
    
    # Get RSI
    rsi = df['RSI'].iloc[-1]
    
    # Calculate price change
    price_change = (df['Close'].iloc[-1] - df['Close'].iloc[-window]) / df['Close'].iloc[-window] * 100
    
    # Determine trend based on multiple factors
    if current_price > sma_20 and sma_20 > sma_50 and rsi > 50 and price_change > 0:
        return "Bullish"
    elif current_price < sma_20 and sma_20 < sma_50 and rsi < 50 and price_change < 0:
        return "Bearish"
    else:
        return "Neutral" 

def calculate_relative_performance(main_df, related_df, suffix=''):
    """
    Calculate relative performance metrics between main stock and related stock.
    
    Args:
        main_df: DataFrame of main stock
        related_df: DataFrame of related stock
        suffix: Suffix to add to column names for the related stock
    
    Returns:
        DataFrame with relative performance metrics
    """
    if main_df.empty or related_df.empty:
        return main_df
    
    # Align the DataFrames
    main_df, related_df = main_df.align(related_df, join='inner', axis=0)
    
    # Price ratios
    main_df[f'Price_Ratio{suffix}'] = related_df['Close'] / main_df['Close']
    
    # Returns
    main_returns = main_df['Close'].pct_change()
    related_returns = related_df['Close'].pct_change()
    main_df[f'Return_Difference{suffix}'] = related_returns - main_returns
    
    # Relative strength
    main_df[f'Relative_Strength{suffix}'] = (related_df['Close'] / main_df['Close']) * 100
    
    # Correlation
    rolling_corr = main_df['Close'].rolling(window=20).corr(related_df['Close'])
    main_df[f'Rolling_Correlation{suffix}'] = rolling_corr
    
    # Relative volume
    main_df[f'Volume_Ratio{suffix}'] = related_df['Volume'] / main_df['Volume']
    
    # Beta (20-day)
    main_returns_std = main_returns.rolling(window=20).std()
    related_returns_std = related_returns.rolling(window=20).std()
    rolling_corr = main_returns.rolling(window=20).corr(related_returns)
    main_df[f'Beta{suffix}'] = rolling_corr * (related_returns_std / main_returns_std)
    
    return main_df 