import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import datetime
from polygon import RESTClient
import os
from datetime import datetime, timedelta
from utils import add_technical_indicators, calculate_relative_performance

# ==========================
#   IMPROVED LSTM MODEL WITH ATTENTION AND RELATED STOCKS
# ==========================
class LSTMWithAttention(nn.Module):
    def __init__(self, main_input_size, num_related_stocks=5, hidden_size=128, num_layers=2, dropout=0.2, attention_heads=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.num_related_stocks = num_related_stocks
        
        # Calculate total input size (main stock features + related stocks features + relative performance features)
        # 23 features per stock (OHLCV + technical indicators)
        # 6 relative performance features per related stock
        base_features_per_stock = 23  # OHLCV + technical indicators
        relative_features_per_stock = 6  # Relative performance metrics
        total_input_size = (base_features_per_stock * (1 + num_related_stocks)) + (relative_features_per_stock * num_related_stocks)
        
        # Main LSTM layer
        self.lstm = nn.LSTM(total_input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Cross-stock attention mechanism
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=attention_heads, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # Shared fully connected layer
        self.fc_shared = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()

        # Regression head for average price prediction
        self.fc_regression = nn.Linear(hidden_size // 2, 1)

        # Classification head for bullish/bearish prediction
        self.fc_classification = nn.Linear(hidden_size // 2, 2)

    def forward(self, x):
        # x shape: (batch_size, seq_len, total_input_size)
        batch_size = x.size(0)

        # LSTM layer
        out, _ = self.lstm(x)

        # Attention mechanism
        attn_output, _ = self.attention(out, out, out)
        attn_output = self.dropout(attn_output)

        # Take the last time step's output
        attn_last = attn_output[:, -1, :]

        # Shared fully connected layer
        shared = self.relu(self.fc_shared(attn_last))
        shared = self.dropout(shared)

        # Regression head
        average_price_pred = self.fc_regression(shared)

        # Classification head
        sentiment_logits = self.fc_classification(shared)
        sentiment_probs = torch.softmax(sentiment_logits, dim=-1)

        return average_price_pred, sentiment_probs

# ==========================
#   DATA PREPARATION WITH RELATED STOCKS
# ==========================
def prepare_enhanced_features(main_df, related_dfs):
    """
    Prepare enhanced feature set including technical indicators and relative performance metrics.
    
    Args:
        main_df: DataFrame for the main stock
        related_dfs: List of DataFrames for related stocks
    
    Returns:
        main_df: Enhanced main stock DataFrame
        related_dfs: List of enhanced related stock DataFrames
    """
    # Add technical indicators to main stock
    main_df = add_technical_indicators(main_df.copy())
    
    # Add technical indicators to related stocks and calculate relative performance
    enhanced_related_dfs = []
    for idx, related_df in enumerate(related_dfs):
        # Add technical indicators to related stock
        enhanced_related_df = add_technical_indicators(related_df.copy())
        
        # Calculate relative performance metrics
        main_df = calculate_relative_performance(
            main_df, 
            enhanced_related_df,
            suffix=f'_stock_{idx+1}'
        )
        
        enhanced_related_dfs.append(enhanced_related_df)
    
    return main_df, enhanced_related_dfs

def prepare_sequences_for_dual_task(
    main_df,
    related_dfs,
    sequence_length=390  # 5 days * 78 bars/day for 5-min intervals
):
    """
    Prepare sequences including data from related stocks with enhanced features
    main_df: DataFrame for the main stock
    related_dfs: List of DataFrames for related stocks
    """
    # Add enhanced features
    main_df, related_dfs = prepare_enhanced_features(main_df, related_dfs)
    
    # Ensure all DataFrames are properly indexed
    main_df = main_df.sort_index() if isinstance(main_df.index, pd.DatetimeIndex) else main_df.sort_values('Time')
    related_dfs = [df.sort_index() if isinstance(df.index, pd.DatetimeIndex) else df.sort_values('Time') for df in related_dfs]

    # Define feature columns for main stock
    main_feature_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "SMA_20", "SMA_50", "EMA_20", "RSI",
        "BB_middle", "BB_upper", "BB_lower",
        "MACD", "Signal_Line",
        "ATR", "BB_width",
        "%K", "%D", "Williams_R",
        "OBV", "CMF", "PVT", "ADX"
    ]
    
    # Add relative performance columns for each related stock
    for i in range(len(related_dfs)):
        suffix = f'_stock_{i+1}'
        main_feature_cols.extend([
            f'Price_Ratio{suffix}',
            f'Return_Difference{suffix}',
            f'Relative_Strength{suffix}',
            f'Rolling_Correlation{suffix}',
            f'Volume_Ratio{suffix}',
            f'Beta{suffix}'
        ])
    
    # Process main stock features
    main_features = main_df[main_feature_cols].values
    main_min = main_features.min(axis=0)
    main_max = main_features.max(axis=0)
    eps = 1e-8
    main_features_scaled = (main_features - main_min) / (main_max - main_min + eps)

    # Define feature columns for related stocks
    related_feature_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "SMA_20", "SMA_50", "EMA_20", "RSI",
        "BB_middle", "BB_upper", "BB_lower",
        "MACD", "Signal_Line",
        "ATR", "BB_width",
        "%K", "%D", "Williams_R",
        "OBV", "CMF", "PVT", "ADX"
    ]

    # Process related stocks features
    related_features_scaled = []
    for related_df in related_dfs:
        related_features = related_df[related_feature_cols].values
        related_min = related_features.min(axis=0)
        related_max = related_features.max(axis=0)
        related_scaled = (related_features - related_min) / (related_max - related_min + eps)
        related_features_scaled.append(related_scaled)

    # Prepare target variables (using main stock data)
    if isinstance(main_df.index, pd.DatetimeIndex):
        main_df["Date"] = main_df.index.date
    else:
        main_df["Date"] = pd.to_datetime(main_df["Time"]).dt.date

    # Calculate next day's average price
    main_df['NextDayDate'] = main_df['Date'].shift(-1)
    main_df['NextDayDate'] = main_df['NextDayDate'].fillna(main_df['Date'].iloc[-1])

    def get_next_day_avg_price(group):
        next_day_data = main_df[main_df['Date'] == group['NextDayDate'].iloc[0]]
        if len(next_day_data) >= 6:
            first_30_min = next_day_data.iloc[:6]
            return first_30_min[['Open', 'High', 'Low', 'Close']].mean().mean()
        else:
            return np.nan

    main_df['NextDayAvgPrice'] = main_df.groupby('Date').apply(get_next_day_avg_price).reset_index(drop=True)

    # Calculate bullish/bearish label
    main_df['NextDayClose'] = main_df.groupby('Date')['Close'].tail(1).shift(-1)
    main_df['NextDayClose'] = main_df['NextDayClose'].fillna(main_df['Close'].iloc[-1])
    main_df['Sentiment'] = (main_df['NextDayClose'] <= main_df['Close']).astype(int)

    main_df = main_df.dropna(subset=["NextDayAvgPrice", "Sentiment"])

    # Prepare sequences
    X, y_avg_price, y_sentiment = [], [], []
    valid_timestamps = []

    min_length = min(len(main_features_scaled), *[len(rf) for rf in related_features_scaled])

    for i in range(sequence_length, min_length):
        # Get main stock sequence
        main_seq = main_features_scaled[i - sequence_length : i]
        
        # Get related stocks sequences
        related_seqs = [rf[i - sequence_length : i] for rf in related_features_scaled]
        
        # Combine all features
        combined_seq = np.concatenate([main_seq] + related_seqs, axis=1)
        
        X.append(combined_seq)
        y_avg_price.append(main_df["NextDayAvgPrice"].iloc[i])
        y_sentiment.append(main_df["Sentiment"].iloc[i])

        if isinstance(main_df.index, pd.DatetimeIndex):
            valid_timestamps.append(main_df.index[i])
        else:
            valid_timestamps.append(main_df["Time"].iloc[i])

    X = np.array(X, dtype=np.float32)
    y_avg_price = np.array(y_avg_price, dtype=np.float32)
    y_sentiment = np.array(y_sentiment, dtype=np.int64)

    return X, y_avg_price, y_sentiment, main_min, main_max, valid_timestamps

# ==========================
#   TRAIN / VALID SPLIT (remains the same)
# ==========================
def time_based_split(X, y_avg_price, y_sentiment, timestamps, train_ratio=0.8):
    n = len(X)
    split_idx = int(train_ratio * n)
    X_train = X[:split_idx]
    y_avg_price_train = y_avg_price[:split_idx]
    y_sentiment_train = y_sentiment[:split_idx]
    X_val = X[split_idx:]
    y_avg_price_val = y_avg_price[split_idx:]
    y_sentiment_val = y_sentiment[split_idx:]
    ts_train = timestamps[:split_idx]
    ts_val = timestamps[split_idx:]
    return X_train, y_avg_price_train, y_sentiment_train, X_val, y_avg_price_val, y_sentiment_val, ts_train, ts_val

# ==========================
#   MODEL TRAINING (Modified for multi-task)
# ==========================
def train_dual_task_model(
    model,
    X_train, y_avg_price_train, y_sentiment_train,
    X_val, y_avg_price_val, y_sentiment_val,
    epochs=20,
    batch_size=32,
    lr=1e-3,
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion_regression = nn.MSELoss()
    criterion_classification = nn.CrossEntropyLoss() # For multi-class classification

    # Convert data to PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_avg_price_train_t = torch.tensor(y_avg_price_train, dtype=torch.float32).to(device).unsqueeze(-1)
    y_sentiment_train_t = torch.tensor(y_sentiment_train, dtype=torch.long).to(device) # Use torch.long for CrossEntropyLoss

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_avg_price_val_t = torch.tensor(y_avg_price_val, dtype=torch.float32).to(device).unsqueeze(-1)
    y_sentiment_val_t = torch.tensor(y_sentiment_val, dtype=torch.long).to(device)

    train_dataset = TensorDataset(X_train_t, y_avg_price_train_t, y_sentiment_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_val_loss = float("inf")
    best_weights = None
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        # === TRAIN ===
        model.train()
        total_train_loss = 0.0
        for X_batch, y_avg_batch, y_sent_batch in train_loader:
            optimizer.zero_grad()
            avg_price_preds, sentiment_probs = model(X_batch)

            loss_regression = criterion_regression(avg_price_preds, y_avg_batch)
            loss_classification = criterion_classification(sentiment_probs, y_sent_batch)

            # Combine losses (you can weigh them differently if needed)
            total_loss = loss_regression + loss_classification

            total_loss.backward()
            optimizer.step()
            total_train_loss += total_loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # === VALIDATION ===
        model.eval()
        with torch.no_grad():
            val_avg_price_preds, val_sentiment_probs = model(X_val_t)
            val_loss_regression = criterion_regression(val_avg_price_preds, y_avg_price_val_t).item()
            val_loss_classification = criterion_classification(val_sentiment_probs, y_sentiment_val_t).item()
            val_total_loss = val_loss_regression + val_loss_classification

        # Early Stopping Check (based on total validation loss)
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            best_weights = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_total_loss:.6f} (Reg: {val_loss_regression:.6f}, Clf: {val_loss_classification:.6f})")

        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, best_val_loss

# ==========================
#   PREDICT FUNCTION (Modified for dual task)
# ==========================
def predict_next_day(
    model,
    recent_sequence,
    f_min, f_max,
    last_close,
    device=None
):
    if device is None:
        device = next(model.parameters()).device

    # Add logging for input validation
    print(f"Input sequence shape: {recent_sequence.shape}")
    print(f"Input sequence contains nan: {np.isnan(recent_sequence).any()}")
    print(f"Last close price: {last_close:.2f}")

    model.eval()
    with torch.no_grad():
        # Add input validation
        if np.isnan(recent_sequence).any():
            print("WARNING: Input sequence contains NaN values!")
            recent_sequence = np.nan_to_num(recent_sequence, nan=0.0)
        
        inp = torch.tensor(recent_sequence, dtype=torch.float32).to(device)
        returns_pred, sentiment_probs = model(inp)

        # Add prediction validation
        log_return_pred = returns_pred.item()
        print(f"Predicted log return: {log_return_pred:.2%}")
        
        sentiment_probs = sentiment_probs.cpu().numpy()[0]
        print(f"Sentiment probabilities: {sentiment_probs}")

    # Convert log return to price
    predicted_price = last_close * np.exp(log_return_pred)
    print(f"Last close: {last_close:.2f}")
    print(f"Predicted price: {predicted_price:.2f}")

    # Get the predicted sentiment label and confidence
    predicted_class = np.argmax(sentiment_probs)
    sentiment_label = "Bullish" if predicted_class == 0 else "Bearish"
    confidence = sentiment_probs[predicted_class]
    print(f"Predicted sentiment: {sentiment_label} with {confidence:.1%} confidence")

    return predicted_price, sentiment_label, confidence

# ==========================
#   DATA FETCHING FOR RELATED STOCKS
# ==========================
def fetch_stock_data(client, ticker, start_date, end_date, timespan="minute", multiplier=5):
    """
    Fetch stock data from Polygon API
    """
    try:
        aggs = []
        for resp in client.list_aggs(
            ticker=ticker,
            multiplier=multiplier,
            timespan=timespan,
            from_=start_date,
            to=end_date,
            limit=50000
        ):
            aggs.append({
                'Time': pd.Timestamp(resp.timestamp, unit='ms'),
                'Open': resp.open,
                'High': resp.high,
                'Low': resp.low,
                'Close': resp.close,
                'Volume': resp.volume
            })
        
        df = pd.DataFrame(aggs)
        if not df.empty:
            df.set_index('Time', inplace=True)
            df = df.sort_index()
            
            # Filter for market hours (9:30 AM - 4:00 PM ET)
            df = df.between_time('09:30', '16:00')
            
        return df
    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame()

def get_related_stocks_data(client, main_ticker, start_date, end_date, max_related=5):
    """
    Fetch data for the main stock and its related stocks
    """
    # Get related stocks
    try:
        related = client.get_related_companies(main_ticker)
        related_tickers = [comp.ticker for comp in related.results[:max_related]]
        print(f"Found related tickers: {related_tickers}")
    except Exception as e:
        print(f"Error getting related companies: {str(e)}")
        return None, []

    # Fetch main stock data
    main_df = fetch_stock_data(client, main_ticker, start_date, end_date)
    if main_df.empty:
        print(f"Failed to fetch data for main ticker {main_ticker}")
        return None, []

    # Fetch related stocks data
    related_dfs = []
    for ticker in related_tickers:
        df = fetch_stock_data(client, ticker, start_date, end_date)
        if not df.empty:
            related_dfs.append(df)
        else:
            print(f"Skipping {ticker} due to missing data")

    return main_df, related_dfs

# ==========================
#    DEMO / MAIN SCRIPT (Updated for related stocks)
# ==========================
if __name__ == "__main__":
    # Load API key from environment
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise ValueError("Please set POLYGON_API_KEY environment variable")
    
    client = RESTClient(api_key=api_key)
    
    # Set date range for fetching data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # Get 30 days of data
    
    # Fetch main stock and related stocks data
    main_ticker = "AAPL"  # Example ticker
    main_df, related_dfs = get_related_stocks_data(
        client, 
        main_ticker, 
        start_date.strftime('%Y-%m-%d'), 
        end_date.strftime('%Y-%m-%d')
    )
    
    if main_df is None or not related_dfs:
        print("Failed to fetch required data")
        exit(1)

    sequence_length = 78 * 2  # 2 days of 5-min bars
    
    # Prepare sequences for dual task with related stocks
    X, y_avg_price, y_sentiment, f_min, f_max, timestamps = prepare_sequences_for_dual_task(
        main_df, 
        related_dfs,
        sequence_length=sequence_length
    )
    print("X shape:", X.shape)
    print("y_avg_price shape:", y_avg_price.shape)
    print("y_sentiment shape:", y_sentiment.shape)

    # Split into train and validation
    X_train, y_avg_price_train, y_sentiment_train, X_val, y_avg_price_val, y_sentiment_val, ts_train, ts_val = time_based_split(
        X, y_avg_price, y_sentiment, timestamps, train_ratio=0.8
    )
    print("Train size:", len(X_train), "Val size:", len(X_val))

    # Build the improved model with related stocks
    main_input_size = 5  # OHLCV features
    num_related_stocks = len(related_dfs)
    model = LSTMWithAttention(
        main_input_size=main_input_size,
        num_related_stocks=num_related_stocks,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        attention_heads=4
    )

    # Train the dual task model
    model, best_val_loss = train_dual_task_model(
        model,
        X_train, y_avg_price_train, y_sentiment_train,
        X_val, y_avg_price_val, y_sentiment_val,
        epochs=20,
        batch_size=32,
        lr=1e-3,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print("Best validation loss (combined):", best_val_loss)

    # Predict on the last available sequence
    if len(X_val) > 0:
        last_seq = X_val[-1:].copy()
        last_close = y_avg_price_val[-1]
        predicted_price, predicted_sentiment, confidence = predict_next_day(model, last_seq, f_min, f_max, last_close)
        true_avg_price = y_avg_price_val[-1]
        true_sentiment = "Bullish" if y_sentiment_val[-1] == 0 else "Bearish"

        print("\nPrediction Results:")
        print("==================")
        print(f"Predicted next day Average Price: ${predicted_price:.2f}")
        print(f"Actual next day Average Price: ${true_avg_price:.2f}")
        print(f"Predicted next day Sentiment: {predicted_sentiment} with {confidence:.1%} confidence")
        print(f"Actual next day Sentiment: {true_sentiment}")
        print(f"Timestamp: {ts_val[-1]}")
        
        # Print related stocks used
        print("\nRelated Stocks Used:")
        print("==================")
        for i, df in enumerate(related_dfs):
            print(f"{i+1}. {df.index.name if df.index.name else 'Unknown'}")
    else:
        print("Not enough data to form a validation set sample!")