import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from models.lstm_model import LSTMWithAttention, predict_next_day  # Added predict_next_day import
from models.utils import get_device_with_memory_check, calculate_trading_steps # Assuming utils.py
import os
import joblib
import logging
import pandas as pd
from datetime import timedelta

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    A class for training an LSTM model to predict the next day's average price
    and bullish/bearish sentiment from 5-minute historical data.
    """

    def __init__(
        self,
        model_type='lstm_attention',  # New model type
        forecast_days=1,
        interval='5m',
        sequence_length=None,
        hidden_size=128,  # Updated default
        num_layers=2,
        dropout=0.2,
        attention_heads=4, # Add attention heads parameter
        lr=1e-3,
        epochs=20,
        batch_size=32,
        device=None,
        model_save_path=None,
        norm_save_path=None,
        patience=5,
        related_dfs=None  # Add related_dfs parameter
    ):
        self.model_type = model_type.lower()
        if self.model_type not in ['lstm_attention', 'xgboost']:  # Updated model type
            raise ValueError("model_type must be either 'lstm_attention' or 'xgboost'")

        self.forecast_days = forecast_days
        self.interval = interval
        self._sequence_length = sequence_length  # Store the user-provided sequence length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.attention_heads = attention_heads # Store attention heads
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or get_device_with_memory_check()
        self.related_dfs = related_dfs or []  # Store related stocks dataframes

        if model_save_path is None:
            model_save_path = f"model_{self.model_type}_{self.interval}_{self.forecast_days}d.pt"
        if norm_save_path is None:
            norm_save_path = f"norm_params_{self.model_type}_{self.interval}_{self.forecast_days}d.joblib"

        self.model_save_path = model_save_path
        self.norm_save_path = norm_save_path
        self.patience = patience

        self.model = None
        self.f_min = None
        self.f_max = None
        self.best_val_loss = None

        # Base features for each stock
        self.base_features = [
            "Open", "High", "Low", "Close", "Volume",
            "SMA_20", "SMA_50", "EMA_20", "RSI",
            "BB_middle", "BB_upper", "BB_lower",
            "MACD", "Signal_Line",
            "ATR", "BB_width",
            "%K", "%D", "Williams_R",
            "OBV", "CMF", "PVT", "ADX"
        ]
        
        # Initialize feature columns for main stock
        self.feature_cols = self.base_features.copy()

    def _prepare_sequences_for_dual_task(self, df):
        """Prepares sequences for predicting next day's average price and sentiment."""
        logger.info(f"Initial data shape: {df.shape}")
        
        df = df.sort_index() if isinstance(df.index, pd.DatetimeIndex) else df.sort_values("Time")
        from utils import add_technical_indicators
        df = add_technical_indicators(df)
        
        # Store original prices
        df['OriginalClose'] = df['Close'].copy()
        df['OriginalOpen'] = df['Open'].copy()
        df['OriginalHigh'] = df['High'].copy()
        df['OriginalLow'] = df['Low'].copy()
        
        # Calculate log returns for better numerical stability
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            df[f'{col}_Return'] = np.log(df[col] / df[col].shift(1))
        
        # Fill first row NaN from log returns
        for col in price_cols:
            df.loc[df.index[0], f'{col}_Return'] = 0.0
        
        # Update feature columns to use returns for price columns
        self.feature_cols = [col + '_Return' if col in price_cols else col for col in self.feature_cols]
        
        logger.info(f"Shape after adding technical indicators: {df.shape}")
        
        # Check for NaN values before filling
        nan_cols = df[self.feature_cols].isna().sum()
        logger.info(f"NaN values in features before filling: {nan_cols[nan_cols > 0]}")  # Only show columns with NaN values
        
        # Fill NaN values in technical indicators
        df[self.feature_cols] = df[self.feature_cols].ffill().bfill()
        
        # Normalize the data with robust scaling
        data_values = df[self.feature_cols].values
        self.f_min = np.percentile(data_values, 1, axis=0)
        self.f_max = np.percentile(data_values, 99, axis=0)
        
        # Ensure no zero ranges
        range_mask = (self.f_max - self.f_min) < 1e-8
        if range_mask.any():
            logger.warning(f"Features with too small range: {[self.feature_cols[i] for i, small in enumerate(range_mask) if small]}")
            for i in np.where(range_mask)[0]:
                abs_max = np.abs(data_values[:, i]).max()
                if abs_max < 1e-8:
                    self.f_min[i] = -0.001
                    self.f_max[i] = 0.001
                else:
                    self.f_min[i] = -abs_max
                    self.f_max[i] = abs_max
        
        # Scale the data
        eps = 1e-8
        data_scaled = np.clip((data_values - self.f_min) / (self.f_max - self.f_min + eps), -10, 10)
        
        if np.isnan(data_scaled).any():
            logger.error("NaN values found in scaled data!")
            raise ValueError("NaN values in scaled data")

        # Process dates
        if isinstance(df.index, pd.DatetimeIndex):
            df["Date"] = df.index.date
        else:
            df["Date"] = pd.to_datetime(df["Time"]).dt.date

        # Calculate next day returns
        date_groups = df.groupby('Date')
        unique_dates = sorted(df['Date'].unique())
        
        date_to_next = {}
        for i in range(len(unique_dates) - 1):
            date_to_next[unique_dates[i]] = unique_dates[i + 1]
        date_to_next[unique_dates[-1]] = unique_dates[-1]
        
        df['NextDayDate'] = df['Date'].map(date_to_next)

        def get_next_day_return(group_date):
            if group_date not in date_to_next:
                return 0.0
                
            next_date = date_to_next[group_date]
            if next_date == group_date:
                return 0.0
                
            next_day_data = df[df['Date'] == next_date]
            current_day_data = df[df['Date'] == group_date]
            
            if len(next_day_data) >= 2 and len(current_day_data) > 0:
                current_close = current_day_data['OriginalClose'].iloc[-1]
                next_day_avg = next_day_data[['OriginalOpen', 'OriginalHigh', 'OriginalLow', 'OriginalClose']].mean(axis=1).mean()
                return np.log(next_day_avg / current_close)
            
            return 0.0

        # Calculate next day returns
        next_day_returns = {}
        for date in unique_dates:
            next_day_returns[date] = get_next_day_return(date)
        
        # Map the returns back to the DataFrame
        df['NextDayReturn'] = df['Date'].map(next_day_returns)
        
        # Calculate sentiment based on next day's return
        df['Bullish'] = (df['NextDayReturn'] > 0).astype(int)
        df['Bearish'] = (df['NextDayReturn'] <= 0).astype(int)
        df['Sentiment'] = df['Bearish']

        # Remove the last day's data
        df = df.iloc[:-1]
        logger.info(f"Final shape before sequence formation: {df.shape}")

        X, y_returns, y_sentiment, timestamps = [], [], [], []
        for i in range(self.sequence_length, len(df)):
            seq = data_scaled[i - self.sequence_length : i]
            return_label = df["NextDayReturn"].iloc[i]
            sentiment_label = int(df["Sentiment"].iloc[i])

            if not np.isnan(return_label):
                X.append(seq)
                y_returns.append(return_label)
                y_sentiment.append(sentiment_label)
                
                if isinstance(df.index, pd.DatetimeIndex):
                    timestamps.append(df.index[i])
                else:
                    timestamps.append(df["Time"].iloc[i])

        X = np.array(X, dtype=np.float32)
        y_returns = np.array(y_returns, dtype=np.float32)
        y_sentiment = np.array(y_sentiment, dtype=np.int64)
        
        # Store the last close price for converting returns back to prices
        self.last_close = df['OriginalClose'].iloc[-1]
        logger.info(f"Last close price: {self.last_close:.2f}")
        
        # Final validation
        if len(X) == 0:
            raise ValueError("No valid sequences could be created")
        if np.isnan(X).any() or np.isnan(y_returns).any():
            raise ValueError("NaN values in final data")
        
        logger.info(f"Final sequence shape: {X.shape}")
        return X, y_returns, y_sentiment, timestamps

    def _time_based_split_dual_task(self, X, y_avg_price, y_sentiment, timestamps, train_ratio=0.8):
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

    def _prepare_sequences_with_related_stocks(self, main_df):
        """Prepares sequences including related stocks data as features."""
        logger.info(f"Initial main data shape: {main_df.shape}")
        
        # Process main stock data
        main_df = main_df.sort_index() if isinstance(main_df.index, pd.DatetimeIndex) else main_df.sort_values("Time")
        from utils import add_technical_indicators
        main_df = add_technical_indicators(main_df)
        
        # Store original prices from main stock
        main_df['OriginalClose'] = main_df['Close'].copy()
        main_df['OriginalOpen'] = main_df['Open'].copy()
        main_df['OriginalHigh'] = main_df['High'].copy()
        main_df['OriginalLow'] = main_df['Low'].copy()
        
        # Process related stocks data
        processed_related_dfs = []
        for i, related_df in enumerate(self.related_dfs):
            related_df = related_df.sort_index() if isinstance(related_df.index, pd.DatetimeIndex) else related_df.sort_values("Time")
            related_df = add_technical_indicators(related_df)
            processed_related_dfs.append(related_df)
            
            # Add related stock features to feature_cols with prefix
            for feature in self.base_features:
                self.feature_cols.append(f"Related_{i+1}_{feature}")
        
        # Calculate log returns for all stocks
        price_cols = ['Open', 'High', 'Low', 'Close']
        
        # Process main stock returns
        for col in price_cols:
            main_df[f'{col}_Return'] = np.log(main_df[col] / main_df[col].shift(1))
            main_df.loc[main_df.index[0], f'{col}_Return'] = 0.0
        
        # Process related stocks returns and align with main stock
        for i, related_df in enumerate(processed_related_dfs):
            for col in price_cols:
                related_df[f'{col}_Return'] = np.log(related_df[col] / related_df[col].shift(1))
                related_df.loc[related_df.index[0], f'{col}_Return'] = 0.0
        
        # Update feature columns to use returns for price columns
        updated_features = []
        for feature in self.feature_cols:
            if any(price_col in feature and not feature.endswith('_Return') for price_col in price_cols):
                updated_features.append(feature + '_Return')
            else:
                updated_features.append(feature)
        self.feature_cols = updated_features
        
        # Prepare main stock features
        main_features = main_df[self.base_features].values
        
        # Combine features from all stocks
        combined_features = [main_features]
        for related_df in processed_related_dfs:
            related_features = related_df[self.base_features].values
            combined_features.append(related_features)
        
        # Align and combine all features
        data_values = np.concatenate(combined_features, axis=1)
        
        # Normalize the combined data
        self.f_min = np.percentile(data_values, 1, axis=0)
        self.f_max = np.percentile(data_values, 99, axis=0)
        
        # Handle zero ranges
        range_mask = (self.f_max - self.f_min) < 1e-8
        if range_mask.any():
            logger.warning(f"Features with too small range: {[self.feature_cols[i] for i, small in enumerate(range_mask) if small]}")
            for i in np.where(range_mask)[0]:
                abs_max = np.abs(data_values[:, i]).max()
                if abs_max < 1e-8:
                    self.f_min[i] = -0.001
                    self.f_max[i] = 0.001
                else:
                    self.f_min[i] = -abs_max
                    self.f_max[i] = abs_max
        
        # Scale the data
        eps = 1e-8
        data_scaled = np.clip((data_values - self.f_min) / (self.f_max - self.f_min + eps), -10, 10)
        
        if np.isnan(data_scaled).any():
            logger.error("NaN values found in scaled data!")
            raise ValueError("NaN values in scaled data")

        # Process dates and calculate targets (using main stock data only)
        if isinstance(main_df.index, pd.DatetimeIndex):
            main_df["Date"] = main_df.index.date
        else:
            main_df["Date"] = pd.to_datetime(main_df["Time"]).dt.date

        # Calculate next day returns for main stock
        date_groups = main_df.groupby('Date')
        unique_dates = sorted(main_df['Date'].unique())
        
        date_to_next = {}
        for i in range(len(unique_dates) - 1):
            date_to_next[unique_dates[i]] = unique_dates[i + 1]
        date_to_next[unique_dates[-1]] = unique_dates[-1]
        
        main_df['NextDayDate'] = main_df['Date'].map(date_to_next)

        def get_next_day_return(group_date):
            if group_date not in date_to_next:
                return 0.0
            next_date = date_to_next[group_date]
            if next_date == group_date:
                return 0.0
            next_day_data = main_df[main_df['Date'] == next_date]
            current_day_data = main_df[main_df['Date'] == group_date]
            if len(next_day_data) >= 2 and len(current_day_data) > 0:
                current_close = current_day_data['OriginalClose'].iloc[-1]
                next_day_avg = next_day_data[['OriginalOpen', 'OriginalHigh', 'OriginalLow', 'OriginalClose']].mean(axis=1).mean()
                return np.log(next_day_avg / current_close)
            return 0.0

        # Calculate next day returns
        next_day_returns = {}
        for date in unique_dates:
            next_day_returns[date] = get_next_day_return(date)
        
        main_df['NextDayReturn'] = main_df['Date'].map(next_day_returns)
        main_df['Sentiment'] = (main_df['NextDayReturn'] <= 0).astype(int)

        # Remove the last day's data
        main_df = main_df.iloc[:-1]
        data_scaled = data_scaled[:-1]
        
        logger.info(f"Final shape before sequence formation: {data_scaled.shape}")

        # Prepare sequences
        X, y_returns, y_sentiment, timestamps = [], [], [], []
        for i in range(self.sequence_length, len(main_df)):
            seq = data_scaled[i - self.sequence_length : i]
            return_label = main_df["NextDayReturn"].iloc[i]
            sentiment_label = int(main_df["Sentiment"].iloc[i])

            if not np.isnan(return_label):
                X.append(seq)
                y_returns.append(return_label)
                y_sentiment.append(sentiment_label)
                
                if isinstance(main_df.index, pd.DatetimeIndex):
                    timestamps.append(main_df.index[i])
                else:
                    timestamps.append(main_df["Time"].iloc[i])

        X = np.array(X, dtype=np.float32)
        y_returns = np.array(y_returns, dtype=np.float32)
        y_sentiment = np.array(y_sentiment, dtype=np.int64)
        
        # Store the last close price for converting returns back to prices
        self.last_close = main_df['OriginalClose'].iloc[-1]
        logger.info(f"Last close price: {self.last_close:.2f}")
        
        logger.info(f"Final sequence shape: {X.shape}")
        return X, y_returns, y_sentiment, timestamps

    def prepare_data(self, df, train_ratio=0.8):
        # Calculate sequence length based on data
        if self._sequence_length is None:
            steps_per_day = calculate_trading_steps(self.interval, 1)
            history_days = max(3, self.forecast_days)  # Reduced from 5 to 3 days minimum
            self.sequence_length = min(
                int(steps_per_day * history_days),
                int(len(df) * 0.2)  # Use at most 20% of data length for sequence
            )
        else:
            self.sequence_length = self._sequence_length

        if self.model_type == 'lstm_attention':
            if self.related_dfs:
                # Prepare sequences with related stocks data
                X, y_returns, y_sentiment, timestamps = self._prepare_sequences_with_related_stocks(df)
            else:
                # Use original preparation method for single stock
                X, y_returns, y_sentiment, timestamps = self._prepare_sequences_for_dual_task(df)
            
            # Split the data
            split_data = self._time_based_split_dual_task(
                X, y_returns, y_sentiment, timestamps, train_ratio
            )
            
            # Store split data for training
            (self.X_train, self.y_returns_train, self.y_sentiment_train,
             self.X_val, self.y_returns_val, self.y_sentiment_val,
             self.ts_train, self.ts_val) = split_data
            
            # Initialize the model with the correct parameters
            main_input_size = len(self.base_features)  # Number of features per stock
            num_related_stocks = len(self.related_dfs)
            self.model = LSTMWithAttention(
                main_input_size=main_input_size,
                num_related_stocks=num_related_stocks,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                attention_heads=self.attention_heads
            )
            
            return split_data
        else:
            # XGBoost preparation remains the same
            return self._prepare_sequences_xgboost(df, train_ratio)

    def train(self):
        """Train the model using the prepared data."""
        if self.model is None:
            raise ValueError("Model not initialized. Call prepare_data first.")

        if self.model_type == 'lstm_attention':
            return self._train_lstm_attention()
        else:
            return self._train_xgboost()

    def _train_lstm_attention(self):
        """Train the LSTM with attention model."""
        self.model = self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion_regression = nn.MSELoss()
        criterion_classification = nn.CrossEntropyLoss()

        # Convert data to PyTorch tensors
        X_train = torch.tensor(self.X_train, dtype=torch.float32).to(self.device)
        y_returns_train = torch.tensor(self.y_returns_train, dtype=torch.float32).to(self.device).unsqueeze(-1)
        y_sentiment_train = torch.tensor(self.y_sentiment_train, dtype=torch.long).to(self.device)

        X_val = torch.tensor(self.X_val, dtype=torch.float32).to(self.device)
        y_returns_val = torch.tensor(self.y_returns_val, dtype=torch.float32).to(self.device).unsqueeze(-1)
        y_sentiment_val = torch.tensor(self.y_sentiment_val, dtype=torch.long).to(self.device)

        train_dataset = TensorDataset(X_train, y_returns_train, y_sentiment_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        best_val_loss = float('inf')
        best_weights = None
        patience_counter = 0

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            total_train_loss = 0.0
            for X_batch, y_returns_batch, y_sent_batch in train_loader:
                optimizer.zero_grad()
                returns_pred, sentiment_probs = self.model(X_batch)

                loss_regression = criterion_regression(returns_pred, y_returns_batch)
                loss_classification = criterion_classification(sentiment_probs, y_sent_batch)

                total_loss = loss_regression + loss_classification
                total_loss.backward()
                optimizer.step()
                total_train_loss += total_loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_returns_pred, val_sentiment_probs = self.model(X_val)
                val_loss_regression = criterion_regression(val_returns_pred, y_returns_val)
                val_loss_classification = criterion_classification(val_sentiment_probs, y_sentiment_val)
                val_total_loss = val_loss_regression + val_loss_classification

            # Early Stopping
            if val_total_loss < best_val_loss:
                best_val_loss = val_total_loss
                best_weights = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            logger.info(f"Epoch [{epoch+1}/{self.epochs}] "
                       f"Train Loss: {avg_train_loss:.6f} "
                       f"Val Loss: {val_total_loss:.6f} "
                       f"(Reg: {val_loss_regression:.6f}, Clf: {val_loss_classification:.6f})")

            if patience_counter >= self.patience:
                logger.info("Early stopping triggered!")
                break

        if best_weights is not None:
            self.model.load_state_dict(best_weights)
            self.best_val_loss = best_val_loss

        # Save the model and normalization parameters
        torch.save(self.model.state_dict(), self.model_save_path)
        joblib.dump({'f_min': self.f_min, 'f_max': self.f_max}, self.norm_save_path)

        return self.model, best_val_loss

    def _train_xgboost(self):
        """Train the XGBoost model."""
        if self.model is None:
            raise ValueError("Model not initialized. Call prepare_data first.")
            
        eval_set = [(self.X_val, self.y_val)]
        self.model.fit(
            self.X_train, self.y_train,
            eval_set=eval_set,
            eval_metric='rmse',
            early_stopping_rounds=self.patience,
            verbose=True
        )
        
        self.best_val_loss = self.model.best_score
        
        # Save the model and normalization parameters
        self.model.save_model(self.model_save_path)
        joblib.dump({'f_min': self.f_min, 'f_max': self.f_max}, self.norm_save_path)
        
        logger.info(f"Best validation RMSE: {self.best_val_loss:.6f}")
        logger.info(f"Model saved to {self.model_save_path}")
        
        return self.model, self.best_val_loss

    def predict_next_day_open(self, recent_sequence):
        """
        Predict the next day's average price and sentiment.
        """
        if self.model is None:
            raise ValueError("Model not found. Train or load a model first.")

        if self.model_type == 'lstm_attention':
            self.model.eval()
            with torch.no_grad():
                x_t = torch.tensor(recent_sequence, dtype=torch.float32).to(self.device)
                avg_price_pred_scaled, sentiment_probs = self.model(x_t)
                avg_price_pred_scaled = avg_price_pred_scaled.item()
                sentiment_probs = sentiment_probs.cpu().numpy()[0]
                predicted_sentiment_label = "Bearish" if np.argmax(sentiment_probs) == 1 else "Bullish"

            open_min = self.f_min[0]
            open_max = self.f_max[0]
            avg_price_pred_unscaled = self._min_max_inv_scale(avg_price_pred_scaled, open_min, open_max)
            return avg_price_pred_unscaled, predicted_sentiment_label, sentiment_probs

        elif self.model_type == 'xgboost':
            if len(recent_sequence.shape) == 3:
                recent_sequence = recent_sequence.reshape(1, -1)
            pred_scaled = self.model.predict(recent_sequence)[0]
            open_min = self.f_min[0]
            open_max = self.f_max[0]
            pred_unscaled = self._min_max_inv_scale(pred_scaled, open_min, open_max)
            return pred_unscaled, None, None # No sentiment prediction for XGBoost

    def load_model(self):
        if not os.path.exists(self.norm_save_path):
            raise FileNotFoundError(f"No normalization params file found at {self.norm_save_path}")
        self.f_min, self.f_max = joblib.load(self.norm_save_path)

        if not os.path.exists(self.model_save_path):
            raise FileNotFoundError(f"No model file found at {self.model_save_path}")

        if self.model_type == 'lstm_attention':
            if self.model is None:
                self.model = LSTMWithAttention(
                    input_size=len(self.feature_cols),
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    dropout=self.dropout,
                    attention_heads=self.attention_heads
                ).to(self.device)
            self.model.load_state_dict(torch.load(self.model_save_path, map_location=self.device))
            self.model.eval()
        elif self.model_type == 'xgboost':
            import xgboost as xgb
            self.model = xgb.XGBRegressor()
            self.model.load_model(self.model_save_path)

        print(f"Model loaded from {self.model_save_path}")

    def get_recent_sequence_from_df(self, df):
        if self.f_min is None or self.f_max is None:
            raise ValueError("Normalization parameters not loaded or trained. Call load_model or train first.")

        # Sort the dataframe
        df_sorted = df.sort_index() if isinstance(df.index, pd.DatetimeIndex) else df.sort_values("Time")
        
        # Store original prices
        df_sorted = df_sorted.copy()
        df_sorted['OriginalClose'] = df_sorted['Close'].copy()
        df_sorted['OriginalOpen'] = df_sorted['Open'].copy()
        df_sorted['OriginalHigh'] = df_sorted['High'].copy()
        df_sorted['OriginalLow'] = df_sorted['Low'].copy()
        
        # Calculate log returns for price columns
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            df_sorted[f'{col}_Return'] = np.log(df_sorted[col] / df_sorted[col].shift(1))
        
        # Fill first row NaN from log returns
        for col in price_cols:
            df_sorted.loc[df_sorted.index[0], f'{col}_Return'] = 0.0
        
        # Add technical indicators if they don't exist
        from utils import add_technical_indicators
        df_sorted = add_technical_indicators(df_sorted)
        
        # Update feature columns to match training format
        feature_cols = [col + '_Return' if col in price_cols else col for col in self.feature_cols]
        
        # Fill any NaN values
        df_sorted[feature_cols] = df_sorted[feature_cols].ffill().bfill()
        
        # Get the last sequence_length rows
        data_values = df_sorted[feature_cols].values[-self.sequence_length:]
        if len(data_values) < self.sequence_length:
            raise ValueError(f"Not enough rows in df to form a {self.sequence_length}-step sequence.")
        
        # Scale the data
        eps = 1e-8
        scaled_seq = np.clip((data_values - self.f_min) / (self.f_max - self.f_min + eps), -10, 10)
        scaled_seq = scaled_seq[np.newaxis, :, :]  # Add batch dimension
        
        # Store last close price
        self.last_close = df_sorted['OriginalClose'].iloc[-1]
        logger.info(f"Last close price for prediction: {self.last_close:.2f}")
        
        return scaled_seq

    def cleanup(self):
        """Clean up model weights and saved files."""
        logger.info("Cleaning up model resources...")
        
        # Delete saved model files if they exist
        if os.path.exists(self.model_save_path):
            try:
                os.remove(self.model_save_path)
                logger.info(f"Deleted model file: {self.model_save_path}")
            except Exception as e:
                logger.error(f"Error deleting model file: {str(e)}")
        
        if os.path.exists(self.norm_save_path):
            try:
                os.remove(self.norm_save_path)
                logger.info(f"Deleted normalization params file: {self.norm_save_path}")
            except Exception as e:
                logger.error(f"Error deleting normalization params file: {str(e)}")
        
        # Clean up model from memory
        if self.model is not None:
            if self.model_type == 'lstm_attention':
                # Move model to CPU before deletion to free GPU memory if it was on GPU
                self.model.cpu()
                del self.model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            elif self.model_type == 'xgboost':
                del self.model
            
            self.model = None
            logger.info("Cleared model from memory")

    def predict(self, df):
        """Predict future prices and sentiment."""
        try:
            if self.model is None:
                raise ValueError("Model not found. Train or load a model first.")

            recent_sequence = self.get_recent_sequence_from_df(df)
            last_timestamp = df.index[-1]
            future_timestamps = []

            if self.interval.endswith('m'):
                delta = timedelta(minutes=int(self.interval[:-1]))
            elif self.interval.endswith('h'):
                delta = timedelta(hours=int(self.interval[:-1]))
            elif self.interval.endswith('d'):
                delta = timedelta(days=int(self.interval[:-1]))
            else:
                raise ValueError(f"Unsupported interval format: {self.interval}")

            current_timestamp = last_timestamp
            for _ in range(self.forecast_days):
                while True:
                    current_timestamp += delta
                    if current_timestamp.weekday() < 5:  # Skip weekends
                        break
                future_timestamps.append(current_timestamp)

            predictions = []
            current_sequence = recent_sequence.copy()
            eps = 1e-8

            for future_ts in future_timestamps:
                if self.model_type == 'lstm_attention':
                    try:
                        next_price, next_sentiment, confidence = predict_next_day(
                            self.model,
                            current_sequence,
                            self.f_min,
                            self.f_max,
                            self.last_close,
                            self.device
                        )
                        predictions.append((future_ts, next_price, next_sentiment, confidence))
                        logger.info(f"Predicted price for {future_ts}: {next_price:.2f} ({next_sentiment} with {confidence:.1%} confidence)")

                        if len(future_timestamps) > 1:
                            # Update sequence for next prediction
                            log_return = np.log(next_price / self.last_close)
                            scaled_return = np.clip((log_return - self.f_min[0]) / (self.f_max[0] - self.f_min[0] + eps), -10, 10)
                            current_sequence = np.roll(current_sequence, shift=-1, axis=1)
                            current_sequence[0, -1, 0] = scaled_return
                            self.last_close = next_price
                    except Exception as e:
                        logger.error(f"Error making LSTM prediction for {future_ts}: {str(e)}")
                        break
                else:  # XGBoost
                    try:
                        # For XGBoost, we just need the latest data point
                        if len(current_sequence.shape) == 3:
                            # Take the last timestep's features
                            features = current_sequence[0, -1, :]
                        else:
                            features = current_sequence[-1, :]
                            
                        # Reshape for prediction
                        features = features.reshape(1, -1)
                        
                        # Get prediction
                        pred_scaled = self.model.predict(features)[0]
                        
                        # Convert scaled prediction back to price
                        next_return = self._min_max_inv_scale(pred_scaled, self.f_min[0], self.f_max[0])
                        next_price = self.last_close * np.exp(next_return)
                        
                        # For XGBoost, we'll use a simple confidence based on prediction magnitude
                        confidence = min(1.0, abs(next_return))  # Scale between 0 and 1
                        sentiment = "Bullish" if next_return > 0 else "Bearish"
                        
                        predictions.append((future_ts, next_price, sentiment, confidence))
                        logger.info(f"Predicted price for {future_ts}: {next_price:.2f} ({sentiment} with {confidence:.1%} confidence)")

                        if len(future_timestamps) > 1:
                            # Update for next prediction
                            log_return = np.log(next_price / self.last_close)
                            scaled_return = np.clip((log_return - self.f_min[0]) / (self.f_max[0] - self.f_min[0] + eps), -10, 10)
                            if len(current_sequence.shape) == 3:
                                current_sequence = np.roll(current_sequence, shift=-1, axis=1)
                                current_sequence[0, -1, 0] = scaled_return
                            else:
                                current_sequence = np.roll(current_sequence, shift=-1, axis=0)
                                current_sequence[-1] = scaled_return
                            self.last_close = next_price
                    except Exception as e:
                        logger.error(f"Error making XGBoost prediction for {future_ts}: {str(e)}")
                        logger.error(f"Current sequence shape: {current_sequence.shape}")
                        break

            if not predictions:
                logger.warning("No predictions were generated!")
                # Add at least one prediction with confidence
                sentiment = "Neutral"
                confidence = 0.5  # Default confidence
                predictions.append((future_timestamps[0], self.last_close, sentiment, confidence))

            return predictions
        finally:
            # Clean up model resources after prediction
            self.cleanup()
            logger.info("Model resources cleaned up after prediction")

    def _min_max_scale(self, data, min_vals, max_vals):
        """Min-max scaling with epsilon to avoid division by zero."""
        eps = 1e-8
        return (data - min_vals) / (max_vals - min_vals + eps)

    def _min_max_inv_scale(self, scaled_data, min_val, max_val):
        """Inverse transform of min-max scaling."""
        eps = 1e-8
        return scaled_data * (max_val - min_val + eps) + min_val

    def _prepare_sequences_xgboost(self, df, train_ratio=0.8):
        """
        Prepare data for XGBoost model using the same features as LSTM.
        """
        logger.info(f"Initial data shape: {df.shape}")
        
        # Sort and add technical indicators
        df = df.sort_index() if isinstance(df.index, pd.DatetimeIndex) else df.sort_values("Time")
        from utils import add_technical_indicators
        df = add_technical_indicators(df)
        
        # Store original prices
        df['OriginalClose'] = df['Close'].copy()
        
        # Calculate log returns for price columns
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            df[f'{col}_Return'] = np.log(df[col] / df[col].shift(1))
            df.loc[df.index[0], f'{col}_Return'] = 0.0
        
        # Update feature columns to use returns for price columns
        feature_cols = [col + '_Return' if col in price_cols else col for col in self.base_features]
        
        # Fill NaN values
        df[feature_cols] = df[feature_cols].ffill().bfill()
        
        # Normalize the data
        data_values = df[feature_cols].values
        self.f_min = np.percentile(data_values, 1, axis=0)
        self.f_max = np.percentile(data_values, 99, axis=0)
        
        # Handle zero ranges
        range_mask = (self.f_max - self.f_min) < 1e-8
        if range_mask.any():
            logger.warning(f"Features with too small range: {[feature_cols[i] for i, small in enumerate(range_mask) if small]}")
            for i in np.where(range_mask)[0]:
                abs_max = np.abs(data_values[:, i]).max()
                if abs_max < 1e-8:
                    self.f_min[i] = -0.001
                    self.f_max[i] = 0.001
                else:
                    self.f_min[i] = -abs_max
                    self.f_max[i] = abs_max
        
        # Scale the data
        eps = 1e-8
        data_scaled = np.clip((data_values - self.f_min) / (self.f_max - self.f_min + eps), -10, 10)
        
        # Calculate target (next day's return)
        if isinstance(df.index, pd.DatetimeIndex):
            df["Date"] = df.index.date
        else:
            df["Date"] = pd.to_datetime(df["Time"]).dt.date
            
        date_groups = df.groupby('Date')
        unique_dates = sorted(df['Date'].unique())
        
        date_to_next = {}
        for i in range(len(unique_dates) - 1):
            date_to_next[unique_dates[i]] = unique_dates[i + 1]
        date_to_next[unique_dates[-1]] = unique_dates[-1]
        
        df['NextDayDate'] = df['Date'].map(date_to_next)

        def get_next_day_return(group_date):
            if group_date not in date_to_next:
                return 0.0
            next_date = date_to_next[group_date]
            if next_date == group_date:
                return 0.0
            next_day_data = df[df['Date'] == next_date]
            current_day_data = df[df['Date'] == group_date]
            if len(next_day_data) >= 2 and len(current_day_data) > 0:
                current_close = current_day_data['OriginalClose'].iloc[-1]
                next_day_avg = next_day_data[['Open', 'High', 'Low', 'Close']].mean(axis=1).mean()
                return np.log(next_day_avg / current_close)
            return 0.0

        next_day_returns = {}
        for date in unique_dates:
            next_day_returns[date] = get_next_day_return(date)
        
        df['NextDayReturn'] = df['Date'].map(next_day_returns)
        
        # Remove the last day's data and any NaN values
        df = df.iloc[:-1].dropna()
        
        # Prepare final features and target
        X = data_scaled[:len(df)]  # Ensure X and y have same length
        y = df['NextDayReturn'].values
        timestamps = df.index if isinstance(df.index, pd.DatetimeIndex) else df['Time']
        
        # Store the last close price for predictions
        self.last_close = df['OriginalClose'].iloc[-1]
        
        # Split the data
        split_idx = int(len(X) * train_ratio)
        self.X_train = X[:split_idx]
        self.y_train = y[:split_idx]
        self.X_val = X[split_idx:]
        self.y_val = y[split_idx:]
        ts_train = timestamps[:split_idx]
        ts_val = timestamps[split_idx:]
        
        # Initialize XGBoost model
        import xgboost as xgb
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            objective='reg:squarederror',
            random_state=42
        )
        
        logger.info(f"Training data shape: {self.X_train.shape}")
        logger.info(f"Validation data shape: {self.X_val.shape}")
        
        return self.X_train, self.y_train, None, self.X_val, self.y_val, None, ts_train, ts_val