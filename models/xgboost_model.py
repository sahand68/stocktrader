import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def prepare_xgboost_data(df, target_shift=1):
    """
    Create features from indicators in df to predict future close price.
    :param df: DataFrame with technical indicators
    :param target_shift: How many days ahead to predict (1 for next day, 3 for 3 days ahead)
    :return: X features array, y target array, feature names
    """
    df = df.copy()
    
    # Define feature columns to match LSTM model
    feature_cols = [
        # OHLCV
        'Open', 'High', 'Low', 'Close', 'Volume',
        # Technical Indicators
        'SMA_20', 'SMA_50', 'EMA_20', 'RSI',
        'BB_middle', 'BB_upper', 'BB_lower',
        'MACD', 'Signal_Line',
        'ATR', 'BB_width',
        '%K', '%D', 'Williams_R',
        'OBV', 'CMF', 'PVT', 'ADX'
    ]
    
    # Create target (future price)
    df['Target'] = df['Close'].shift(-target_shift)
    
    # Drop any rows with NaN values
    df.dropna(inplace=True)
    
    # Create feature matrix
    X = df[feature_cols].values
    y = df['Target'].values
    
    return X, y, feature_cols

def train_xgboost(X_train, y_train, X_val, y_val, params=None):
    """
    Train XGBoost model with early stopping
    :param X_train: Training features
    :param y_train: Training targets
    :param X_val: Validation features
    :param y_val: Validation targets
    :param params: XGBoost parameters (optional)
    :return: Trained model and validation MSE
    """
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'objective': 'reg:squarederror',
            'random_state': 42
        }
    
    model = xgb.XGBRegressor(**params)
    
    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=10,
        verbose=False
    )
    
    # Get validation performance
    val_preds = model.predict(X_val)
    val_mse = mean_squared_error(y_val, val_preds)
    
    return model, val_mse

def predict_xgboost(model, X):
    """Make predictions using the trained XGBoost model"""
    return model.predict(X)  # X should be 2D array

def get_feature_importance(model, feature_names):
    """
    Get feature importance from trained XGBoost model
    :param model: Trained XGBoost model
    :param feature_names: List of feature names
    :return: Dictionary of feature importance scores
    """
    importance_scores = model.feature_importances_
    return dict(zip(feature_names, importance_scores)) 