import numpy as np
import torch

def get_device_with_memory_check():
    """
    Check for CUDA availability and memory, returning the appropriate device
    :return: torch.device object
    """
    if torch.cuda.is_available():
        try:
            # Try to allocate a small tensor to check if CUDA is actually usable
            torch.cuda.empty_cache()
            test_tensor = torch.zeros((1, 1)).cuda()
            del test_tensor
            return torch.device("cuda")
        except RuntimeError:
            # If we get a CUDA error (e.g., out of memory), fall back to CPU
            return torch.device("cpu")
    return torch.device("cpu")

def calculate_trading_steps(interval, forecast_days):
    """
    Calculate the number of trading steps based on interval and forecast days
    :param interval: Trading interval (e.g., "1m", "5m", "15m", "30m", "1h", "1d")
    :param forecast_days: Number of days to forecast
    :return: Number of trading steps
    """
    # For intraday intervals, calculate steps based on trading hours (assuming 6.5 hours per day)
    trading_hours_per_day = 6.5
    minutes_per_day = trading_hours_per_day * 60
    
    steps_map = {
        "1m": minutes_per_day,
        "5m": minutes_per_day / 5,
        "15m": minutes_per_day / 15,
        "30m": minutes_per_day / 30,
        "1h": trading_hours_per_day,
        "1d": 1,
        "5d": 1/5,
        "1wk": 1/7,
        "1mo": 1/30
    }
    
    if interval not in steps_map:
        raise ValueError(f"Unsupported interval: {interval}")
    
    # Calculate total steps
    steps = int(forecast_days * steps_map[interval])
    return max(1, steps)  # Ensure at least 1 step

def get_bullish_bearish_confidence(current_price, predicted_price, model_mse):
    """
    Calculate confidence measure for bullish/bearish prediction
    :param current_price: Current stock price
    :param predicted_price: Model's predicted price
    :param model_mse: Model's mean squared error from validation
    :return: Direction (str) and confidence score (float)
    """
    diff = predicted_price - current_price
    
    # If model_mse is None, use a simpler confidence calculation
    if model_mse is None:
        # Use percentage change as confidence
        confidence_score = (diff / current_price) * 100
    else:
        # Convert MSE to RMSE for scaling
        rmse = np.sqrt(model_mse)
        # Scale the difference by RMSE to get a confidence score
        confidence_score = diff / rmse if rmse != 0 else 0
    
    # Determine direction based on confidence threshold
    if confidence_score > 0.5:
        direction = "Bullish"
    elif confidence_score < -0.5:
        direction = "Bearish"
    else:
        direction = "Uncertain"
    
    return direction, abs(confidence_score)  # Return absolute value for confidence

def calculate_metrics(y_true, y_pred):
    """
    Calculate various performance metrics
    :param y_true: True values
    :param y_pred: Predicted values
    :return: Dictionary of metrics
    """
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Calculate directional accuracy
    direction_true = np.diff(y_true) > 0
    direction_pred = np.diff(y_pred) > 0
    directional_accuracy = np.mean(direction_true == direction_pred)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'directional_accuracy': directional_accuracy
    }

def normalize_data(data, method='zscore'):
    """
    Normalize data using various methods
    :param data: numpy array of data to normalize
    :param method: 'zscore' or 'minmax'
    :return: normalized data, params used for denormalization
    """
    if method == 'zscore':
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        normalized = (data - mean) / (std + 1e-8)
        params = {'mean': mean, 'std': std}
    
    elif method == 'minmax':
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        normalized = (data - min_val) / (max_val - min_val + 1e-8)
        params = {'min': min_val, 'max': max_val}
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized, params

def denormalize_data(normalized_data, params, method='zscore'):
    """
    Denormalize data using saved parameters
    :param normalized_data: normalized data
    :param params: parameters used for normalization
    :param method: 'zscore' or 'minmax'
    :return: denormalized data
    """
    if method == 'zscore':
        return normalized_data * params['std'] + params['mean']
    elif method == 'minmax':
        return normalized_data * (params['max'] - params['min']) + params['min']
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def create_sequences(data, seq_length):
    """
    Create sequences for time series data
    :param data: Input data array
    :param seq_length: Length of each sequence
    :return: Array of sequences
    """
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences) 