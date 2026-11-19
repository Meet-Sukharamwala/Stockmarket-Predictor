# src/feature_engineering.py
import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)

class FeatureEngineer:
    def __init__(self, window_sizes=[5, 10, 20, 50], lags=[1, 2, 3, 5, 10]):
        self.window_sizes = window_sizes
        self.lags = lags

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all features from OHLCV data."""
        data = df.copy()
        data['Return'] = data['Close'].pct_change()                # Daily return
        data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))

        # Rolling averages of Close (multiple windows)
        for w in self.window_sizes:
            data[f'Close_SMA_{w}'] = data['Close'].rolling(w).mean()
            data[f'Volume_SMA_{w}'] = data['Volume'].rolling(w).mean()

        # Volatility: rolling standard deviation of returns
        data['Volatility_5'] = data['Return'].rolling(5).std()
        data['Volatility_20'] = data['Return'].rolling(20).std()

        # Momentum: price change over N days
        for w in self.window_sizes:
            data[f'Momentum_{w}'] = data['Close'] - data['Close'].shift(w)

        # Lag features of Close and Volume
        for lag in self.lags:
            data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)
            data[f'Volume_Lag_{lag}'] = data['Volume'].shift(lag)

        # High-Low spread, Close-Open spread
        data['High_Low_Spread'] = data['High'] - data['Low']
        data['Close_Open_Spread'] = data['Close'] - data['Open']

        # Drop rows with NaN caused by rolling/lags
        data.dropna(inplace=True)
        logger.info(f"Created {len(data.columns) - 6} new features. Final shape: {data.shape}")
        return data
