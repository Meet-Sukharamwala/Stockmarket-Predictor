# src/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from src.utils.logger import get_logger

logger = get_logger(__name__)

class Preprocessor:
    def __init__(self, test_size=0.2, scale=True):
        self.test_size = test_size
        self.scale = scale
        self.scaler = MinMaxScaler()

    def prepare_features(self, df: pd.DataFrame, target_col: str = "Close"):
        """Create a clean feature set without leakage."""
        features = df.copy()
        # Drop non-predictive columns
        drop_cols = ['Dividends', 'Stock Splits']
        features = features.drop(columns=[c for c in drop_cols if c in features.columns], errors='ignore')

        # Handle missing values (forward fill then drop remaining)
        features.ffill(inplace=True)
        features.dropna(inplace=True)

        self.feature_cols = [c for c in features.columns if c != target_col]
        return features

    def split_data(self, df: pd.DataFrame, target_col: str = "Close"):
        """Time-series split: last test_size% of data for test."""
        split_idx = int(len(df) * (1 - self.test_size))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        X_train, y_train = train_df[self.feature_cols], train_df[target_col]
        X_test, y_test = test_df[self.feature_cols], test_df[target_col]
        return X_train, X_test, y_train, y_test

    def scale_data(self, X_train, X_test):
        """Fit scaler on train, transform both."""
        if self.scale:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            return pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index), \
                   pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        return X_train, X_test
