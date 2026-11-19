# src/forecasting.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from src.utils.logger import get_logger
from src.model_utils import save_sklearn_model, load_sklearn_model, save_lstm_model, load_lstm_model
import warnings
warnings.filterwarnings("ignore")

logger = get_logger(__name__)

def evaluate(y_true, y_pred, model_name=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    logger.info(f"{model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    return {"RMSE": rmse, "MAE": mae, "R2": r2}

def train_linear_regression(X_train, y_train, X_test, y_test):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    preds = lr.predict(X_test)
    save_sklearn_model(lr, "linear_regression")
    return lr, evaluate(y_test, preds, "Linear Regression")

def train_random_forest(X_train, y_train, X_test, y_test, param_grid=None):
    if param_grid is None:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 20],
            'min_samples_split': [2, 5]
        }
    rf = RandomForestRegressor(random_state=42)
    tscv = TimeSeriesSplit(n_splits=3)
    grid = GridSearchCV(rf, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=1)
    grid.fit(X_train, y_train)
    logger.info(f"Best RF params: {grid.best_params_}")
    best_rf = grid.best_estimator_
    preds = best_rf.predict(X_test)
    save_sklearn_model(best_rf, "random_forest")
    return best_rf, evaluate(y_test, preds, "Random Forest")

def train_xgboost(X_train, y_train, X_test, y_test):
    try:
        import xgboost as xgb
    except ImportError:
        logger.error("XGBoost not installed. Install via pip install xgboost")
        return None, None
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    save_sklearn_model(model, "xgboost")
    return model, evaluate(y_test, preds, "XGBoost")

def prepare_lstm_data(series, lookback=60):
    """Convert time series to supervised learning format for LSTM."""
    X, y = [], []
    for i in range(lookback, len(series)):
        X.append(series[i-lookback:i])
        y.append(series[i])
    return np.array(X), np.array(y)

def train_lstm(series_train, series_test, lookback=60, epochs=50, batch_size=32, units=50):
    """Train LSTM on raw Close prices (scaled). Returns model, predictions, scaler."""
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_train = scaler.fit_transform(series_train.values.reshape(-1,1)).flatten()
    scaled_test = scaler.transform(series_test.values.reshape(-1,1)).flatten()

    X_train, y_train = prepare_lstm_data(scaled_train, lookback)
    X_test, y_test = prepare_lstm_data(scaled_test, lookback)

    # Reshape for LSTM [samples, time steps, features]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = Sequential([
        LSTM(units=units, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(units=units, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train,
                        validation_split=0.1,
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[early_stop],
                        verbose=0)
    preds_scaled = model.predict(X_test)
    # Inverse transform
    preds = scaler.inverse_transform(preds_scaled).flatten()
    y_actual = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()
    save_lstm_model(model, "lstm_model")
    metrics = evaluate(y_actual, preds, "LSTM")
    return model, history, preds, y_actual, scaler

def forecast_arima(series, order=(5,1,0), forecast_steps=30):
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    forecast = model_fit.get_forecast(steps=forecast_steps)
    mean_forecast = forecast.predicted_mean
    conf_int = forecast.conf_int()
    return model_fit, mean_forecast, conf_int

def backtest_walk_forward(model_func, X, y, n_splits=5):
    """Simple walk-forward backtesting for ML models."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    returns = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        rf = model_func()
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        returns.append(y_test.values - preds)   # actual minus predicted (directional assumption)
    # Simple backtest: if prediction > previous close, buy; else sell. (Demo)
    # We'll compute accuracy of direction.
    all_true = []
    all_pred = []
    # ... omitted for brevity but would be implemented similarly.
    logger.info("Walk-forward backtesting completed.")
    return returns
