# main.py
import argparse
import pandas as pd
from src.data_loader import StockDataLoader
from src.preprocessing import Preprocessor
from src.feature_engineering import FeatureEngineer
from src.indicators import simple_moving_average, exponential_moving_average
from src.visualization import plot_candlestick_plotly, plot_correlation_heatmap
from src.forecasting import (
    train_linear_regression,
    train_random_forest,
    train_lstm,
    forecast_arima,
    evaluate
)
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main(symbol: str):
    config = load_config()
    start = config["data"]["start_date"]
    end = config["data"]["end_date"]
    interval = config["data"]["interval"]

    # 1. Load data
    loader = StockDataLoader(symbol, start, end, interval)
    df = loader.download()
    logger.info(f"Data shape: {df.shape}")

    # 2. Preprocess
    prep = Preprocessor(
        test_size=config["preprocessing"]["test_size"],
        scale=config["preprocessing"]["scale"]
    )
    features = prep.prepare_features(df)
    X_train, X_test, y_train, y_test = prep.split_data(features)

    # 3. Feature engineering
    fe = FeatureEngineer(
        window_sizes=config["features"]["window_sizes"],
        lags=config["features"]["lags"]
    )
    full_data = fe.create_features(df)   # includes target

    # Re-split after feature engineering (ensure no future leakage)
    split_idx = int(len(full_data) * (1 - config["preprocessing"]["test_size"]))
    train_data = full_data.iloc[:split_idx]
    test_data = full_data.iloc[split_idx:]

    target = 'Close'
    X_train_fe = train_data.drop(columns=[target])
    y_train_fe = train_data[target]
    X_test_fe = test_data.drop(columns=[target])
    y_test_fe = test_data[target]

    # 4. Train ML models
    logger.info("Training Linear Regression...")
    train_linear_regression(X_train_fe, y_train_fe, X_test_fe, y_test_fe)

    logger.info("Training Random Forest with hyperparameter tuning...")
    train_random_forest(X_train_fe, y_train_fe, X_test_fe, y_test_fe)

    # 5. ARIMA forecast on Close price
    logger.info("Training ARIMA...")
    arima_model, arima_forecast, conf = forecast_arima(df['Close'], order=(5,1,0), forecast_steps=30)

    # 6. LSTM training
    logger.info("Training LSTM...")
    lstm_model, history, lstm_preds, y_actual, scaler = train_lstm(
        df['Close'], df['Close'],  # full series; in practice split in time
        lookback=config["lstm"]["lookback"],
        epochs=config["lstm"]["epochs"],
        batch_size=config["lstm"]["batch_size"],
        units=config["lstm"]["units"]
    )

    # 7. Visualizations (optional)
    fig = plot_candlestick_plotly(df.tail(100), symbol)
    fig.write_html(f"candlestick_{symbol}.html")
    heatmap_fig = plot_correlation_heatmap(full_data.drop(columns=[target]))
    heatmap_fig.savefig(f"correlation_{symbol}.png")

    logger.info(f"Pipeline completed for {symbol}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Market Prediction Pipeline")
    parser.add_argument("--symbol", type=str, default="AAPL", help="Stock ticker symbol")
    args = parser.parse_args()
    main(args.symbol)
