# 📈 Stock Market Prediction & Analytics System

A full‑stack machine learning and analytics platform that fetches live/historical stock data, performs financial analysis, and predicts future prices using classical ML, ARIMA, and deep learning (LSTM). Comes with an interactive Streamlit dashboard.

![Dashboard Screenshot](screenshots/dashboard.png)

## Features

- **Real‑time Data Collection**: yfinance integration for multiple symbols, with local caching.
- **Comprehensive EDA**: Trend analysis, volatility, correlations, distribution plots.
- **Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands with trading interpretations.
- **Candlestick Charts**: Interactive Plotly & static mplfinance charts with volume and moving average overlays.
- **Advanced Feature Engineering**: Lagged features, rolling averages, momentum, and volatility metrics.
- **Machine Learning Models**:
  - Linear Regression (baseline)
  - Random Forest with hyperparameter tuning (TimeSeriesSplit cross‑validation)
  - XGBoost
- **Time Series Forecasting**: ARIMA with confidence intervals, LSTM neural network.
- **Model Evaluation**: RMSE, MAE, R², walk‑forward backtesting.
- **Interactive Dashboard**: Streamlit web app for on‑the‑fly analysis and predictions.
- **Production‑ready Engineering**:
  - Config‑driven (YAML)
  - Modular architecture
  - Centralized logging
  - Model persistence (joblib, HDF5)
  - Exception handling & caching

## Tech Stack
Python, pandas, NumPy, yfinance, scikit‑learn, statsmodels, TensorFlow/Keras, XGBoost, Streamlit, Plotly, mplfinance, PyYAML, joblib.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stock-market-predictor.git
   cd stock-market-predictor
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
