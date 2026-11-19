# dashboard/app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.indicators import compute_all_indicators  # wrapper to add indicators to df
from src.visualization import plot_candlestick_plotly
from src.forecasting import load_lstm_model, load_sklearn_model
from src.preprocessing import Preprocessor
from src.feature_engineering import FeatureEngineer
import joblib

st.set_page_config(layout="wide")
st.title("📈 Stock Market Prediction & Analytics System")
symbol = st.sidebar.text_input("Stock Symbol", value="AAPL")
data_period = st.sidebar.selectbox("Period", ["1y", "2y", "5y", "ytd", "max"])

@st.cache_data
def load_data(symbol, period):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)
    if df.empty:
        st.error("No data found.")
        return None
    return df

df = load_data(symbol, data_period)
if df is not None:
    st.subheader(f"{symbol} Historical Data")
    st.dataframe(df.tail())

    # Candlestick chart
    fig = plot_candlestick_plotly(df, symbol)
    st.plotly_chart(fig, use_container_width=True)

    # Technical Indicators
    st.subheader("Technical Indicators")
    if st.checkbox("Show RSI"):
        from src.indicators import relative_strength_index
        df['RSI'] = relative_strength_index(df['Close'])
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'))
        fig_rsi.add_hline(y=70, line_dash="dot", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dot", line_color="green")
        st.plotly_chart(fig_rsi)

    # Predictions
    st.subheader("ML Predictions")
    model_option = st.selectbox("Choose model", ["Random Forest", "LSTM"])
    if st.button("Predict Next Day"):
        # Simplified: Load model, process latest data and show prediction
        # In a real dashboard, we'd run preprocessing and feature engineering on the fly.
        if model_option == "Random Forest":
            try:
                rf = load_sklearn_model("random_forest")
                # Here we'd prepare features from the last row... (omitted for brevity)
                st.write(f"Predicted next close: $150.23 (demo)")
            except Exception as e:
                st.error(f"Model not found. Train first. Error: {e}")
        else:
            try:
                lstm_model = load_lstm_model("lstm_model")
                # Prepare last 60 days...
                st.write(f"LSTM predicted next close: $148.90 (demo)")
            except Exception as e:
                st.error(f"LSTM model not found. Error: {e}")

    st.subheader("Forecast with ARIMA")
    # Display ARIMA forecast chart (pre-computed or on-demand)
    # ...
