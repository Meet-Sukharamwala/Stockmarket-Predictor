# src/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mplfinance as mpf

def plot_candlestick(data: pd.DataFrame, symbol: str, ma_windows=[20, 50]):
    """Professional candlestick chart with volume and moving averages using mplfinance."""
    df = data[['Open','High','Low','Close','Volume']].copy()
    mpf.plot(df, type='candle', style='charles', volume=True,
             mav=ma_windows, title=f'{symbol} Candlestick Chart',
             savefig=f'candlestick_{symbol}.png')
    plt.close()

def plot_candlestick_plotly(data: pd.DataFrame, symbol: str, ma_windows=[20, 50]):
    """Interactive candlestick using Plotly."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=data.index, open=data.Open, high=data.High,
                                 low=data.Low, close=data.Close, name='OHLC'), row=1, col=1)
    for w in ma_windows:
        data[f'ma_{w}'] = data.Close.rolling(w).mean()
        fig.add_trace(go.Scatter(x=data.index, y=data[f'ma_{w}'], mode='lines', name=f'MA {w}'), row=1, col=1)
    fig.add_trace(go.Bar(x=data.index, y=data.Volume, name='Volume'), row=2, col=1)
    fig.update_layout(title=f'{symbol} Interactive Candlestick', xaxis_rangeslider_visible=False)
    return fig

def plot_correlation_heatmap(data: pd.DataFrame):
    """Plot correlation heatmap of numeric features."""
    plt.figure(figsize=(10, 8))
    numeric = data.select_dtypes(include=[np.number])
    sns.heatmap(numeric.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    return plt.gcf()

# Additional functions for daily returns, volatility, histograms...
