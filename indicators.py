# src/indicators.py
import pandas as pd
import numpy as np

def simple_moving_average(series: pd.Series, window: int = 20) -> pd.Series:
    """SMA: sum of last N closing prices divided by N.
    Use: Smooths price data to identify trend direction.
    Interpretation: Price above SMA → uptrend; below → downtrend."""
    return series.rolling(window).mean()

def exponential_moving_average(series: pd.Series, span: int = 20) -> pd.Series:
    """EMA: weighted moving average giving more importance to recent prices.
    Use: Faster reaction to price changes.
    Interpretation: Crosses of short EMA above long EMA signal buy opportunities."""
    return series.ewm(span=span, adjust=False).mean()

def relative_strength_index(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI = 100 - (100 / (1 + RS)), where RS = avg gain / avg loss over period.
    Use: Momentum oscillator measuring speed and change of price movements.
    Interpretation: >70 overbought, <30 oversold."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    """MACD: difference between fast EMA and slow EMA.
    Signal line: EMA of MACD.
    Use: Identifies momentum changes.
    Interpretation: MACD crossing above signal → bullish; below → bearish.
    Returns DataFrame with MACD, Signal, Histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({'MACD': macd_line, 'Signal': signal_line, 'Histogram': histogram}, index=series.index)

def bollinger_bands(series: pd.Series, window=20, num_std=2):
    """Bollinger Bands: middle = SMA, upper/lower = middle ± num_std * rolling std.
    Use: Measure volatility and overbought/oversold conditions.
    Interpretation: Price near upper band → potentially overbought; lower → oversold."""
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return pd.DataFrame({'Middle Band': sma, 'Upper Band': upper, 'Lower Band': lower}, index=series.index)
