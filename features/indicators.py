"""
Indicadores técnicos y features para el bot DeepFin.

Todas las funciones son puras (entrant pandas Series/DataFrame, salen pandas).
No introducen look-ahead: cada valor en la fila t usa solo info hasta t.
Incluye ATR (para stop-loss adaptativo) y ADX (para filtro de régimen).
"""
from __future__ import annotations

import pandas as pd
import numpy as np


def sma(series: pd.Series, window: int = 20) -> pd.Series:
    return series.rolling(window).mean()


def ema(series: pd.Series, window: int = 20) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger(series: pd.Series, window: int = 20, num_std: float = 2.0):
    mid = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    pct_b = (series - lower) / (upper - lower + 1e-9)
    return upper, lower, pct_b


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm = (high - high.shift(1)).clip(lower=0)
    minus_dm = (low.shift(1) - low).clip(lower=0)
    tr = atr(df, 1)
    plus_di = 100 * (plus_dm.rolling(window).mean() / (tr.rolling(window).mean() + 1e-9))
    minus_di = 100 * (minus_dm.rolling(window).mean() / (tr.rolling(window).mean() + 1e-9))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    return dx.rolling(window).mean()


def rolling_returns(close: pd.Series, window: int = 5) -> pd.Series:
    return close.pct_change(window)


def rolling_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    return close.pct_change().rolling(window).std()
