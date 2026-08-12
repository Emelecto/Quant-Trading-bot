"""
Módulo de gestión de riesgo DeepFin.

Reglas acordadas (basadas en evidencia):
  - Stop-Loss = 2 * ATR (adaptativo a la volatilidad real).
  - Take-Profit = R:R 1:3 respecto al SL.
  - Filtro de régimen: ADX < umbral => mercado plano => NO operar.
  - Trailing stop opcional.
  - Tamaño de posición: fijo fraccional 1% del capital por trade.
Todas las funciones son puras y testeables.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from features import indicators as ind


def compute_stop_loss(df: pd.DataFrame, atr_mult: float = 2.0, window: int = 14) -> pd.Series:
    """SL en precio = close - (atr_mult * ATR). Adaptativo a volatilidad."""
    atr_val = ind.atr(df, window)
    return df["close"] - atr_mult * atr_val


def compute_take_profit(entry_price: float, stop_loss: float, rr_ratio: float = 3.0) -> float:
    """TP a R:R 1:rr_ratio respecto a la distancia del SL."""
    risk = entry_price - stop_loss
    return entry_price + rr_ratio * risk


def apply_trailing_stop(
    price_path: pd.Series,
    entry_price: float,
    stop_loss: float,
    rr_ratio: float = 3.0,
) -> tuple[float, int]:
    """Simula un trailing stop sobre una trayectoria de precios.

    Returns:
        (precio_salida, índice_salida). Si no toca SL ni TP, sale al último.
    """
    tp = compute_take_profit(entry_price, stop_loss, rr_ratio)
    highest = entry_price
    trailing_sl = stop_loss
    for i, p in enumerate(price_path.values):
        highest = max(highest, p)
        trailing_sl = max(trailing_sl, highest - (entry_price - stop_loss))
        if p >= tp:
            return float(p), i
        if p <= trailing_sl:
            return float(p), i
    return float(price_path.values[-1]), len(price_path) - 1


def regime_filter(df: pd.DataFrame, adx_threshold: float = 20.0) -> pd.Series:
    """Máscara booleana: True si el mercado tiene tendencia (ADX>=umbral)."""
    adx_val = ind.adx(df, 14)
    return adx_val >= adx_threshold


def position_size(capital: float, risk_pct: float = 0.01, stop_loss_dist: float = 0.0) -> float:
    """Tamaño de posición en unidades del activo (estilo tradicional 1% riesgo)."""
    risk_capital = capital * risk_pct
    if stop_loss_dist > 0:
        return risk_capital / stop_loss_dist
    return risk_capital
