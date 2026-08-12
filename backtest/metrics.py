"""
Métricas financieras estándar para evaluar estrategias (modelos y ensemble).

Todas reciben una serie de retornos (ya netos de la señal/posición) y devuelven
un escalar. Implementadas de forma pura y testeable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    """Sharpe Ratio anualizado (retorno ajustado al riesgo total)."""
    r = returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return 0.0
    excess = r - risk_free / periods_per_year
    return float(np.sqrt(periods_per_year) * excess.mean() / r.std())


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    """Sortino Ratio anualizado (solo riesgo a la baja)."""
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    downside = r[r < 0]
    dd = downside.std()
    if dd == 0:
        return 0.0
    excess = r - risk_free / periods_per_year
    return float(np.sqrt(periods_per_year) * excess.mean() / dd)


def max_drawdown(equity_curve: pd.Series) -> float:
    """Máxima caída desde un pico histórico (como fracción negativa)."""
    eq = equity_curve.dropna()
    if len(eq) == 0:
        return 0.0
    running_max = eq.cummax()
    drawdown = eq / running_max - 1.0
    return float(drawdown.min())


def win_rate(trades_returns: pd.Series) -> float:
    """Proporción de operaciones ganadoras."""
    r = trades_returns.dropna()
    if len(r) == 0:
        return 0.0
    return float((r > 0).mean())


def profit_factor(trades_returns: pd.Series) -> float:
    """Ratio de ganancias / pérdidas absolutas."""
    r = trades_returns.dropna()
    if len(r) == 0:
        return 0.0
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def cagr(equity_curve: pd.Series) -> float:
    """Tasa de crecimiento anual compuesta."""
    eq = equity_curve.dropna()
    if len(eq) < 2 or eq.iloc[0] <= 0:
        return 0.0
    years = len(eq) / 252.0
    if years <= 0:
        return 0.0
    return float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1)


def all_metrics(returns: pd.Series, equity_curve: pd.Series | None = None) -> dict:
    """Devuelve un diccionario con todas las métricas estándar."""
    eq = equity_curve if equity_curve is not None else (1 + returns.fillna(0)).cumprod()
    return {
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "max_drawdown": max_drawdown(eq),
        "win_rate": win_rate(returns),
        "profit_factor": profit_factor(returns),
        "cagr": cagr(eq),
    }
