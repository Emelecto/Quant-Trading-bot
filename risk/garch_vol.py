"""
Volatilidad GARCH para stop-loss adaptativo (extra avanzado Fase 7-8).

GARCH(1,1) modela la volatilidad CONDICIONAL del retorno: cuando la volatilidad
está alta, el SL se aleja; cuando está baja, el SL se acerca. Esto mejora el
SL fijo de 2*ATR en regímenes de volatilidad cambiante (típico en crypto).

Requiere statsmodels (ya en requirements.txt).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acovf


def garch_volatility(returns: pd.Series, omega: float = 0.05, alpha: float = 0.10, beta: float = 0.85) -> pd.Series:
    """GARCH(1,1) de forma iterativa: sigma_t^2 = omega + alpha*r_{t-1}^2 + beta*sigma_{t-1}^2.

    Args:
        returns: serie de retornos (ej. close.pct_change()).
        omega, alpha, beta: parámetros GARCH (alpha+beta < 1 para estacionariedad).
    Returns:
        Serie de desviación estándar condicional (volatilidad) alineada a returns.
    """
    r = returns.dropna().astype(float).values
    if len(r) < 2:
        return pd.Series(index=returns.index, dtype=float)
    sigma2 = np.zeros(len(r))
    # inicialización con varianza muestral
    sigma2[0] = np.var(r) if len(r) > 1 else 1e-6
    for t in range(1, len(r)):
        sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]
    vol = np.sqrt(np.maximum(sigma2, 1e-12))
    return pd.Series(vol, index=returns.dropna().index)


def garch_stop_loss(
    df: pd.DataFrame,
    vol_mult: float = 2.0,
    window_case: float = 0.02,
) -> pd.Series:
    """SL basado en volatilidad GARCH en lugar de ATR fijo.

    SL = close - vol_mult * (close * sigma_garch). La distancia se escala a la
    volatilidad condicional del momento. Se acota a un rango sensato [1%, 25%]
    del precio para evitar SL absurdos (ni demasiado cerca ni a -50%).

    Args:
        df: OHLCV con 'close'.
        vol_mult: multiplicador de la volatilidad GARCH.
    """
    from features import indicators as ind
    rets = df["close"].pct_change()
    gvol = garch_volatility(rets)
    # distancia del SL como fracción del precio (volatilidad condicional)
    sl_frac = (vol_mult * gvol).fillna(window_case)
    # acotar a rango sensato: 1% minimo, 25% maximo del precio
    sl_frac = sl_frac.clip(lower=0.01, upper=0.25)
    return df["close"] * (1 - sl_frac)
