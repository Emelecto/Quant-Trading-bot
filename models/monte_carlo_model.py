"""
Modelo base: Monte Carlo (capa PROBABILÍSTICA / RIESGO, no vota dirección).

Estima, vía simulación GBM, la probabilidad de que el precio supere un umbral
alcista en el horizonte, y devuelve un score suave en [-1, +1].

Uso en el ensemble: aporta una perspectiva puramente estocástica que los modelos
deterministas no ven. Se combina en el promedio ponderado / stacking.
No hay look-ahead: usa solo media y volatilidad histórica hasta t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from models.base import BaseModel


class MonteCarloModel(BaseModel):
    name = "monte_carlo"

    def __init__(self, n_paths: int = 1000, random_state: int = 42, horizon: int = 1):
        self.n_paths = n_paths
        self.random_state = random_state
        self.horizon = horizon
        self._mu = None
        self._sigma = None
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series, timestamps: pd.Series | None = None) -> "MonteCarloModel":
        # Monte Carlo necesita la serie de precios 'close'. La acepta ya sea
        # como columna de X, o como atributo inyectado por el engine (self._prices).
        if "close" in X.columns:
            close = X["close"].astype(float)
        elif getattr(self, "_prices", None) is not None:
            close = self._prices.astype(float)
        else:
            raise ValueError("MonteCarlo requiere columna 'close' en X o self._prices inyectado.")
        rets = close.pct_change().dropna()
        self._mu = rets.mean()
        self._sigma = rets.std()
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if not self._fitted:
            raise RuntimeError("Debe llamar fit() antes de predict().")
        if getattr(self, "_prices", None) is not None:
            prices = self._prices.astype(float)
        elif "close" in X.columns:
            prices = X["close"].astype(float)
        else:
            raise ValueError("MonteCarlo requiere 'close' para predecir.")
        rng = np.random.default_rng(self.random_state)
        scores = []
        for price in prices.values:
            z = rng.standard_normal(self.n_paths)
            drift = (self._mu - 0.5 * self._sigma**2) * self.horizon
            diffusion = self._sigma * np.sqrt(self.horizon) * z
            simulated = price * np.exp(drift + diffusion)
            p_up = np.mean(simulated > price)
            scores.append(2.0 * p_up - 1.0)
        return pd.Series(scores, index=X.index, name=self.name)
