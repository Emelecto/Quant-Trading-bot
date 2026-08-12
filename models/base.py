"""
Interfaz base común para TODOS los modelos del ensemble.

Cada modelo (Regresión Lineal, XGBoost, Monte Carlo, LSTM, CatBoost, etc.)
DEBE heredar de BaseModel e implementar:
  - fit(X, y, timestamps): entrena con datos de entrenamiento.
  - predict(X): devuelve señal continua en [-1, +1] (score de dirección).
  - name: identificador legible.

Esto habilita que el módulo ensemble los combine sin saber su implementación
interna, y que varias personas del equipo trabajen módulos distintos.

IMPORTANTE: se prohibe el look-ahead. predict solo debe usar info hasta t.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd


class BaseModel(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, timestamps: pd.Series | None = None) -> "BaseModel":
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def predict_direction(self, X: pd.DataFrame, threshold: float = 0.0) -> pd.Series:
        score = self.predict(X)
        return (score > threshold).astype(int) - (score < -threshold).astype(int)
