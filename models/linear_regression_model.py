"""
Modelo base: Regresión Lineal (estadístico, interpretable).

Predice el retorno futuro continuo a partir de las features técnicas,
luego escala a un score de señal en [-1, +1] con tanh.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from models.base import BaseModel


class LinearRegressionModel(BaseModel):
    name = "linear_regression"

    def __init__(self, feature_cols: list[str] | None = None):
        self.feature_cols = feature_cols
        self.estimator = LinearRegression()
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series, timestamps: pd.Series | None = None) -> "LinearRegressionModel":
        cols = self.feature_cols or [c for c in X.columns if c != "target"]
        self.feature_cols = cols
        self.estimator.fit(X[cols].astype(float), y.values)
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if not self._fitted:
            raise RuntimeError("Debe llamar fit() antes de predict().")
        raw = self.estimator.predict(X[self.feature_cols].astype(float))
        score = np.tanh(raw * 10.0)
        return pd.Series(score, index=X.index, name=self.name)
