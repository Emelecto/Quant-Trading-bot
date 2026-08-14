"""
Modelo base: CatBoost (machine learning, árbol de decisión gradiente).

Similar a XGBoost pero suele manejar mejor categorías y es robusto sin
mucha tuning. Lo usamos como 2º árbol para diversificar el ensemble
(si ambos árboles coinciden en señal, refuerza; si difieren, aporta contraste).

Devuelve score continuo en [-1, +1] = 2*P(suba) - 1.
"""
from __future__ import annotations

import pandas as pd
from catboost import CatBoostClassifier

from models.base import BaseModel


class CatBoostModel(BaseModel):
    name = "catboost"

    def __init__(self, feature_cols: list[str] | None = None, random_state: int = 42):
        self.feature_cols = feature_cols
        self.random_state = random_state
        self.estimator = CatBoostClassifier(
            iterations=300,
            depth=4,
            learning_rate=0.05,
            l2_leaf_reg=3.0,           # regularización (anti-overfit)
            loss_function="Logloss",
            random_state=random_state,
            verbose=False,
            allow_writing_files=False,
        )
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series, timestamps: pd.Series | None = None) -> "CatBoostModel":
        cols = self.feature_cols or [c for c in X.columns if c != "target"]
        self.feature_cols = cols
        self.estimator.fit(X[cols].astype(float), y.values)
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if not self._fitted:
            raise RuntimeError("Debe llamar fit() antes de predict().")
        proba_up = self.estimator.predict_proba(X[self.feature_cols].astype(float))[:, 1]
        score = 2.0 * proba_up - 1.0
        return pd.Series(score, index=X.index, name=self.name)
