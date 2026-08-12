"""
Modelo base: XGBoost (machine learning, captura no-linealidades).

Entrena un clasificador sobre la variable target (dirección del retorno).
La probabilidad P(suba) se convierte en score continuo: score = 2*P - 1 => [-1, +1].
"""
from __future__ import annotations

import pandas as pd
import xgboost as xgb

from models.base import BaseModel


class XGBoostModel(BaseModel):
    name = "xgboost"

    def __init__(self, feature_cols: list[str] | None = None, random_state: int = 42):
        self.feature_cols = feature_cols
        self.random_state = random_state
        self.estimator = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic",
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        )
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series, timestamps: pd.Series | None = None) -> "XGBoostModel":
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
