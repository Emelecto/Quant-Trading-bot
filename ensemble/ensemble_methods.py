"""
Módulos de ensemble: combinan las señales de los modelos base.

Orden evaluado en walk-forward (según especificación):
  1. voting   - mayoría simple de direcciones {-1,0,+1}.
  2. weighted  - promedio ponderado por Sharpe histórico (principal).
  3. stacking  - meta-modelo (solo si supera claramente a 'weighted').

Cada función recibe un dict {model_name: score_series} y devuelve señal final.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def voting(signals: dict[str, pd.Series], threshold: float = 0.0) -> pd.Series:
    """Voting por mayoría: cuenta direcciones discretas y toma la mayoría."""
    mats = []
    for s in signals.values():
        mats.append((s > threshold).astype(int) - (s < -threshold).astype(int))
    stacked = pd.concat(mats, axis=1)
    final = stacked.sum(axis=1)
    return (final > 0).astype(int) - (final < 0).astype(int)


def weighted(
    signals: dict[str, pd.Series],
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """Promedio ponderado de scores continuos.

    Si no se dan pesos, usa iguales. Los pesos pueden ser Sharpe histórico
    normalizado (ver ensemble.compute_model_weights).
    """
    if weights is None:
        weights = {k: 1.0 / len(signals) for k in signals}
    total_w = sum(weights.values())
    acc = None
    for name, s in signals.items():
        w = weights.get(name, 0.0) / total_w
        contrib = s * w
        acc = contrib if acc is None else acc + contrib
    return acc


def stacking(
    signals: dict[str, pd.Series],
    meta_features: pd.DataFrame | None = None,
) -> pd.Series:
    """Stacking como meta-modelo lineal sobre las señales base.

    Meta-modelo: regresión logística que aprende a combinar los scores base
    hacia la dirección real. Requiere 'meta_features' con columna 'target'.
    """
    from sklearn.linear_model import LogisticRegression
    X = pd.DataFrame(signals)
    if meta_features is None or "target" not in meta_features.columns:
        raise ValueError("Stacking requiere meta_features con columna 'target'.")
    y = meta_features["target"]
    common = X.dropna().index.intersection(y.index)
    model = LogisticRegression()
    model.fit(X.loc[common], y.loc[common])
    proba = model.predict_proba(X)[:, 1]
    return pd.Series(2.0 * proba - 1.0, index=X.index, name="stacking")


def compute_model_weights(
    scores_by_model: dict[str, pd.Series],
    returns_by_model: dict[str, pd.Series],
    metric: str = "sharpe",
) -> dict[str, float]:
    """Calcula pesos a partir de una métrica de desempeño fuera de muestra."""
    from backtest.metrics import sharpe_ratio
    raw = {}
    for name in scores_by_model:
        if metric == "sharpe":
            raw[name] = sharpe_ratio(returns_by_model[name])
        else:
            raise ValueError(f"Métrica '{metric}' no soportada.")
    pos = {k: max(v, 0.0) for k, v in raw.items()}
    total = sum(pos.values()) or 1.0
    return {k: v / total for k, v in pos.items()}
