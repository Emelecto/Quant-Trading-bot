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


def directional_accuracy(signal: pd.Series, future_return: pd.Series) -> float:
    """Accuracy direccional fuera de muestra: % de veces que sign(senal)==sign(retorno)."""
    common = signal.dropna().index.intersection(future_return.dropna().index)
    if len(common) < 5:
        return 0.5
    return float((np.sign(signal.loc[common]) == np.sign(future_return.loc[common])).mean())


def compute_accuracy_weights(
    scores_by_model: dict[str, pd.Series],
    future_returns: pd.Series,
) -> dict[str, float]:
    """Pesos por accuracy direccional OOS de cada modelo en la ventana train.

    Los modelos con accuracy ~0.5 (ruido) reciben peso ~0; los que predicen
    bien (XGBoost) reciben peso cercano a 1. Esto EVITA diluir la señal útil
    con modelos inútiles (el error original del ensemble equal-weight).
    """
    accs = {name: directional_accuracy(s, future_returns) for name, s in scores_by_model.items()}
    # premium sobre 0.5 (mitad del azar): (acc - 0.5) / 0.5  -> [0, 1]
    prem = {k: max((v - 0.5) / 0.5, 0.0) for k, v in accs.items()}
    total = sum(prem.values()) or 1.0
    return {k: v / total for k, v in prem.items()}


def accuracy_weighted(
    signals: dict[str, pd.Series],
    future_returns: pd.Series,
    scores_by_model: dict[str, pd.Series] | None = None,
) -> pd.Series:
    """Promedio ponderado por accuracy direccional OOS (ver compute_accuracy_weights)."""
    sbm = scores_by_model if scores_by_model is not None else signals
    w = compute_accuracy_weights(sbm, future_returns)
    return weighted(signals, w)
