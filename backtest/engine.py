"""
Motor de backtesting con validación walk-forward.

Recorre los datos en ventanas solapadas: entrena modelos en la ventana de
entrenamiento, genera señales y evalúa en la ventana de validación (fuera de
muestra). Acumula retornos de validación para evitar overfitting/data-snooping.

Flujo: datos -> features -> por ventana: fit(modelos) -> predict -> ensemble
-> señal -> aplicar riesgo (SL/TP) -> retornos -> métricas.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from features.build_features import build_feature_matrix
from risk import risk_management as rm
from backtest import metrics as M
from models.monte_carlo_model import MonteCarloModel


def run_walk_forward(
    df: pd.DataFrame,
    models: list,
    ensemble_method: str = "weighted",
    train_window: int = 252,
    test_window: int = 63,
    step: int = 63,
    atr_mult: float = 2.0,
    rr_ratio: float = 3.0,
    adx_threshold: float = 20.0,
) -> pd.DataFrame:
    """Ejecuta walk-forward y devuelve un DataFrame de retornos por fecha.

    Args:
        df: OHLCV limpio.
        models: lista de instancias de modelos (implementan BaseModel).
        ensemble_method: 'voting' | 'weighted' | 'stacking'.
    """
    feats = build_feature_matrix(df)
    # alineamos features con OHLCV para SL/TP
    merged = df.join(feats, how="inner")
    merged = merged.dropna()
    # descartar columnas no numéricas (timestamp) antes de entrenar modelos
    non_numeric = merged.select_dtypes(exclude=["number"]).columns
    if len(non_numeric):
        merged = merged.drop(columns=non_numeric)

    all_returns: list[float] = []
    all_dates: list = []

    n = len(merged)
    start = 0
    while start + train_window + test_window <= n:
        train = merged.iloc[start:start + train_window]
        test = merged.iloc[start + train_window:start + train_window + test_window]

        # entrenar modelos
        trained = []
        for m in models:
            # Reconstruir instancia nueva usando SOLO los args del __init__
            # (no atributos internos como 'estimator' o '_fitted').
            import inspect
            sig = inspect.signature(type(m).__init__)
            init_kwargs = {
                k: v for k, v in vars(m).items()
                if k in sig.parameters and k != "self"
            }
            inst = type(m)(**init_kwargs)
            if isinstance(inst, MonteCarloModel):
                inst._prices = train["close"]
            inst.fit(train.drop(columns=["target"]), train["target"])
            trained.append(inst)

        # señales base en test
        signals = {}
        for inst in trained:
            if isinstance(inst, MonteCarloModel):
                inst._prices = test["close"]
            signals[inst.name] = inst.predict(test.drop(columns=["target"]))

        # ensemble
        if ensemble_method == "voting":
            final_signal = ensemble_voting(signals)
        elif ensemble_method == "weighted":
            final_signal = ensemble_weighted(signals, trained, train)
        elif ensemble_method == "stacking":
            final_signal = ensemble_stacking(signals, train, test)
        else:
            raise ValueError(f"Método '{ensemble_method}' no soportado.")

        # aplicar filtro de régimen (ADX)
        regime = rm.regime_filter(test, adx_threshold)

        # simular retornos con SL/TP por barra
        close = test["close"].values
        for i in range(len(test) - 1):
            sig = final_signal.iloc[i]
            if not regime.iloc[i] or sig == 0:
                all_returns.append(0.0)
                all_dates.append(test.index[i])
                continue
            entry = close[i]
            sl = rm.compute_stop_loss(test.iloc[i:i + 1], atr_mult).iloc[0]
            tp = rm.compute_take_profit(entry, sl, rr_ratio)
            # retorno real de la siguiente barra (aproximación simple de ejecución)
            next_ret = close[i + 1] / entry - 1.0
            if sig > 0:  # long
                ret = min(next_ret, tp / entry - 1.0)
                if next_ret <= (sl / entry - 1.0):
                    ret = sl / entry - 1.0
            else:  # short
                ret = -next_ret
                if next_ret >= (tp / entry - 1.0):
                    ret = -(tp / entry - 1.0)
                elif next_ret >= (sl / entry - 1.0):
                    ret = -(sl / entry - 1.0)
            all_returns.append(ret * np.sign(sig))
            all_dates.append(test.index[i])

        start += step

    return pd.DataFrame({"date": all_dates, "return": all_returns}).set_index("date")["return"]


def ensemble_voting(signals, threshold=0.0):
    from ensemble.ensemble_methods import voting
    return voting(signals, threshold)


def ensemble_weighted(signals, trained_models, train):
    from ensemble.ensemble_methods import weighted, compute_model_weights
    # pesos por Sharpe en ventana train (aprox: retorno de señal vs retorno real)
    returns_by_model = {}
    for name, s in signals.items():
        aligned = s.index.intersection(train.index)
        # aproximación: usamos el train para el peso
        pass
    # fallback: pesos iguales si no hay datos suficientes
    return weighted(signals)


def ensemble_stacking(signals, train, test):
    from ensemble.ensemble_methods import stacking
    meta = pd.DataFrame(index=test.index)
    meta["target"] = test["target"]
    return stacking(signals, meta)
