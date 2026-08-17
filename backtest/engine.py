"""
Motor de backtesting con validación walk-forward.

Recorre los datos en ventanas solapadas: entrena modelos en la ventana de
entrenamiento, genera señales y evalúa en la ventana de validación (fuera de
muestra). Acumula retornos de validación para evitar overfitting/data-snooping.

Mejoras Fase 7-8:
  - Devuelve retornos INDIVIDUALES de cada modelo (para comparación head-to-head).
  - El ensemble 'weighted' usa pesos por Sharpe fuera de muestra por ventana.
  - Soporta SL basado en GARCH (volatilidad adaptativa) además de ATR.

Flujo: datos -> features -> por ventana: fit(modelos) -> predict -> ensemble
-> señal -> aplicar riesgo (SL/TP) -> retornos -> métricas.
"""
from __future__ import annotations

import inspect
import pandas as pd
import numpy as np

from features.build_features import build_feature_matrix
from risk import risk_management as rm
from backtest import metrics as M
from models.monte_carlo_model import MonteCarloModel


def _simulate_returns_from_signal(
    final_signal: pd.Series,
    close: np.ndarray,
    sl_series: pd.Series,
    regime: pd.Series,
    rr_ratio: float = 3.0,
) -> list[float]:
    """Dada una señal de ensemble por barra, simula retornos con SL/TP.

    Aproximación de ejecución: usa el retorno real de la siguiente barra,
    truncado por SL/TP según la dirección.
    """
    out = []
    sl_vals = sl_series.values
    for i in range(len(final_signal) - 1):
        sig = final_signal.iloc[i]
        if not regime.iloc[i] or sig == 0:
            out.append(0.0)
            continue
        entry = close[i]
        sl = sl_vals[i]
        tp = rm.compute_take_profit(entry, sl, rr_ratio)
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
        out.append(ret * np.sign(sig))
    return out


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
    use_garch_sl: bool = False,
    return_individual: bool = False,
    feature_df: pd.DataFrame | None = None,
):
    """Ejecuta walk-forward y devuelve retornos (ensemble y, opcionalmente, individuales).

    Args:
        feature_df: matriz de features precomputada (ej. horizonte mensual).
            Si es None, se usa build_feature_matrix(df) por defecto.
    """
    feats = feature_df if feature_df is not None else build_feature_matrix(df)
    merged = df.join(feats, how="inner")
    merged = merged.dropna()
    non_numeric = merged.select_dtypes(exclude=["number"]).columns
    if len(non_numeric):
        merged = merged.drop(columns=non_numeric)

    # buffers
    ens_returns: list[float] = []
    ens_dates: list = []
    indiv_buffers = {m.name: [] for m in models}
    indiv_dates = {m.name: [] for m in models}

    n = len(merged)
    start = 0
    while start + train_window + test_window <= n:
        train = merged.iloc[start:start + train_window]
        test = merged.iloc[start + train_window:start + train_window + test_window]

        # entrenar modelos
        trained = []
        for m in models:
            sig = inspect.signature(type(m).__init__)
            init_kwargs = {k: v for k, v in vars(m).items() if k in sig.parameters and k != "self"}
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

        # retorno futuro de train (para pesos por accuracy/dir de la ventana train)
        train_fwd = train["close"].pct_change().shift(-1)

        # ensemble
        if ensemble_method == "voting":
            final_signal = ensemble_voting(signals)
        elif ensemble_method == "weighted":
            final_signal = ensemble_weighted(signals, trained, train, test)
        elif ensemble_method == "accuracy":
            final_signal = ensemble_accuracy_weighted(signals, trained, train, train_fwd)
        elif ensemble_method == "stacking":
            final_signal = ensemble_stacking(signals, train, test)
        else:
            raise ValueError(f"Método '{ensemble_method}' no soportado.")

        # SL por barra (ATR o GARCH)
        if use_garch_sl:
            from risk.garch_vol import garch_stop_loss
            sl_series = garch_stop_loss(test, vol_mult=atr_mult)
        else:
            sl_series = rm.compute_stop_loss(test, atr_mult)

        regime = rm.regime_filter(test, adx_threshold)
        close = test["close"].values

        # retornos del ensemble
        ens_chunk = _simulate_returns_from_signal(final_signal, close, sl_series, regime, rr_ratio)
        ens_returns.extend(ens_chunk)
        ens_dates.extend(test.index[:len(ens_chunk)])

        # retornos individuales (señal propia de cada modelo, mismo riesgo)
        if return_individual:
            for inst in trained:
                name = inst.name
                indiv_chunk = _simulate_returns_from_signal(signals[name], close, sl_series, regime, rr_ratio)
                indiv_buffers[name].extend(indiv_chunk)
                indiv_dates[name].extend(test.index[:len(indiv_chunk)])

        start += step

    ens_series = pd.Series(ens_returns, index=ens_dates, name="ensemble")
    if not return_individual:
        return ens_series

    indiv_series = {name: pd.Series(buf, index=indiv_dates[name], name=name) for name, buf in indiv_buffers.items()}
    return {"ensemble": ens_series, "models": indiv_series}


def ensemble_voting(signals, threshold=0.0):
    from ensemble.ensemble_methods import voting
    return voting(signals, threshold)


def ensemble_weighted(signals, trained_models, train, test):
    """Promedio ponderado por Sharpe fuera de muestra de la ventana train.

    Calcula el Sharpe de la señal de cada modelo SOBRE LA VENTANA TRAIN
    (señal de train vs retorno real de train) y pondera la señal de TEST.
    Si no hay datos suficientes, cae a pesos iguales.
    """
    from ensemble.ensemble_methods import weighted, compute_model_weights
    train_ret = train["close"].pct_change().shift(-1).reindex(train.index).fillna(0)
    # señales de los modelos ya entrenados, evaluadas en train
    train_signals = {}
    for inst in trained_models:
        if isinstance(inst, MonteCarloModel):
            inst._prices = train["close"]
        train_signals[inst.name] = inst.predict(train.drop(columns=["target"]))
    returns_by_model = {}
    for name, s in train_signals.items():
        aligned = s.index.intersection(train_ret.index)
        if len(aligned) > 5:
            returns_by_model[name] = s.loc[aligned] * train_ret.loc[aligned]
    if returns_by_model:
        weights = compute_model_weights(train_signals, returns_by_model, metric="sharpe")
        return weighted(signals, weights)
    return weighted(signals)


def ensemble_accuracy_weighted(signals, trained_models, train, train_fwd):
    """Promedio ponderado por accuracy direccional OOS de la ventana train.

    Evita diluir la señal útil (XGBoost) con modelos de ruido (LR/Monte Carlo):
    pesa cada modelo por cuánto supera 50% de acierto direccional en train.
    """
    from ensemble.ensemble_methods import accuracy_weighted
    # señales de los modelos ya entrenados, evaluadas en train
    train_signals = {}
    for inst in trained_models:
        if isinstance(inst, MonteCarloModel):
            inst._prices = train["close"]
        train_signals[inst.name] = inst.predict(train.drop(columns=["target"]))
    return accuracy_weighted(signals, train_fwd, scores_by_model=train_signals)


def ensemble_stacking(signals, train, test):
    from ensemble.ensemble_methods import stacking
    meta = pd.DataFrame(index=test.index)
    meta["target"] = test["target"]
    return stacking(signals, meta)
