"""
Tests unitarios de los módulos del bot DeepFin.

Usan datos SINTÉTICOS (sin red) para validar que cada componente funcione
en aislamiento: indicadores, features, modelos, ensemble y riesgo.
"""
import numpy as np
import pandas as pd
import pytest


# ---------- Helpers ----------

def make_synthetic_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    # random walk con volatilidad
    rets = rng.normal(0.001, 0.02, n)
    close = 100 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.01, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.01, n)))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    volume = rng.integers(1_000, 10_000, n).astype(float)
    return pd.DataFrame({
        "timestamp": dates,
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    })


# ---------- Indicadores ----------

def test_indicators_run():
    from features import indicators as ind
    df = make_synthetic_ohlcv()
    assert ind.sma(df["close"]).notna().any()
    assert ind.rsi(df["close"]).notna().any()
    assert ind.atr(df).notna().any()
    adx = ind.adx(df)
    assert adx.notna().any()
    assert ((adx.dropna() >= 0) & (adx.dropna() <= 100)).all()


# ---------- Features ----------

def test_build_feature_matrix():
    from features.build_features import build_feature_matrix
    df = make_synthetic_ohlcv()
    feats = build_feature_matrix(df)
    assert "target" in feats.columns
    assert feats["target"].isin([0, 1]).all()
    assert len(feats) > 50


# ---------- Modelos (interfaz común) ----------

def test_models_interface():
    from models.linear_regression_model import LinearRegressionModel
    from models.xgboost_model import XGBoostModel
    from models.monte_carlo_model import MonteCarloModel
    from features.build_features import build_feature_matrix

    df = make_synthetic_ohlcv()
    feats = build_feature_matrix(df)
    X = feats.drop(columns=["target"])
    y = feats["target"]
    close_series = df["close"]

    for ModelCls in [LinearRegressionModel, XGBoostModel, MonteCarloModel]:
        m = ModelCls()
        if ModelCls is MonteCarloModel:
            m._prices = close_series.reindex(X.index)
        m.fit(X, y)
        score = m.predict(X)
        assert isinstance(score, pd.Series)
        assert score.between(-1.01, 1.01).all(), f"{ModelCls.__name__} score fuera de rango"


# ---------- Ensemble ----------

def test_ensemble_methods():
    from ensemble.ensemble_methods import voting, weighted
    idx = pd.date_range("2022-01-01", periods=10, freq="D")
    s1 = pd.Series([0.5, -0.5, 0.3, 0.0, 0.8, -0.9, 0.1, 0.2, -0.3, 0.4], index=idx)
    s2 = pd.Series([0.4, -0.6, 0.2, 0.1, 0.7, -0.8, 0.0, 0.3, -0.2, 0.5], index=idx)
    v = voting({"a": s1, "b": s2})
    assert v.isin([-1, 0, 1]).all()
    w = weighted({"a": s1, "b": s2})
    assert w.notna().all()


# ---------- Riesgo ----------

def test_risk_management():
    from risk import risk_management as rm
    df = make_synthetic_ohlcv(50)
    sl = rm.compute_stop_loss(df)
    valid = sl.dropna()
    assert (valid < df["close"].loc[valid.index]).all()  # SL siempre debajo del precio
    tp = rm.compute_take_profit(100.0, 95.0, 3.0)
    assert tp == 115.0  # 100 + 3*(100-95)
    pos = rm.position_size(1000.0, 0.01, 5.0)
    assert pos == 2.0  # 10 / 5


# ---------- Métricas ----------

def test_metrics():
    from backtest import metrics as M
    r = pd.Series(np.random.default_rng(1).normal(0.001, 0.02, 200))
    eq = (1 + r).cumprod()
    m = M.all_metrics(r, eq)
    for k in ["sharpe", "sortino", "max_drawdown", "win_rate", "profit_factor", "cagr"]:
        assert k in m
    assert m["max_drawdown"] <= 0
    assert 0 <= m["win_rate"] <= 1
