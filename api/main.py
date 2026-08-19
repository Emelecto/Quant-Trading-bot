"""
API FastAPI mínima para el bot DeepFin (paper trading).

Expone endpoints para consultar señales del ensemble y métricas de backtest.
En MVP la API no ejecuta órdenes reales; sirve para alimentar el dashboard
y futuras integraciones. CORS abierto para desarrollo local.
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import fetch_data as fd
from features.build_features import build_feature_matrix
from models.linear_regression_model import LinearRegressionModel
from models.xgboost_model import XGBoostModel
from models.monte_carlo_model import MonteCarloModel
from ensemble.ensemble_methods import weighted
from backtest import metrics as M

app = FastAPI(title="DeepFin Trading Bot API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "mode": "paper_trading"}


@app.get("/signal/{symbol}")
def get_signal(symbol: str = "BTC/USDT"):
    """Devuelve la señal actual del ensemble (score en [-1, +1])."""
    try:
        df = fd.ensure_raw(symbol)
    except FileNotFoundError:
        return {"error": f"Sin datos para {symbol}. Ejecuta data.fetch_data"}
    df = fd.clean_ohlcv(df)
    feats = build_feature_matrix(df)
    X = feats.drop(columns=["target"])
    y = feats["target"]
    close = df["close"]

    models = [LinearRegressionModel(), XGBoostModel(), MonteCarloModel(n_paths=300)]
    signals = {}
    for m in models:
        if m.name == "monte_carlo":
            m._prices = close.reindex(X.index)
        m.fit(X, y)
        signals[m.name] = m.predict(X).iloc[-1]
    final = weighted(signals).iloc[-1]
    return {
        "symbol": symbol,
        "ensemble_score": round(float(final), 4),
        "direction": "buy" if final > 0 else ("sell" if final < 0 else "hold"),
        "components": {k: round(float(v), 4) for k, v in signals.items()},
    }


@app.get("/metrics/{symbol}")
def get_metrics(symbol: str = "BTC/USDT"):
    """Métricas de Buy & Hold (baseline) para comparación rápida."""
    try:
        df = fd.ensure_raw(symbol)
    except FileNotFoundError:
        return {"error": f"Sin datos para {symbol}."}
    df = fd.clean_ohlcv(df)
    rets = df["close"].pct_change().dropna()
    eq = (1 + rets).cumprod()
    return {"symbol": symbol, "buy_and_hold": {k: round(float(v), 4) for k, v in M.all_metrics(rets, eq).items()}}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
